#include "PassDetails.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "MemAcc/Passes/Passes.h"

#define DEBUG_TYPE "memory-access-to-llvm"

using namespace mlir;
using namespace mlir::MemAcc;

Type convertMemrefElementTypeForLLVMPointer(
    MemRefType type, const LLVMTypeConverter &converter);

namespace {
struct MemAccToLLVMPass
    : public MemAccToLLVMBase<MemAccToLLVMPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace{

static Value traceIntegerType(Value size, ConversionPatternRewriter &rewriter) {
  auto intType = rewriter.getI64Type();
  // Check if the type is already i64
  if (size.getType().isa<IntegerType>()) {
    return size;
  } else {
    // Trace back the origin of the size if it's not an i64
    if (auto castOp = size.getDefiningOp<UnrealizedConversionCastOp>()) {
      Value original = castOp.getOperand(0);
      // Check if the original value is of i64 type
      if (original.getType().isa<IntegerType>() && original.getType().cast<IntegerType>().getWidth() == 64) {
        return original;
      } else {
        // If still not i64, add a cast to i64, this is a fallback and might not be semantically correct
        auto castedSize = rewriter.create<LLVM::SExtOp>(size.getLoc(), intType, original);
        return castedSize;
      }
    } else {
      // If we can't trace back to a cast operation, we cast directly here (fallback)
      auto castedSize = rewriter.create<LLVM::SExtOp>(size.getLoc(), intType, size);
      return castedSize;
    }
  }
}

static Value tracePtrType(Value ptr, ConversionPatternRewriter &rewriter) {
  auto type = ptr.getType();
  // Check if the type is already i64
  if (type.isa<LLVM::LLVMPointerType>()) {
    return ptr;
  } else {
    // Trace back the origin of the size if it's not an i64
    if (auto castOp = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      Value original = castOp.getOperand(0);
      // Check if the original value is of i64 type
      return original;
    } else {
      assert(false && "Unsupported type for tracing");
    }
  }
}

class PackedGenericLoadOpLowering : public ConvertOpToLLVMPattern<PackedGenericLoadOp> {
  using ConvertOpToLLVMPattern<PackedGenericLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PackedGenericLoadOp load, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    /*
    "memacc.packed_generic_load"(%alloc_spd, %0) <{indirectionLevel = 1 : i64, lowerBoundMap = #map, operandSegmentSizes = array<i32: 1, 0, 1, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg4: index):
        %1 = memacc.load %arg2[%arg4] : memref<?xi32>
        %2 = memacc.index_cast %1 : i32 to index
        %3 = memacc.load %arg1[%2] : memref<?xf64>
        %4 = memacc.yield %3 : (f64) -> f64
      }) : (memref<?xf64>, index) -> ()

      will be transformed into

      "llvm.maa.setloopbound"(0, %0)
      "llvm.maa.setloopstep"(1)
      "llvm.maa.setindirectionlevel"(1)
      "llvm.maa.setdataptr(%arg2)" // set the data pointer for indices
      "llvm.maa.setspdptr(%arg1)" // set the data pointer for final data that would be stored into spd
    */

    auto loc = load.getLoc();

    //now assuming only one operand is used for lower bound and upper bound
    //TODO: Consider more complicated cases where affine map is non-trivial
    int constLowerBound = 0;
    Value lowerBound;
    if (load.hasConstantLowerBound()){
      constLowerBound = load.getConstantLowerBound();
      // create llvm constant for lower bound
      lowerBound = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), constLowerBound));
    } else {
      lowerBound = traceIntegerType(load.getLowerBoundOperands()[0], rewriter);
    }

    int constUpperBound = 0;
    Value upperBound;
    if (load.hasConstantUpperBound()){
      constUpperBound = load.getConstantUpperBound();
      // create llvm constant for upper bound
      upperBound = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), constUpperBound));
    } else {
      upperBound = traceIntegerType(load.getUpperBoundOperands()[0], rewriter);
    }

    //set step size
    auto step = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), load.getStepAsAPInt());
    
    //set loop configuration
    auto iter = rewriter.create<LLVM::MAA_SetLoopOp>(loc, rewriter.getI32Type(), lowerBound, upperBound, step).getResult();

    //convert internal memacc.load to llvm load intrinsics
    int ins_idx = 0;
    for (auto& I : load.getBody().front()){
      if (isa<MemAcc::LoadOp>(I)){
        if (ins_idx == 0){
          //set data pointer for indices
          auto dataPtr = tracePtrType(I.getOperand(0), rewriter);
          iter = rewriter.create<LLVM::MAA_StreamAccess>(loc, rewriter.getI32Type(),iter, dataPtr);
        } else {
          //set data pointer for final data that would be stored into spd
          auto dataPtr = tracePtrType(I.getOperand(0), rewriter);
          iter = rewriter.create<LLVM::MAA_IndirectAccess>(loc, rewriter.getI32Type(),iter, dataPtr);
        }
      }
      ins_idx++;
    }

    rewriter.eraseOp(load);
    return success();
  }
};

class AllocSPDOpLowering : public ConvertOpToLLVMPattern<AllocSPDOp> {
  using ConvertOpToLLVMPattern<AllocSPDOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AllocSPDOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newTy = typeConverter->convertType(alloc.getResult().getType());

    // for all sizes, trace the size back to the original i64
    auto intType = rewriter.getI64Type();
    SmallVector<Value> sizes;

    //FIXME: should use a more robust way to get size
    for (auto size : alloc.getOperands()) {
      // trace the operand back to the original i64
      auto tracedSize = traceIntegerType(size, rewriter);
      sizes.push_back(tracedSize);
    }
    rewriter.replaceOpWithNewOp<LLVM::MAASpdAllocOp>(alloc, newTy, sizes);
    return success();
  }
};
}

namespace {
    void MemAccToLLVMPass::runOnOperation() {
    auto m = getOperation();
    
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = false;
    options.dataLayout = llvm::DataLayout("");
    options.useOpaquePointers = false;
    LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);
    // // TODO: figure out whether to use C-style memref descriptor or not later
    bool useCStyleMemRef = true;
    if (useCStyleMemRef) {
      converter.addConversion([&](MemRefType type) -> std::optional<Type> {
        auto elTy = convertMemrefElementTypeForLLVMPointer(type, converter);
        if (!elTy)
          return Type();
        return LLVM::LLVMPointerType::get(type.getContext(),
                                          type.getMemorySpaceAsInt());
      });
    }
    LLVMConversionTarget target(getContext());
    target.addIllegalOp<AllocSPDOp>();
    target.addIllegalOp<PackedGenericLoadOp>();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AllocSPDOpLowering>(converter);
    patterns.add<PackedGenericLoadOpLowering>(converter);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
        signalPassFailure();
    }
} // end namespace.

namespace mlir {
namespace MemAcc {
std::unique_ptr<Pass> createTestMemAccToLLVMPass() {
  return std::make_unique<MemAccToLLVMPass>();
}
} // namespace MemAcc
} // namespace mlir