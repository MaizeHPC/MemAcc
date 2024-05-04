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
      // Check if the type is already i64
      if (size.getType().isa<IntegerType>() && size.getType().cast<IntegerType>().getWidth() == 64) {
        sizes.push_back(size);
      } else {
        // Trace back the origin of the size if it's not an i64
        if (auto castOp = size.getDefiningOp<UnrealizedConversionCastOp>()) {
          Value original = castOp.getOperand(0);
          // Check if the original value is of i64 type
          if (original.getType().isa<IntegerType>() && original.getType().cast<IntegerType>().getWidth() == 64) {
            sizes.push_back(original);
          } else {
            // If still not i64, add a cast to i64, this is a fallback and might not be semantically correct
            auto castedSize = rewriter.create<LLVM::SExtOp>(alloc.getLoc(), intType, original);
            sizes.push_back(castedSize);
          }
        } else {
          // If we can't trace back to a cast operation, we cast directly here (fallback)
          auto castedSize = rewriter.create<LLVM::SExtOp>(alloc.getLoc(), intType, size);
          sizes.push_back(castedSize);
        }
      }
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
    // bool useCStyleMemRef = true;
    // if (useCStyleMemRef) {
    //   converter.addConversion([&](MemRefType type) -> std::optional<Type> {
    //     auto elTy = convertMemrefElementTypeForLLVMPointer(type, converter);
    //     if (!elTy)
    //       return Type();
    //     return LLVM::LLVMPointerType::get(type.getContext(),
    //                                       type.getMemorySpaceAsInt());
    //   });
    // }
    LLVMConversionTarget target(getContext());
    target.addIllegalOp<AllocSPDOp>();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AllocSPDOpLowering>(converter);

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