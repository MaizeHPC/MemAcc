#include "MemAcc/Passes/MemAccAnalysis.h"
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
#include <tuple>

#include "MemAcc/Passes/Passes.h"

#define DEBUG_TYPE "memory-access-to-llvm"

using namespace mlir;
using namespace mlir::MemAcc;
using namespace affine;

Type convertMemrefElementTypeForLLVMPointer(MemRefType type,
                                            const LLVMTypeConverter &converter);

namespace {
struct MemAccToLLVMPass : public MemAccToLLVMBase<MemAccToLLVMPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace {

DenseMap<Value, Value> spdAllocConversionMap;
DenseMap<AffineForOp, Value> forOpToMAAInitMap;
DenseMap<AffineForOp, Value> forOpToMAAStartMap;

static inline Value getIntegerOpResult(Value size,
                                       ConversionPatternRewriter &rewriter) {
  auto intType = rewriter.getI64Type();
  // Check if the type is already i64
  if (size.getType().isa<IntegerType>()) {
    return size;
  } else {
    // create a UnrealizedConversionCastOp, which will be removed by mlir-opt
    auto castedSize = rewriter.create<UnrealizedConversionCastOp>(
        size.getLoc(), intType, size);
    return castedSize.getResult(0);
  }
}

static Value getPtrOpResult(Value ptr, ConversionPatternRewriter &rewriter) {
  auto type = ptr.getType();
  // Check if the type is already i64
  if (type.isa<LLVM::LLVMPointerType>()) {
    return ptr;
  } else {
    // create a UnrealizedConversionCastOp, which will be removed by mlir-opt
    auto castedPtr = rewriter.create<UnrealizedConversionCastOp>(
        ptr.getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext(), 0),
        ptr);
    return castedPtr.getResult(0);
  }
}

// check whether the pakced Op has the same header of the for op
template <typename PackedOpType>
static bool isTheSameForOp(PackedOpType storeOp, AffineForOp forOp) {
  return storeOp.getLowerBoundOperands() == forOp.getLowerBoundOperands() &&
         storeOp.getUpperBoundOperands() == forOp.getUpperBoundOperands() &&
         storeOp.getStep() == forOp.getStep();
}

template <typename PackedOpType>
static AffineForOp getTargetForOp(PackedOpType packedOp) {
  AffineForOp targetForOp = nullptr;
  for (auto &op : packedOp->getParentRegion()->front()) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      // Check if the store op has the same header of the for op
      // if (isTheSameForOp(packedOp, forOp)) {
      //   targetForOp = forOp;
      //   break;
      // }
      return forOp;
    }
  }
  return targetForOp;
}

static Value getMAAInitOpForOp(AffineForOp forOp,
                               ConversionPatternRewriter &rewriter) {
  if (forOpToMAAInitMap.count(forOp) == 0) {
    auto maa = rewriter
                   .create<LLVM::MAA_Init>(
                       forOp.getLoc(),
                       LLVM::LLVMPointerType::get(rewriter.getContext(), 0))
                   .getResult();
    forOpToMAAInitMap[forOp] = maa;
  }
  return forOpToMAAInitMap[forOp];
}

// configure loop for MAA, return loop root
template <typename PackedOpType>
static std::tuple<Value, Value>
configureLoop(PackedOpType packedOp, ConversionPatternRewriter &rewriter,
              Location loc, Value maa) {
  // now assuming only one operand is used for lower bound and upper bound
  // TODO: Consider more complicated cases where affine map is non-trivial
  int constLowerBound = 0;
  Value lowerBound;
  if (packedOp.hasConstantLowerBound()) {
    constLowerBound = packedOp.getConstantLowerBound();
    // create llvm constant for lower bound
    lowerBound = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), constLowerBound));
  } else {
    lowerBound =
        getIntegerOpResult(packedOp.getLowerBoundOperands()[0], rewriter);
  }

  int constUpperBound = 0;
  Value upperBound;
  if (packedOp.hasConstantUpperBound()) {
    constUpperBound = packedOp.getConstantUpperBound();
    // create llvm constant for upper bound
    upperBound = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), constUpperBound));
  } else {
    upperBound =
        getIntegerOpResult(packedOp.getUpperBoundOperands()[0], rewriter);
  }

  // Step1: set step size
  auto step = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                packedOp.getStepAsAPInt());

  // Step2: Set loop configuration based on address dependency
  auto root =
      rewriter
          .create<LLVM::MAA_SetLoopOp>(loc, rewriter.getI32Type(), lowerBound,
                                       upperBound, step, maa)
          .getResult();

  // Step3: Calculate loop size; assuming step is 1 for now
  auto loopSize = rewriter
                      .create<LLVM::SubOp>(loc, rewriter.getI64Type(),
                                           upperBound, lowerBound)
                      .getResult();
  return std::make_tuple(root, loopSize);
}

class PackedGenericRMWOpLowering
    : public ConvertOpToLLVMPattern<PackedGenericRmwOp> {
  using ConvertOpToLLVMPattern<PackedGenericRmwOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PackedGenericRmwOp packedRmwOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    /// Step1: Find the affine.for op that the generic store op belongs to
    // Search through all instructions under its' parent's region to find the
    // for op
    AffineForOp targetForOp = getTargetForOp(packedRmwOp);
    assert(targetForOp && "Target for op not found");

    /// Step2: Generate MAA configuration instruction for the store op
    // Clone the store op before the for op
    rewriter.setInsertionPoint(targetForOp);

    // Get dependency address information and rmw path
    DFS dfs;
    dfs.analyzeLoadOps<PackedGenericRmwOp>(packedRmwOp);
    auto addressDependencyMap = dfs.getAddressDependencyMap();
    auto rmwPath = dfs.getRMWPath();
    auto loc = packedRmwOp.getLoc();

    Value maa = getMAAInitOpForOp(targetForOp, rewriter);
    // Configure loop header for MAA
    auto [root, loopSize] = configureLoop(packedRmwOp, rewriter, loc, maa);
    DenseMap<Operation *, Value> addressDependencyOpToMAAInst;
    addressDependencyOpToMAAInst[packedRmwOp] = root;
    for (auto &[I, condOp, condBranch] : rmwPath.indirectChain) {
      if (isa<MemAcc::LoadOp>(I)) {
        // Get the llvm.ptr op result for load base address
        auto dataPtr = getPtrOpResult(I->getOperand(0), rewriter);

        // Assert that the address dependency is found and get the dependent MAA
        // inst in a robust way
        assert(addressDependencyMap.count(I) > 0 &&
               "Address dependency not found");
        assert(addressDependencyOpToMAAInst.count(addressDependencyMap[I]) >
                   0 &&
               "MAA dependent Inst not found");
        auto dependentMAAInst =
            addressDependencyOpToMAAInst[addressDependencyMap[I]];
        // set data pointer for internal data access (indices
        addressDependencyOpToMAAInst[I] =
            rewriter
                .create<LLVM::MAA_LoadAccessInt>(loc, rewriter.getI32Type(),
                                                 dependentMAAInst, dataPtr, maa)
                .getResult();
      } else if (isa<MemAcc::RMWOp>(I)) {
        auto rmwOp = dyn_cast<MemAcc::RMWOp>(I);
        assert(addressDependencyMap.count(I) > 0 &&
               "Address dependency not found");
        assert(addressDependencyOpToMAAInst.count(addressDependencyMap[I]) >
                   0 &&
               "MAA dependent Inst not found");
        auto dependentMAAInst =
            addressDependencyOpToMAAInst[addressDependencyMap[I]];
        // Get the llvm.ptr op result for store base address
        auto dataPtr = getPtrOpResult(I->getOperand(1), rewriter);

        // Assert the store op's val is a memacc.load from spd buffer;
        auto modifiedDataOp =
            rmwPath.storeToRmwOp[I].modifiedValue.getDefiningOp();
        assert(isa<MemAcc::LoadOp>(modifiedDataOp));
        auto spdBuf = spdAllocConversionMap[modifiedDataOp->getOperand(0)];
        rewriter.create<LLVM::MAA_RMWAccess>(loc, dependentMAAInst, dataPtr,
                                             spdBuf, maa,
                                             (uint32_t)rmwOp.getKind());
      }
    }
    if (forOpToMAAStartMap.count(targetForOp) > 0) {
      rewriter.eraseOp(forOpToMAAStartMap[targetForOp].getDefiningOp());
    }
    rewriter.setInsertionPoint(targetForOp);
    forOpToMAAStartMap[targetForOp] = rewriter.create<LLVM::MAA_Start>(
        loc, rewriter.getI32Type(), root, loopSize, maa);

    /// Step3: Replace current packed store op with MAA flush
    rewriter.setInsertionPointAfter(packedRmwOp);
    rewriter.create<LLVM::MAA_Flush>(loc, rewriter.getI32Type(), root, loopSize,
                                     maa);
    rewriter.eraseOp(packedRmwOp);
    return success();
  }
};

class PackedGenericStoreOpLowering
    : public ConvertOpToLLVMPattern<PackedGenericStoreOp> {
  using ConvertOpToLLVMPattern<PackedGenericStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PackedGenericStoreOp packedStoreOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    /// Step1: Find the affine.for op that the generic store op belongs to
    // Search through all instructions under its' parent's region to find the
    // for op
    AffineForOp targetForOp = getTargetForOp(packedStoreOp);
    assert(targetForOp && "Target for op not found");

    /// Step2: Generate MAA configuration instruction for the store op
    // Clone the store op before the for op
    rewriter.setInsertionPoint(targetForOp);
    // Get dependency address information and gather path
    DFS dfs;
    dfs.analyzeLoadOps<PackedGenericStoreOp>(packedStoreOp);
    auto addressDependencyMap = dfs.getAddressDependencyMap();
    auto scatterPath = dfs.getScatterPath();
    auto loc = packedStoreOp.getLoc();

    Value maa = getMAAInitOpForOp(targetForOp, rewriter);
    // Configure loop header for MAA
    auto [root, loopSize] = configureLoop(packedStoreOp, rewriter, loc, maa);
    DenseMap<Operation *, Value> addressDependencyOpToMAAInst;
    addressDependencyOpToMAAInst[packedStoreOp] = root;
    for (auto &[I, condOp, condBranch] : scatterPath.indirectChain) {
      // Assert that the address dependency is found and get the dependent MAA
      // inst in a robust way
      if (isa<MemAcc::LoadOp>(I)) {
        assert(addressDependencyMap.count(I) > 0 &&
               "Address dependency not found");
        assert(addressDependencyOpToMAAInst.count(addressDependencyMap[I]) >
                   0 &&
               "MAA dependent Inst not found");
        auto dependentMAAInst =
            addressDependencyOpToMAAInst[addressDependencyMap[I]];
        // Get the llvm.ptr op result for load base address
        auto dataPtr = getPtrOpResult(I->getOperand(0), rewriter);
        // set data pointer for internal data access (indices
        addressDependencyOpToMAAInst[I] =
            rewriter
                .create<LLVM::MAA_LoadAccessInt>(loc, rewriter.getI32Type(),
                                                 dependentMAAInst, dataPtr, maa)
                .getResult();
      } else if (isa<MemAcc::StoreOp>(I)) {
        assert(addressDependencyMap.count(I) > 0 &&
               "Address dependency not found");
        assert(addressDependencyOpToMAAInst.count(addressDependencyMap[I]) >
                   0 &&
               "MAA dependent Inst not found");
        auto dependentMAAInst =
            addressDependencyOpToMAAInst[addressDependencyMap[I]];
        // Get the llvm.ptr op result for store base address
        auto dataPtr = getPtrOpResult(I->getOperand(1), rewriter);

        // Assert the store op's val is a memacc.load from spd buffer;
        assert(isa<MemAcc::LoadOp>(scatterPath.storeOpVals[I].getDefiningOp()));
        auto spdBuf = spdAllocConversionMap
            [scatterPath.storeOpVals[I].getDefiningOp()->getOperand(0)];
        rewriter.create<LLVM::MAA_StoreAccess>(loc, dependentMAAInst, dataPtr,
                                               spdBuf, maa);
      }
    }

    if (forOpToMAAStartMap.count(targetForOp) > 0) {
      rewriter.eraseOp(forOpToMAAStartMap[targetForOp].getDefiningOp());
    }
    rewriter.setInsertionPoint(targetForOp);
    forOpToMAAStartMap[targetForOp] = rewriter.create<LLVM::MAA_Start>(
        loc, rewriter.getI32Type(), root, loopSize, maa);

    /// Step3: Replace current packed store op with MAA flush
    rewriter.setInsertionPointAfter(packedStoreOp);
    rewriter.create<LLVM::MAA_Flush>(loc, rewriter.getI32Type(), root, loopSize,
                                     maa);
    rewriter.eraseOp(packedStoreOp);
    return success();
  }
};

class PackedGenericLoadOpLowering
    : public ConvertOpToLLVMPattern<PackedGenericLoadOp> {
  using ConvertOpToLLVMPattern<PackedGenericLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PackedGenericLoadOp packedLoadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    /*
    "memacc.packed_generic_load"(%alloc_spd, %0) <{indirectionLevel = 1 : i64,
    lowerBoundMap = #map, operandSegmentSizes = array<i32: 1, 0, 1, 0>, step = 1
    : index, upperBoundMap = #map1}> ({ ^bb0(%arg4: index): %1 = memacc.load
    %arg2[%arg4] : memref<?xi32> %2 = memacc.index_cast %1 : i32 to index %3 =
    memacc.load %arg1[%2] : memref<?xf64> %4 = memacc.yield %3 : (f64) -> f64
      }) : (memref<?xf64>, index) -> ()

      will be transformed into
       %5 = "llvm.intr.maa.init"() : () -> !llvm.ptr
      %6 = llvm.mlir.constant(1 : i32) : i32
      %7 = "llvm.intr.maa.setloop"(%4, %2, %6, %5) : (i64, i64, i32, !llvm.ptr)
    -> i32 %8 = "llvm.intr.maa.load.int"(%7, %arg2, %5) : (i32, !llvm.ptr,
    !llvm.ptr) -> i32 %9 = "llvm.intr.maa.load.ext"(%8, %arg1, %3, %5) : (i32,
    !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 %10 = "llvm.intr.maa.start"(%7, %5)
    : (i32, !llvm.ptr) -> i32

    */

    auto loc = packedLoadOp.getLoc();

    AffineForOp targetForOp = getTargetForOp(packedLoadOp);
    assert(targetForOp && "Target for op not found");

    // Get dependency address information and gather path
    DFS dfs;
    dfs.analyzeLoadOps<PackedGenericLoadOp>(packedLoadOp);
    auto addressDependencyMap = dfs.getAddressDependencyMap();
    auto gatherPath = dfs.getGatherPath();
    /// Step1: initialize MAA;
    auto maa = getMAAInitOpForOp(targetForOp, rewriter);
    // Configure loop header for MAA
    auto [root, loopSize] = configureLoop(packedLoadOp, rewriter, loc, maa);
    DenseMap<Operation *, Value> addressDependencyOpToMAAInst;
    addressDependencyOpToMAAInst[packedLoadOp] = root;
    for (auto &[I, condOp, condBranch] : gatherPath.indirectChain) {
      if (isa<MemAcc::LoadOp>(I)) {
        // Get the llvm.ptr op result for load base address
        auto dataPtr = getPtrOpResult(I->getOperand(0), rewriter);

        // Assert that the address dependency is found and get the dependent MAA
        // inst in a robust way
        assert(addressDependencyMap.count(I) > 0 &&
               "Address dependency not found");
        assert(addressDependencyOpToMAAInst.count(addressDependencyMap[I]) >
                   0 &&
               "MAA dependent Inst not found");
        auto dependentMAAInst =
            addressDependencyOpToMAAInst[addressDependencyMap[I]];
        if (gatherPath.deepestLoadToExternUsers.count(I) > 0) {
          assert(gatherPath.deepestLoadToExternUsers[I].operandIdx.size() ==
                     1 &&
                 "Should only have one operand index for yield op");
          auto spdBufIndex =
              gatherPath.deepestLoadToExternUsers[I].operandIdx[0];
          // set data pointer for final data that would be stored into spd
          addressDependencyOpToMAAInst[I] =
              rewriter
                  .create<LLVM::MAA_LoadAccessExt>(
                      loc, rewriter.getI32Type(), dependentMAAInst, dataPtr,
                      spdAllocConversionMap[packedLoadOp
                                                .getBufs()[spdBufIndex]],
                      maa)
                  .getResult();
        } else {
          // set data pointer for internal data access (indices
          addressDependencyOpToMAAInst[I] =
              rewriter
                  .create<LLVM::MAA_LoadAccessInt>(loc, rewriter.getI32Type(),
                                                   dependentMAAInst, dataPtr,
                                                   maa)
                  .getResult();
        }
      }
    }

    // finally initiate the loop
    if (forOpToMAAStartMap.count(targetForOp) > 0) {
      rewriter.eraseOp(forOpToMAAStartMap[targetForOp].getDefiningOp());
    }
    rewriter.setInsertionPoint(targetForOp);
    forOpToMAAStartMap[targetForOp] = rewriter.create<LLVM::MAA_Start>(
        loc, rewriter.getI32Type(), root, loopSize, maa);
    rewriter.eraseOp(packedLoadOp);
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
    SmallVector<Value> sizes;

    // FIXME: should use a more robust way to get size
    for (auto size : alloc.getOperands()) {
      // trace the operand back to the original i64
      auto tracedSize = getIntegerOpResult(size, rewriter);
      sizes.push_back(tracedSize);
    }
    auto newOp =
        rewriter.replaceOpWithNewOp<LLVM::MAASpdAllocOp>(alloc, newTy, sizes);
    spdAllocConversionMap[alloc.getResult()] = newOp.getResult();
    return success();
  }
};
} // namespace

namespace {
void MemAccToLLVMPass::runOnOperation() {
  auto m = getOperation();

  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(m));
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
  target.addIllegalOp<PackedGenericStoreOp>();
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<AllocSPDOpLowering>(converter, /*benefit=*/3);
  patterns.add<PackedGenericLoadOpLowering>(converter, /*benefit=*/1);
  patterns.add<PackedGenericStoreOpLowering>(converter, /*benefit=*/1);
  patterns.add<PackedGenericRMWOpLowering>(converter, /*benefit=*/1);

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