#include "MemAcc/Dialect.h"
#include "MemAcc/Passes/MemAccAnalysis.h"
#include "MemAcc/Passes/Passes.h"
#include "PassDetails.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memory-access-hoist-loads"

using namespace mlir;
using namespace mlir::MemAcc;
using namespace mlir::arith;
using namespace mlir::affine;
// Define the data structures at the beginning of your pass

namespace {
struct MemAccHoistLoadsPass
    : public MemAccHoistLoadsBase<MemAccHoistLoadsPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace {

llvm::DenseMap<Operation *, DFS::GatherPath> forOpToGatherPath;
llvm::DenseMap<Operation *, DFS::ScatterPath> forOpToScatterPath;
llvm::DenseMap<Operation *, bool> forOpDone;

static void getForToIndirectAccess(Operation *op) {
  // DFS to find all gather traces and scatter traces
  // for all AffineForOps
  op->walk([&](Operation *currentOp) {
    if (isa<affine::AffineForOp>(currentOp)) {
      DFS dfs;
      DFS::GatherPath gatherPath;
      DFS::ScatterPath scatterPath;
      dfs.analyzeLoadOps<affine::AffineForOp>(
          dyn_cast<affine::AffineForOp>(currentOp), gatherPath, scatterPath);
      forOpToGatherPath[currentOp] = gatherPath;
      forOpToScatterPath[currentOp] = scatterPath;
      forOpDone[currentOp] = false;
    }
  });

  // Print forOp -> GatherPath/SactterPath
  for (auto &forOpGatherPath : forOpToGatherPath) {
    PRINT("ForOp:");
    PRINT(*forOpGatherPath.first);

    PRINT("GatherPath:");
    assert(forOpToGatherPath.count(forOpGatherPath.first) == 1);
    assert(forOpToScatterPath.count(forOpGatherPath.first) == 1);
    forOpGatherPath.second.print();
    forOpToScatterPath[forOpGatherPath.first].print();
  }
}

template <typename SrcOpType, typename DestOpType>
class ConvertArithToMemAccPattern : public OpRewritePattern<SrcOpType> {
public:
  using OpRewritePattern<SrcOpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOpType op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<MemAcc::PackedGenericLoadOp>()
    || op->template getParentOfType<MemAcc::PackedGenericStoreOp>()) {
      rewriter.replaceOpWithNewOp<DestOpType>(op, op.getResult().getType(),
                                              op.getOperands());
      return success();
    }
    return failure();
  }
};

class ConvertArithIndexCastToMemAccIndexCastPattern
    : public OpRewritePattern<arith::IndexCastOp> {
public:
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<MemAcc::PackedGenericLoadOp>()
    || op->template getParentOfType<MemAcc::PackedGenericStoreOp>()) {
      rewriter.replaceOpWithNewOp<MemAcc::IndexCastOp>(
          op, op.getResult().getType(), op.getOperand());
      return success();
    }
    return failure();
  }
};

template <typename LoadOpType>
struct LoadOpConversionPattern : public OpRewritePattern<LoadOpType> {
  using OpRewritePattern<LoadOpType>::OpRewritePattern;

  void rewriteLoadOp(LoadOpType loadOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<MemAcc::LoadOp>(loadOp, loadOp.getMemRef(),
                                                loadOp.getIndices());
  }

  LogicalResult matchAndRewrite(LoadOpType loadOp,
                                PatternRewriter &rewriter) const override {
    if (loadOp->template getParentOfType<MemAcc::PackedGenericLoadOp>() ||
        loadOp->template getParentOfType<MemAcc::PackedGenericStoreOp>()) {
      rewriteLoadOp(loadOp, rewriter);
      return success();
    } else {
      return failure();
    }
  }
};

// Define the pattern to hoist generic indirect loads outside of loops
class PackGenericLoadOpOutsideLoop
    : public OpRewritePattern<affine::AffineForOp> {
public:
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  Value getForLoopLength(AffineForOp forOp) const {
    // auto lb = forOp.getLowerBound();
    auto ub = forOp.getUpperBoundOperands()[0];
    // auto step = forOp.getStep();
    // auto diff = rewriter.create<SubIOp>(forOp.getLoc(), ub, lb);
    // auto div = rewriter.create<SignedDivIOp>(forOp.getLoc(), diff, step);
    // TODO: handle the case where the loop is not a constant step
    return ub;
  }

  template <typename PackedOpType>
  void generatePackedMemAccOp(Location loc, PatternRewriter &rewriter,
                              llvm::ArrayRef<mlir::Value> alloc_spds_ref,
                              AffineForOp forOp,
                              const DFS::GatherPath &gatherPath,
                              unsigned int indirectLevel) const {

    llvm::SmallVector<Operation *, 16> indirectChain = gatherPath.indirectChain;
    // Create PackedOpType outside of the loop
    auto packedLoadOp = rewriter.create<PackedOpType>(
        loc, alloc_spds_ref, forOp.getLowerBoundOperands(),
        forOp.getLowerBoundMap(), forOp.getUpperBoundOperands(),
        forOp.getUpperBoundMap(), forOp.getStep(), forOp.getInits(), indirectLevel);

    // for each instruction in GenericLoadOp, clone them into
    // PackedOpType, replace all users of AffineForOp's induction
    // variable with the induction variable of PackedOpType
    auto newInductionVar = packedLoadOp.getInductionVar();
    auto originalInductionVar =
        forOp.getInductionVar(); // Get this from the original AffineForOp
                                 // context
    rewriter.setInsertionPointToStart(&packedLoadOp.getBody().front());
    DenseMap<Operation *, Value> opToResultMap;
    for (auto I : indirectChain) {
      auto newI = rewriter.clone(*I);
      opToResultMap[I] = newI->getResult(0);
      for (unsigned idx = 0; idx < newI->getNumOperands(); ++idx) {
        // Check if the operand is the original induction variable
        if (newI->getOperand(idx) == originalInductionVar) {
          // Replace the operand with the new induction variable
          newI->setOperand(idx, newInductionVar);
        } else if (opToResultMap.count(newI->getOperand(idx).getDefiningOp())) {
          newI->setOperand(
              idx, opToResultMap[newI->getOperand(idx).getDefiningOp()]);
        } // assert operand is not in the old for loop
      }
    }

    /// Insert a memacc.yield op at the end of the packed load op
    // result type of the yield is the element type of each spd buffer
    SmallVector<Type, 4> resultTypes;
    for (auto spdBuffer : alloc_spds_ref) {
      if (auto memrefType = spdBuffer.getType().dyn_cast<MemRefType>()) {
        resultTypes.push_back(memrefType.getElementType());
      } else {
        assert(false && "Unexpected type for spd buffer");
      }
    }
    SmallVector<Value, 4> resultVals;
    // result val are the keys of external users(deepest load)
    for (auto deepestLoad : gatherPath.externUsers) {
      resultVals.push_back(opToResultMap[deepestLoad.first]);
    }
    rewriter.create<MemAcc::YieldOp>(loc, resultTypes, resultVals);
    PRINT("Done cloning: " << *packedLoadOp);
  }

  Value getSpdBuffer(Type resultType, PatternRewriter &rewriter,
                     Value loopLength, Location loc) const {
    // Determine the element type, whether resultType is already a memref or
    // not
    SmallVector<int64_t, 4> shape{mlir::ShapedType::kDynamic};
    mlir::Type elementType;
    if (auto memrefType = resultType.dyn_cast<mlir::MemRefType>()) {
      // If resultType is a MemRefType, extract its element type
      elementType = memrefType.getElementType();
      shape.append(memrefType.getShape().begin(), memrefType.getShape().end());
    } else {
      // Otherwise, use resultType directly as the element type
      elementType = resultType;
    }
    // Create a MemRefType with a dynamic size in the first dimension and the
    // obtained element type
    auto memRefType = mlir::MemRefType::get(shape, elementType);
    return rewriter.create<AllocSPDOp>(loc, memRefType,
                                       ValueRange({loopLength}));
  }

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOpDone[forOp]) {
      return failure();
    }
    bool hasGatherPath = (forOpToGatherPath[forOp].indirectChain.size() != 0);
    bool hasScatterPath = (forOpToScatterPath[forOp].indirectChain.size() != 0);
    auto InductionVar = forOp.getInductionVar();
    if (!hasGatherPath && !hasScatterPath) {
      forOpDone[forOp] = true;
      return failure();
    }

    // Step1: Create alloc_spd ops before affine.for loop
    // the size of the alloc_spd is the same as the loop length
    // the type of the alloc_spd is the same as the result type of the generic
    // op also record the result idx that used for store idx
    Value loopLength = getForLoopLength(forOp);
    llvm::SmallVector<Value> spdBufferGather;
    llvm::SmallVector<Value> spdBufferScatter;
    if (hasGatherPath) {
      // create spd buffer for external users of gather path
      for (auto &opToUserPair : forOpToGatherPath[forOp].externUsers) {
        rewriter.setInsertionPoint(forOp);
        for (unsigned int i = 0; i < opToUserPair.second.users.size(); i++) {
          auto user = opToUserPair.second.users[i];
          unsigned int operandIdx = opToUserPair.second.operandIdx[i];
          assert(opToUserPair.second.users.size() ==
                     opToUserPair.second.operandIdx.size() &&
                 "Mismatch between users and operandIdx");
          assert(operandIdx < user->getNumOperands() &&
                 "operandIdx out of bound");
          auto operand = user->getOperand(operandIdx);
          Value spdBuffer = getSpdBuffer(operand.getType(), rewriter,
                                         loopLength, forOp->getLoc());
          rewriter.setInsertionPoint(user);
          auto newOperand =
              rewriter
                  .create<memref::LoadOp>(forOp.getLoc(), spdBuffer,
                                          ValueRange({InductionVar}))
                  .getResult();
          user->setOperand(operandIdx, newOperand);
          spdBufferGather.push_back(spdBuffer);
        }
      } // end for

      /// Step2: Create PackedGenericLoadOp and hoist outside of the loop
      rewriter.setInsertionPoint(forOp);
      generatePackedMemAccOp<MemAcc::PackedGenericLoadOp>(
          forOp.getLoc(), rewriter,
          llvm::ArrayRef<mlir::Value>{spdBufferGather}, forOp,
          forOpToGatherPath[forOp], forOpToGatherPath[forOp].indirectDepth);
    }
    if (hasScatterPath) {
      // create spd buffer for store values of scatter path
      for (auto &opToValPair : forOpToScatterPath[forOp].storeOpVals) {
        rewriter.setInsertionPoint(forOp);
        Value spdBuffer = getSpdBuffer(opToValPair.second.getType(), rewriter,
                                       loopLength, forOp->getLoc());

        // replace the original StoreOp to store to spd buffer
        // assume last instruction is a StoreOp

        // change base address to InductionVar
        opToValPair.first->setOperand(1, spdBuffer);
        // change offset to InductionVar
        opToValPair.first->setOperand(2, InductionVar);
      }
      /// Step2: Create PackedGenericStoreOp and sink outside of the loop
    } // if

    forOpDone[forOp] = true;
    return success();
  }
};

void MemAccHoistLoadsPass::runOnOperation() {
  mlir::MLIRContext *context = getOperation()->getContext();
  mlir::RewritePatternSet patterns(context);

  getForToIndirectAccess(getOperation());
  GreedyRewriteConfig config;
  patterns.insert<PackGenericLoadOpOutsideLoop>(context);
  patterns.insert<LoadOpConversionPattern<memref::LoadOp>>(context);
  patterns.insert<LoadOpConversionPattern<affine::AffineLoadOp>>(context);
  patterns.add<ConvertArithToMemAccPattern<arith::MulIOp, MemAcc::MulIOp>>(
      context);
  patterns.add<ConvertArithToMemAccPattern<arith::AddIOp, MemAcc::AddIOp>>(
      context);
  patterns.add<ConvertArithToMemAccPattern<arith::SubIOp, MemAcc::SubIOp>>(
      context);
  patterns.add<ConvertArithIndexCastToMemAccIndexCastPattern>(context);

  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}
} // end namespace.

namespace mlir {
namespace MemAcc {
std::unique_ptr<Pass> createMemAccHoistLoadsPass() {
  return std::make_unique<MemAccHoistLoadsPass>();
}
} // namespace MemAcc
} // namespace mlir