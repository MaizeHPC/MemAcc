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

static void get_for_to_indirect_memacc(Operation *op) {
  // Step1: DFS to find all gather traces and scatter traces
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
    if (hasGatherPath) {
      // create spd buffer for external users of gather path
      for (auto &opToUserPair : forOpToGatherPath[forOp].externUsers) {
         rewriter.setInsertionPoint(forOp);
        for (unsigned int i = 0; i < opToUserPair.second.users.size(); i++) {
          auto user = opToUserPair.second.users[i];
          unsigned int operandIdx = opToUserPair.second.operandIdx[i];
          assert(opToUserPair.second.users.size() == opToUserPair.second.operandIdx.size() &&
                 "Mismatch between users and operandIdx");
          assert(operandIdx < user->getNumOperands() &&
                 "operandIdx out of bound");
          auto operand = user->getOperand(operandIdx);
          Value spdBuffer = getSpdBuffer(
              operand.getType(), rewriter, loopLength, forOp->getLoc());
          rewriter.setInsertionPoint(user);
          auto newOperand = rewriter.create<memref::LoadOp>(forOp.getLoc(), spdBuffer,
                                            ValueRange({InductionVar})).getResult();
          user->setOperand(operandIdx, newOperand);
        }
      } // end for
    }
    if (hasScatterPath){
      // create spd buffer for store values of scatter path
      for (auto &opToValPair : forOpToScatterPath[forOp].storeOpVals) {
        rewriter.setInsertionPoint(forOp);
        Value spdBuffer = getSpdBuffer(
            opToValPair.second.getType(), rewriter, loopLength, forOp->getLoc());
        
        // replace the original StoreOp to store to spd buffer
        // assume last instruction is a StoreOp

        // change base address to InductionVar
        opToValPair.first->setOperand(1, spdBuffer);
        // change offset to InductionVar
        opToValPair.first->setOperand(2, InductionVar);
      }
    } // if

    forOpDone[forOp] = true;
    return success();
  }
};

void MemAccHoistLoadsPass::runOnOperation() {
  mlir::MLIRContext *context = getOperation()->getContext();
  mlir::RewritePatternSet patterns(context);

  get_for_to_indirect_memacc(getOperation());
  GreedyRewriteConfig config;
  patterns.insert<PackGenericLoadOpOutsideLoop>(context);

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