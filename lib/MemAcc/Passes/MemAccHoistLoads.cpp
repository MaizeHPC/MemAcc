#include "MemAcc/Dialect.h"
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

// Define the pattern to hoist generic indirect loads outside of loops
class PackGenericLoadOpOutsideLoop
    : public OpRewritePattern<MemAcc::GenericLoadOp> {
public:
  using OpRewritePattern<MemAcc::GenericLoadOp>::OpRewritePattern;

  Value getForLoopLength(AffineForOp forOp) const {
    // auto lb = forOp.getLowerBound();
    auto ub = forOp.getUpperBoundOperands()[0];
    // auto step = forOp.getStep();
    // auto diff = rewriter.create<SubIOp>(forOp.getLoc(), ub, lb);
    // auto div = rewriter.create<SignedDivIOp>(forOp.getLoc(), diff, step);
    // TODO: handle the case where the loop is not a constant step
    return ub;
  }

  LogicalResult matchAndRewrite(MemAcc::GenericLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Ensure there exists a GenericLoadOp in an affine.for loop body
    auto forOp = op->getParentOfType<AffineForOp>();
    assert(forOp && "GenericLoadOp must be in a affine.for loop body");

    // Create alloc_spd ops before affine.for loop
    // the size of the alloc_spd is the same as the loop length
    rewriter.setInsertionPoint(forOp);
    auto loop_length = getForLoopLength(forOp);
    SmallVector<Value, 4> alloc_spds;
    for (auto result : op->getResults()) {
      auto resultType = result.getType();

      // Determine the element type, whether resultType is already a memref or
      // not
      SmallVector<int64_t, 4> shape{mlir::ShapedType::kDynamic};
      mlir::Type elementType;
      if (auto memrefType = resultType.dyn_cast<mlir::MemRefType>()) {
        // If resultType is a MemRefType, extract its element type
        elementType = memrefType.getElementType();
        shape.append(memrefType.getShape().begin(),
                     memrefType.getShape().end());
      } else {
        // Otherwise, use resultType directly as the element type
        elementType = resultType;
      }
      // Create a MemRefType with a dynamic size in the first dimension and the
      // obtained element type
      auto memRefType = mlir::MemRefType::get(shape, elementType);
      alloc_spds.push_back(rewriter.create<AllocSPDOp>(
          op.getLoc(), memRefType, ValueRange({loop_length})));
    }

    llvm::ArrayRef<mlir::Value> alloc_spds_ref(alloc_spds);
    // Create PackedGenericLoadOp outside of the loop
    auto packedLoadOp = rewriter.create<PackedGenericLoadOp>(
        op.getLoc(), ValueRange{alloc_spds_ref}, forOp.getLowerBoundOperands(),
        forOp.getLowerBoundMap(), forOp.getUpperBoundOperands(),
        forOp.getUpperBoundMap(), forOp.getStep(), forOp.getInits());

    // for each instruction in GenericLoadOp, clone them into
    // PackedGenericLoadOp, replace all users of AffineForOp's induction
    // variable with the induction variable of PackedGenericLoadOp
    auto newInductionVar = packedLoadOp.getInductionVar();
    auto originalInductionVar =
        forOp.getInductionVar(); // Get this from the original AffineForOp
                                 // context
    PRINT("Done creating PackedGenericLoadOp: " << *packedLoadOp);
    rewriter.setInsertionPointToStart(&packedLoadOp.getBody().front());
    DenseMap<Value, Value> InstMap;
    for (auto &I : op.getRegion().front()) {
      auto newI = rewriter.clone(I);
      InstMap[I.getResult(0)] = newI->getResult(0);
      // Iterate over the operands of the cloned instruction
      for (unsigned idx = 0; idx < newI->getNumOperands(); ++idx) {
        // Check if the operand is the original induction variable
        if (newI->getOperand(idx) == originalInductionVar) {
          // Replace the operand with the new induction variable
          newI->setOperand(idx, newInductionVar);
        } else if (InstMap.count(newI->getOperand(idx))) {
          newI->setOperand(idx, InstMap[newI->getOperand(idx)]);
        } // assert operand is not in the old for loop
          //   else if
        //   (newI->getOperand(idx).getDefiningOp()->getParentOfType<AffineForOp>()
        //   == forOp) {
        //     PRINT("Violation: " << *newI->getOperand(idx).getDefiningOp() <<
        //     " is in the old for loop"); assert(false && "Operand is in the
        //     old for loop; Too complicated!");
        // }
      }
    }

    PRINT("Done cloning: " << *packedLoadOp);

    // Back to op's place, replace all uses of op with the load alloc_spds
    rewriter.setInsertionPoint(op);

    // for each user of op, replace it with the corresponding load op
    // generate memacc.load ops for each alloc_spd
    // SmallVector<Value, 4> load_ops;
    for (unsigned int i = 0; i < op.getNumResults(); i++) {
      auto load_op = rewriter.create<memref::LoadOp>(
          op.getLoc(), alloc_spds[i], ValueRange({originalInductionVar}));
      op->getResult(i).replaceAllUsesWith(load_op);
      // load_ops.push_back(load_op);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

void MemAccHoistLoadsPass::runOnOperation() {
  mlir::MLIRContext *context = getOperation()->getContext();
  mlir::RewritePatternSet patterns(context);

  patterns.insert<PackGenericLoadOpOutsideLoop>(context);
  GreedyRewriteConfig config;
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