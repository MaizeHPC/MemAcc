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

  template <typename PackedOpType>
  void generatePackedMemAccOp(MemAcc::GenericLoadOp op,
                              PatternRewriter &rewriter,
                              llvm::ArrayRef<mlir::Value> alloc_spds_ref,
                              AffineForOp forOp, int indirection_level) const {
    // Create PackedOpType outside of the loop
    auto packedLoadOp = rewriter.create<PackedOpType>(
        op.getLoc(), alloc_spds_ref, forOp.getLowerBoundOperands(),
        forOp.getLowerBoundMap(), forOp.getUpperBoundOperands(),
        forOp.getUpperBoundMap(), forOp.getStep(), forOp.getInits(),
        indirection_level);

    // for each instruction in GenericLoadOp, clone them into
    // PackedOpType, replace all users of AffineForOp's induction
    // variable with the induction variable of PackedOpType
    auto newInductionVar = packedLoadOp.getInductionVar();
    auto originalInductionVar =
        forOp.getInductionVar(); // Get this from the original AffineForOp
                                 // context
    PRINT("Done creating PackedOp: " << *packedLoadOp);
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
      }
    }

    PRINT("Done cloning: " << *packedLoadOp);
  }

  Value getSpdBuffer(Type resultType, PatternRewriter &rewriter,
                     Value loopLength, MemAcc::GenericLoadOp op) const {
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
    return rewriter.create<AllocSPDOp>(op.getLoc(), memRefType,
                                       ValueRange({loopLength}));
  }

  LogicalResult matchAndRewrite(MemAcc::GenericLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Ensure there exists a GenericLoadOp in an affine.for loop body
    auto forOp = op->getParentOfType<AffineForOp>();
    assert(forOp && "GenericLoadOp must be in a affine.for loop body");

    // Step1: Create alloc_spd ops before affine.for loop
    // the size of the alloc_spd is the same as the loop length
    // the type of the alloc_spd is the same as the result type of the generic
    // op also record the result idx that used for store idx
    rewriter.setInsertionPoint(forOp);
    auto loop_length = getForLoopLength(forOp);
    SmallVector<Value, 4> alloc_spds_gather;
    SmallVector<Value, 4> alloc_spds_scatter;
    SmallVector<size_t, 4> store_idx;
    for (size_t i = 0; i < op->getResults().size(); i++) {
      auto result = op->getResult(i);
      auto resultType = result.getType();
      auto original_alloc_spds_scatter_size = alloc_spds_scatter.size();

      // if the result is used for affine/memref store's idx, record the result
      // idx
      for (auto user : result.getUsers()) {
        if (dyn_cast<affine::AffineStoreOp>(user) &&
            user->getOperand(2) == result) {
          // get storeOp's value' type
          alloc_spds_scatter.push_back(getSpdBuffer(
              user->getOperand(0).getType(), rewriter, loop_length, op));

        } else if (dyn_cast<memref::StoreOp>(user) &&
                   user->getOperand(2) == result) {
          PRINT("Found a memref::StoreOp" << *user);
          // get storeOp's value' type
          alloc_spds_scatter.push_back(getSpdBuffer(
              user->getOperand(0).getType(), rewriter, loop_length, op));
        }
      }

      if (alloc_spds_scatter.size() > original_alloc_spds_scatter_size) {
        store_idx.push_back(i);
        continue;
      }

      alloc_spds_gather.push_back(
          getSpdBuffer(resultType, rewriter, loop_length, op));
    }

    auto indirection_level = op.getIndirectionLevel().value();
    // Step2: Create Packed Memory Access Operations outside of the loop
    // First case: If all result idx are used for store idx, we can create a
    // PackedGenericStireOp
    if (store_idx.size() == op->getResults().size()) {
      // create an empty generic op under the op
      // move all memacc.load that used as store idx to the new op
      // change the store value to a memacc.load from a spd buffer
      generatePackedMemAccOp<MemAcc::PackedGenericStoreOp>(
          op, rewriter, llvm::ArrayRef<mlir::Value>{alloc_spds_scatter}, forOp,
          indirection_level);
    } else if (store_idx.size() == 0) {
      generatePackedMemAccOp<MemAcc::PackedGenericLoadOp>(
          op, rewriter, llvm::ArrayRef<mlir::Value>{alloc_spds_gather}, forOp,
          indirection_level);
      // Back to op's place, replace all uses of op with the load alloc_spds
      rewriter.setInsertionPoint(op);
      auto originalInductionVar =
          forOp.getInductionVar(); // Get this from the original AffineForOp

      // for each user of op, replace it with the corresponding load op
      // generate memacc.load ops for each alloc_spd
      // SmallVector<Value, 4> load_ops;
      for (unsigned int i = 0; i < op.getNumResults(); i++) {
        auto load_op =
            rewriter.create<memref::LoadOp>(op.getLoc(), alloc_spds_gather[i],
                                            ValueRange({originalInductionVar}));
        op->getResult(i).replaceAllUsesWith(load_op);
        // load_ops.push_back(load_op);
      }
    }
    // Second case: If not...

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