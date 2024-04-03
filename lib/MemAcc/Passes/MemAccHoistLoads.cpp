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

  LogicalResult matchAndRewrite(MemAcc::GenericLoadOp op,
                                PatternRewriter &rewriter) const override {
  // If there exist a GenericLoadOp, it must in a affine.for loop body
  assert(op->getParentOfType<AffineForOp>() &&
         "GenericLoadOp must be in a affine.for loop body");

  // Create a PackedGenericLoadOp before
  


  return success();
  }

};

    
void MemAccHoistLoadsPass::runOnOperation() {
  mlir::MLIRContext *context = getOperation()->getContext();
  mlir::RewritePatternSet patterns(context);

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