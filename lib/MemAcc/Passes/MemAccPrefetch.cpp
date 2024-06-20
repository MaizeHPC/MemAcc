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

#define DEBUG_TYPE "memory-access-dummypass"

using namespace mlir;
using namespace mlir::MemAcc;
using namespace mlir::arith;
using namespace mlir::affine;
// Define the data structures at the beginning of your pass

namespace {
struct MemAccPrefetchPass
    : public MemAccPrefetchBase<MemAccPrefetchPass> {
  void runOnOperation() override; //{
   //PRINT("Hello from MemAccDummyPass!\n");
  //}
};
} // end namespace.

namespace {
class PrefectGenericLoadOp : public OpRewritePattern<affine::AffineForOp> {
  public:
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  Value getForLoopLength(AffineForOp forOp) const {
   
    auto lb = forOp.getUpperBoundOperands()[0];
    auto step = forOp.getStep();
    //auto ub = lb+step;
   
   return lb;
  }

  };

  void MemAccPrefetchPass::runOnOperation() {
    //PRINT("Hello from MemAccDummyPass!\n");
    mlir::MLIRContext *context = getOperation()->getContext();
    mlir::RewritePatternSet patterns(context);
    GreedyRewriteConfig config;
    patterns.insert<PrefectGenericLoadOp>(context);
  }
}
namespace mlir {
namespace MemAcc {
std::unique_ptr<Pass> createPrefetchPass() {
  return std::make_unique<MemAccPrefetchPass>();
}
} // namespace MemAcc
} // namespace mlir