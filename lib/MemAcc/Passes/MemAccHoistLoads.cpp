#include "PassDetails.h"
#include "MemAcc/Dialect.h"
#include "MemAcc/Passes/Passes.h"
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
struct MemAccHoistLoadsPass : public MemAccHoistLoadsBase<MemAccHoistLoadsPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace {
void MemAccHoistLoadsPass::runOnOperation() {
  // Define the data structures at the beginning of your pass
}
} // end namespace.

namespace mlir {
namespace MemAcc {
std::unique_ptr<Pass> mlir::MemAcc::createMemAccHoistLoadsPass() {
  return std::make_unique<MemAccHoistLoadsPass>();
}
} // namespace MemAcc
} // namespace mlir