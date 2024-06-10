#ifndef MEMACC_DIALECT_MEMACC_PASSES_H
#define MEMACC_DIALECT_MEMACC_PASSES_H

#include "MemAcc/Dialect.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

namespace mlir {
class PatternRewriter;
class RewritePatternSet;
class DominanceInfo;
namespace MemAcc {
std::unique_ptr<Pass> createMemAccHoistLoadsPass();

std::unique_ptr<Pass> createDummyPass();
// TODO: a test pass lowering memacc to llvm; it should first lower to
// target-aware IR then to LLVM should fix later
std::unique_ptr<Pass> createTestMemAccToLLVMPass();
} // namespace MemAcc
} // namespace mlir

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // end namespace arith

namespace math {
class MathDialect;
} // end namespace math

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace func {
class FuncDialect;
}

namespace affine {
class AffineDialect;
}

namespace LLVM {
class LLVMDialect;
}

#define GEN_PASS_REGISTRATION
#include "MemAcc/Passes/Passes.h.inc"

} // end namespace mlir

#endif // POLYGEIST_DIALECT_POLYGEIST_PASSES_H
