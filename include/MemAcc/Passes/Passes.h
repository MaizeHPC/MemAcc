#ifndef MEMACC_DIALECT_MEMACC_PASSES_H
#define MEMACC_DIALECT_MEMACC_PASSES_H

#include "MemAcc/Dialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class PatternRewriter;
class RewritePatternSet;
class DominanceInfo;
namespace MemAcc{
std::unique_ptr<Pass> createMemAccGenPass();
std::unique_ptr<Pass> createMemAccHoistLoadsPass();
}
}

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // end namespace arith

namespace omp {
class OpenMPDialect;
} // end namespace omp

namespace polygeist {
class PolygeistDialect;
} // end namespace polygeist

namespace scf {
class SCFDialect;
} // end namespace scf

namespace cf {
class ControlFlowDialect;
} // end namespace cf

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
