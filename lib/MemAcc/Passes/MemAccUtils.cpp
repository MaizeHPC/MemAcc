#include "MemAcc/Passes/MemAccUtils.h"

namespace mlir {
    namespace MemAcc {
        /// Given a range of values, emit the code that reduces them with "min" or "max"
/// depending on the provided comparison predicate, sgt for max and slt for min.
///
/// Multiple values are scanned in a linear sequence.  This creates a data
/// dependences that wouldn't exist in a tree reduction, but is easier to
/// recognize as a reduction by the subsequent passes.
Value buildMinMaxReductionSeq(Location loc,
                                     arith::CmpIPredicate predicate,
                                     ValueRange values, OpBuilder &builder) {
  assert(!values.empty() && "empty min/max chain");
  assert(predicate == arith::CmpIPredicate::sgt ||
         predicate == arith::CmpIPredicate::slt);

  auto valueIt = values.begin();
  Value value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    if (predicate == arith::CmpIPredicate::sgt)
      value = builder.create<arith::MaxSIOp>(loc, value, *valueIt);
    else
      value = builder.create<arith::MinSIOp>(loc, value, *valueIt);
  }

  return value;
}

/// Emit instructions that correspond to computing the maximum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
Value getAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = mlir::affine::expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::sgt, *values,
                                   builder);
  return nullptr;
}

/// Emit instructions that correspond to computing the minimum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
Value getAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands) {
  if (auto values = mlir::affine::expandAffineMap(builder, loc, map, operands))
    return buildMinMaxReductionSeq(loc, arith::CmpIPredicate::slt, *values,
                                   builder);
  return nullptr;
}
    }
}