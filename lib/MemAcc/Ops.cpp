#include "MemAcc/Ops.h"
#include "mlir/IR/TypeUtilities.h"
using namespace mlir;
using namespace MemAcc;
#include "MemAcc/Dialect.h"

#define GET_OP_CLASSES
#include "MemAcc/MemAccOps.cpp.inc"

#define DEBUG_TYPE "memacc"

template <typename... Types> using type_list = std::tuple<Types...> *;

/// Returns a non-null type only if the provided type is one of the allowed
/// types or one of the allowed shaped types of the allowed types. Returns the
/// element type if a valid shaped type is provided.
template <typename... ShapedTypes, typename... ElementTypes>
static Type getUnderlyingType(Type type, type_list<ShapedTypes...>,
                              type_list<ElementTypes...>) {
  if (llvm::isa<ShapedType>(type) && !llvm::isa<ShapedTypes...>(type))
    return {};

  auto underlyingType = getElementTypeOrSelf(type);
  if (!llvm::isa<ElementTypes...>(underlyingType))
    return {};

  return underlyingType;
}

/// Get allowed underlying types for vectors, tensors, and memrefs.
template <typename... ElementTypes>
static Type getTypeIfLikeOrMemRef(Type type) {
  return getUnderlyingType(type,
                           type_list<VectorType, TensorType, MemRefType>(),
                           type_list<ElementTypes...>());
}

static bool areValidCastInputsAndOutputs(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  return succeeded(verifyCompatibleShapes(inputs.front(), outputs.front()));
}

static bool areIndexCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(outputs.front());
  if (!srcType || !dstType)
    return false;

  return (srcType.isIndex() && dstType.isSignlessInteger()) ||
         (srcType.isSignlessInteger() && dstType.isIndex());
}

bool MemAcc::IndexCastOp::areCastCompatible(TypeRange inputs,
                                            TypeRange outputs) {
  return areIndexCastCompatible(inputs, outputs);
}

//===----------------------------------------------------------------------===//
// PackedGenericLoadOp
//===----------------------------------------------------------------------===//

FailureOr<LoopLikeOpInterface> PackedGenericLoadOp::replaceWithAdditionalYields(
    RewriterBase &rewriter, ValueRange newInitOperands,
    bool replaceInitOperandUsesInLoop,
    const NewYieldValuesFn &newYieldValuesFn) {

  assert(false && "not implemented");
  return failure();
}

void PackedGenericLoadOp::setLowerBound(ValueRange lbOperands, AffineMap map) {
  assert(lbOperands.size() == map.getNumInputs());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");

  SmallVector<Value, 4> newOperands(lbOperands.begin(), lbOperands.end());

  auto ubOperands = getUpperBoundOperands();
  newOperands.append(ubOperands.begin(), ubOperands.end());
  auto iterOperands = getInits();
  newOperands.append(iterOperands.begin(), iterOperands.end());
  (*this)->setOperands(newOperands);

  (*this)->setAttr(getLowerBoundAttrStrName(), AffineMapAttr::get(map));
}

void PackedGenericLoadOp::setUpperBound(ValueRange ubOperands, AffineMap map) {
  assert(ubOperands.size() == map.getNumInputs());
  assert(map.getNumResults() >= 1 && "bound map has at least one result");

  SmallVector<Value, 4> newOperands(getLowerBoundOperands());
  newOperands.append(ubOperands.begin(), ubOperands.end());
  auto iterOperands = getInits();
  newOperands.append(iterOperands.begin(), iterOperands.end());
  (*this)->setOperands(newOperands);

  (*this)->setAttr(getUpperBoundAttrStrName(), AffineMapAttr::get(map));
}

bool PackedGenericLoadOp::hasConstantLowerBound() {
  return getLowerBoundMap().isSingleConstant();
}

bool PackedGenericLoadOp::hasConstantUpperBound() {
  return getUpperBoundMap().isSingleConstant();
}

int64_t PackedGenericLoadOp::getConstantLowerBound() {
  return getLowerBoundMap().getSingleConstantResult();
}

int64_t PackedGenericLoadOp::getConstantUpperBound() {
  return getUpperBoundMap().getSingleConstantResult();
}

void PackedGenericLoadOp::setConstantLowerBound(int64_t value) {
  setLowerBound({}, AffineMap::getConstantMap(value, getContext()));
}

void PackedGenericLoadOp::setConstantUpperBound(int64_t value) {
  setUpperBound({}, AffineMap::getConstantMap(value, getContext()));
}

PackedGenericLoadOp::operand_range PackedGenericLoadOp::getControlOperands() {
  return {operand_begin(), operand_begin() + getLowerBoundOperands().size() +
                               getUpperBoundOperands().size()};
}

bool PackedGenericLoadOp::matchingBoundOperandList() {
  auto lbMap = getLowerBoundMap();
  auto ubMap = getUpperBoundMap();
  if (lbMap.getNumDims() != ubMap.getNumDims() ||
      lbMap.getNumSymbols() != ubMap.getNumSymbols())
    return false;

  unsigned numOperands = lbMap.getNumInputs();
  for (unsigned i = 0, e = lbMap.getNumInputs(); i < e; i++) {
    // Compare Value 's.
    if (getOperand(i) != getOperand(numOperands + i))
      return false;
  }
  return true;
}

std::optional<OpFoldResult> PackedGenericLoadOp::getSingleStep() {
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getStepAsAPInt()));
}

std::optional<OpFoldResult> PackedGenericLoadOp::getSingleUpperBound() {
  if (!hasConstantUpperBound())
    return std::nullopt;
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getConstantUpperBound()));
}

SmallVector<Region *> PackedGenericLoadOp::getLoopRegions() { return {&getBody()}; }

std::optional<Value> PackedGenericLoadOp::getSingleInductionVar() {
  return getInductionVar();
}

std::optional<OpFoldResult> PackedGenericLoadOp::getSingleLowerBound() {
  if (!hasConstantLowerBound())
    return std::nullopt;
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getConstantLowerBound()));
}