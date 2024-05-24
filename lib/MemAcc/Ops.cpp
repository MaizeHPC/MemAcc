#include "MemAcc/Ops.h"
#include "mlir/IR/OpDefinition.h"
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

static bool areValidCastInputsAndOutputs(TypeRange inputs, TypeRange bufs) {
  if (inputs.size() != 1 || bufs.size() != 1)
    return false;
  return succeeded(verifyCompatibleShapes(inputs.front(), bufs.front()));
}

static bool areIndexCastCompatible(TypeRange inputs, TypeRange bufs) {
  if (!areValidCastInputsAndOutputs(inputs, bufs))
    return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(bufs.front());
  if (!srcType || !dstType)
    return false;

  return (srcType.isIndex() && dstType.isSignlessInteger()) ||
         (srcType.isSignlessInteger() && dstType.isIndex());
}

bool MemAcc::IndexCastOp::areCastCompatible(TypeRange inputs, TypeRange bufs) {
  return areIndexCastCompatible(inputs, bufs);
}

// TODO:9tempest: Refactor builder for packedops by creating a base class in
// Ops.h
//===----------------------------------------------------------------------===//
// PackedGenericLoadOp
//===----------------------------------------------------------------------===//
// / 'bodyBuilder' is used to build the body of MemAcc.packed_generic_load If
// iterArgs and / bodyBuilder are empty/null, we include default terminator op.
void PackedGenericLoadOp::build(OpBuilder &builder, OperationState &result,
                                ValueRange bufs, ValueRange lbOperands,
                                AffineMap lbMap, ValueRange ubOperands,
                                AffineMap ubMap, int64_t step,
                                ValueRange iterArgs, int64_t indirection_level,
                                BodyBuilderFn bodyBuilder) {
  assert(((!lbMap && lbOperands.empty()) ||
          lbOperands.size() == lbMap.getNumInputs()) &&
         "lower bound operand count does not match the affine map");
  assert(((!ubMap && ubOperands.empty()) ||
          ubOperands.size() == ubMap.getNumInputs()) &&
         "upper bound operand count does not match the affine map");
  assert(step > 0 && "step has to be a positive integer constant");

  for (Value val : iterArgs)
    result.addTypes(val.getType());

  auto indirection_level_attr = IntegerAttr::get(
      IntegerType::get(builder.getContext(), 64), indirection_level);
  result.addAttribute(getIndirectionLevelAttrStrName(), indirection_level_attr);

  // Add an attribute for the step.
  result.addAttribute(getStepAttrStrName(),
                      builder.getIntegerAttr(builder.getIndexType(), step));
  // Add the bufs.
  result.addOperands(bufs);

  // Add the lower bound.
  result.addAttribute(getLowerBoundAttrStrName(), AffineMapAttr::get(lbMap));
  result.addOperands(lbOperands);

  // Add the upper bound.
  result.addAttribute(getUpperBoundAttrStrName(), AffineMapAttr::get(ubMap));
  result.addOperands(ubOperands);

  result.addOperands(iterArgs);

  result.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(bufs.size()),
                                    static_cast<int32_t>(lbOperands.size()),
                                    static_cast<int32_t>(ubOperands.size()),
                                    static_cast<int32_t>(iterArgs.size())}));
  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  // Value inductionVar =
  bodyBlock.addArgument(builder.getIndexType(), result.location);
  for (Value val : iterArgs)
    bodyBlock.addArgument(val.getType(), val.getLoc());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  // if (iterArgs.empty() && !bodyBuilder) {
  //   ensureTerminator(*bodyRegion, builder, result.location);
  // } else if (bodyBuilder) {
  //   OpBuilder::InsertionGuard guard(builder);
  //   builder.setInsertionPointToStart(&bodyBlock);
  //   bodyBuilder(builder, result.location, inductionVar,
  //               bodyBlock.getArguments().drop_front());
  // }
}

void PackedGenericLoadOp::build(OpBuilder &builder, OperationState &result,
                                ValueRange bufs, int64_t lb, int64_t ub,
                                int64_t step, ValueRange iterArgs,
                                int64_t indirection_level,
                                BodyBuilderFn bodyBuilder) {
  auto lbMap = AffineMap::getConstantMap(lb, builder.getContext());
  auto ubMap = AffineMap::getConstantMap(ub, builder.getContext());
  return build(builder, result, bufs, {}, lbMap, {}, ubMap, step, iterArgs,
               indirection_level, bodyBuilder);
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

SmallVector<Region *> PackedGenericLoadOp::getLoopRegions() {
  return {&getBody()};
}

std::optional<Value> PackedGenericLoadOp::getSingleInductionVar() {
  return getInductionVar();
}

std::optional<OpFoldResult> PackedGenericLoadOp::getSingleLowerBound() {
  if (!hasConstantLowerBound())
    return std::nullopt;
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getConstantLowerBound()));
}

FailureOr<LoopLikeOpInterface> PackedGenericLoadOp::replaceWithAdditionalYields(
    RewriterBase &rewriter, ValueRange newInitOperands,
    bool replaceInitOperandUsesInLoop,
    const NewYieldValuesFn &newYieldValuesFn) {

  assert(false && "not implemented");
  return failure();
}

//===----------------------------------------------------------------------===//
// PackedGenericStoreOp
//===----------------------------------------------------------------------===//
// / 'bodyBuilder' is used to build the body of MemAcc.packed_generic_load If
// iterArgs and / bodyBuilder are empty/null, we include default terminator op.
void PackedGenericStoreOp::build(OpBuilder &builder, OperationState &result,
                                 ValueRange bufs, ValueRange lbOperands,
                                 AffineMap lbMap, ValueRange ubOperands,
                                 AffineMap ubMap, int64_t step,
                                 ValueRange iterArgs, int64_t indirection_level,
                                 BodyBuilderFn bodyBuilder) {
  assert(((!lbMap && lbOperands.empty()) ||
          lbOperands.size() == lbMap.getNumInputs()) &&
         "lower bound operand count does not match the affine map");
  assert(((!ubMap && ubOperands.empty()) ||
          ubOperands.size() == ubMap.getNumInputs()) &&
         "upper bound operand count does not match the affine map");
  assert(step > 0 && "step has to be a positive integer constant");

  for (Value val : iterArgs)
    result.addTypes(val.getType());

  auto indirection_level_attr = IntegerAttr::get(
      IntegerType::get(builder.getContext(), 64), indirection_level);
  result.addAttribute(getIndirectionLevelAttrStrName(), indirection_level_attr);

  // Add an attribute for the step.
  result.addAttribute(getStepAttrStrName(),
                      builder.getIntegerAttr(builder.getIndexType(), step));
  // Add the bufs.
  result.addOperands(bufs);

  // Add the lower bound.
  result.addAttribute(getLowerBoundAttrStrName(), AffineMapAttr::get(lbMap));
  result.addOperands(lbOperands);

  // Add the upper bound.
  result.addAttribute(getUpperBoundAttrStrName(), AffineMapAttr::get(ubMap));
  result.addOperands(ubOperands);

  result.addOperands(iterArgs);

  result.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(bufs.size()),
                                    static_cast<int32_t>(lbOperands.size()),
                                    static_cast<int32_t>(ubOperands.size()),
                                    static_cast<int32_t>(iterArgs.size())}));
  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  // Value inductionVar =
  bodyBlock.addArgument(builder.getIndexType(), result.location);
  for (Value val : iterArgs)
    bodyBlock.addArgument(val.getType(), val.getLoc());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  // if (iterArgs.empty() && !bodyBuilder) {
  //   ensureTerminator(*bodyRegion, builder, result.location);
  // } else if (bodyBuilder) {
  //   OpBuilder::InsertionGuard guard(builder);
  //   builder.setInsertionPointToStart(&bodyBlock);
  //   bodyBuilder(builder, result.location, inductionVar,
  //               bodyBlock.getArguments().drop_front());
  // }
}

void PackedGenericStoreOp::build(OpBuilder &builder, OperationState &result,
                                 ValueRange bufs, int64_t lb, int64_t ub,
                                 int64_t step, ValueRange iterArgs,
                                 int64_t indirection_level,
                                 BodyBuilderFn bodyBuilder) {
  auto lbMap = AffineMap::getConstantMap(lb, builder.getContext());
  auto ubMap = AffineMap::getConstantMap(ub, builder.getContext());
  return build(builder, result, bufs, {}, lbMap, {}, ubMap, step, iterArgs,
               indirection_level, bodyBuilder);
}

std::optional<OpFoldResult> PackedGenericStoreOp::getSingleStep() {
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getStepAsAPInt()));
}

std::optional<OpFoldResult> PackedGenericStoreOp::getSingleUpperBound() {
  if (!hasConstantUpperBound())
    return std::nullopt;
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getConstantUpperBound()));
}

SmallVector<Region *> PackedGenericStoreOp::getLoopRegions() {
  return {&getBody()};
}

std::optional<Value> PackedGenericStoreOp::getSingleInductionVar() {
  return getInductionVar();
}

std::optional<OpFoldResult> PackedGenericStoreOp::getSingleLowerBound() {
  if (!hasConstantLowerBound())
    return std::nullopt;
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getConstantLowerBound()));
}

FailureOr<LoopLikeOpInterface>
PackedGenericStoreOp::replaceWithAdditionalYields(
    RewriterBase &rewriter, ValueRange newInitOperands,
    bool replaceInitOperandUsesInLoop,
    const NewYieldValuesFn &newYieldValuesFn) {

  assert(false && "not implemented");
  return failure();
}

//===----------------------------------------------------------------------===//
// PackedGenericRmwOp
//===----------------------------------------------------------------------===//
// / 'bodyBuilder' is used to build the body of MemAcc.packed_generic_load If
// iterArgs and / bodyBuilder are empty/null, we include default terminator op.
void PackedGenericRmwOp::build(OpBuilder &builder, OperationState &result,
                               ValueRange bufs, ValueRange lbOperands,
                               AffineMap lbMap, ValueRange ubOperands,
                               AffineMap ubMap, int64_t step,
                               ValueRange iterArgs, int64_t indirection_level,
                               BodyBuilderFn bodyBuilder) {
  assert(((!lbMap && lbOperands.empty()) ||
          lbOperands.size() == lbMap.getNumInputs()) &&
         "lower bound operand count does not match the affine map");
  assert(((!ubMap && ubOperands.empty()) ||
          ubOperands.size() == ubMap.getNumInputs()) &&
         "upper bound operand count does not match the affine map");
  assert(step > 0 && "step has to be a positive integer constant");

  for (Value val : iterArgs)
    result.addTypes(val.getType());

  auto indirection_level_attr = IntegerAttr::get(
      IntegerType::get(builder.getContext(), 64), indirection_level);
  result.addAttribute(getIndirectionLevelAttrStrName(), indirection_level_attr);

  // Add an attribute for the step.
  result.addAttribute(getStepAttrStrName(),
                      builder.getIntegerAttr(builder.getIndexType(), step));
  // Add the bufs.
  result.addOperands(bufs);

  // Add the lower bound.
  result.addAttribute(getLowerBoundAttrStrName(), AffineMapAttr::get(lbMap));
  result.addOperands(lbOperands);

  // Add the upper bound.
  result.addAttribute(getUpperBoundAttrStrName(), AffineMapAttr::get(ubMap));
  result.addOperands(ubOperands);

  result.addOperands(iterArgs);

  result.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(bufs.size()),
                                    static_cast<int32_t>(lbOperands.size()),
                                    static_cast<int32_t>(ubOperands.size()),
                                    static_cast<int32_t>(iterArgs.size())}));
  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  // Value inductionVar =
  bodyBlock.addArgument(builder.getIndexType(), result.location);
  for (Value val : iterArgs)
    bodyBlock.addArgument(val.getType(), val.getLoc());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  // if (iterArgs.empty() && !bodyBuilder) {
  //   ensureTerminator(*bodyRegion, builder, result.location);
  // } else if (bodyBuilder) {
  //   OpBuilder::InsertionGuard guard(builder);
  //   builder.setInsertionPointToStart(&bodyBlock);
  //   bodyBuilder(builder, result.location, inductionVar,
  //               bodyBlock.getArguments().drop_front());
  // }
}

void PackedGenericRmwOp::build(OpBuilder &builder, OperationState &result,
                               ValueRange bufs, int64_t lb, int64_t ub,
                               int64_t step, ValueRange iterArgs,
                               int64_t indirection_level,
                               BodyBuilderFn bodyBuilder) {
  auto lbMap = AffineMap::getConstantMap(lb, builder.getContext());
  auto ubMap = AffineMap::getConstantMap(ub, builder.getContext());
  return build(builder, result, bufs, {}, lbMap, {}, ubMap, step, iterArgs,
               indirection_level, bodyBuilder);
}

std::optional<OpFoldResult> PackedGenericRmwOp::getSingleStep() {
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getStepAsAPInt()));
}

std::optional<OpFoldResult> PackedGenericRmwOp::getSingleUpperBound() {
  if (!hasConstantUpperBound())
    return std::nullopt;
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getConstantUpperBound()));
}

SmallVector<Region *> PackedGenericRmwOp::getLoopRegions() {
  return {&getBody()};
}

std::optional<Value> PackedGenericRmwOp::getSingleInductionVar() {
  return getInductionVar();
}

std::optional<OpFoldResult> PackedGenericRmwOp::getSingleLowerBound() {
  if (!hasConstantLowerBound())
    return std::nullopt;
  OpBuilder b(getContext());
  return OpFoldResult(b.getI64IntegerAttr(getConstantLowerBound()));
}

FailureOr<LoopLikeOpInterface> PackedGenericRmwOp::replaceWithAdditionalYields(
    RewriterBase &rewriter, ValueRange newInitOperands,
    bool replaceInitOperandUsesInLoop,
    const NewYieldValuesFn &newYieldValuesFn) {

  assert(false && "not implemented");
  return failure();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

void AllocSPDOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "alloc_spd");
}