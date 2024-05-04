#include "MemAcc/Ops.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/OpDefinition.h"
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

/// 'bodyBuilder' is used to build the body of MemAcc.packed_generic_load If iterArgs and
/// bodyBuilder are empty/null, we include default terminator op.
void PackedGenericLoadOp::build(OpBuilder &builder, OperationState &result,
                        ValueRange outputs,
                        ValueRange lbOperands, AffineMap lbMap,
                        ValueRange ubOperands, AffineMap ubMap, int64_t step,
                        ValueRange iterArgs, BodyBuilderFn bodyBuilder) {
  assert(((!lbMap && lbOperands.empty()) ||
          lbOperands.size() == lbMap.getNumInputs()) &&
         "lower bound operand count does not match the affine map");
  assert(((!ubMap && ubOperands.empty()) ||
          ubOperands.size() == ubMap.getNumInputs()) &&
         "upper bound operand count does not match the affine map");
  assert(step > 0 && "step has to be a positive integer constant");

  for (Value val : iterArgs)
    result.addTypes(val.getType());

  // Add an attribute for the step.
  result.addAttribute(getStepAttrStrName(),
                      builder.getIntegerAttr(builder.getIndexType(), step));
  // Add the outputs.
  result.addOperands(outputs);

  // Add the lower bound.
  result.addAttribute(getLowerBoundAttrStrName(), AffineMapAttr::get(lbMap));
  result.addOperands(lbOperands);

  // Add the upper bound.
  result.addAttribute(getUpperBoundAttrStrName(), AffineMapAttr::get(ubMap));
  result.addOperands(ubOperands);

  result.addOperands(iterArgs);

  result.addAttribute("operandSegmentSizes",
                      builder.getDenseI32ArrayAttr(
                          {static_cast<int32_t>(outputs.size()),
                           static_cast<int32_t>(lbOperands.size()),
                           static_cast<int32_t>(ubOperands.size()),
                           static_cast<int32_t>(iterArgs.size())}));
  // Create a region and a block for the body.  The argument of the region is
  // the loop induction variable.
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  Value inductionVar =
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

void PackedGenericLoadOp::build(OpBuilder &builder, OperationState &result, ValueRange outputs, int64_t lb,
                        int64_t ub, int64_t step, ValueRange iterArgs,
                        BodyBuilderFn bodyBuilder) {
  auto lbMap = AffineMap::getConstantMap(lb, builder.getContext());
  auto ubMap = AffineMap::getConstantMap(ub, builder.getContext());
  return build(builder, result, outputs, {}, lbMap, {}, ubMap, step, iterArgs,
               bodyBuilder);
}

// /// Parse a for operation loop bounds.
// static ParseResult parseBound(bool isLower, OperationState &result,
//                               OpAsmParser &p) {
//   // 'min' / 'max' prefixes are generally syntactic sugar, but are required if
//   // the map has multiple results.
//   bool failedToParsedMinMax =
//       failed(p.parseOptionalKeyword(isLower ? "max" : "min"));

//   auto &builder = p.getBuilder();
//   auto boundAttrStrName =
//       isLower ? getLowerBoundMapAttrName(result.name)
//               : getUpperBoundMapAttrName(result.name);

//   // Parse ssa-id as identity map.
//   SmallVector<OpAsmParser::UnresolvedOperand, 1> boundOpInfos;
//   if (p.parseOperandList(boundOpInfos))
//     return failure();

//   if (!boundOpInfos.empty()) {
//     // Check that only one operand was parsed.
//     if (boundOpInfos.size() > 1)
//       return p.emitError(p.getNameLoc(),
//                          "expected only one loop bound operand");

//     // TODO: improve error message when SSA value is not of index type.
//     // Currently it is 'use of value ... expects different type than prior uses'
//     if (p.resolveOperand(boundOpInfos.front(), builder.getIndexType(),
//                          result.operands))
//       return failure();

//     // Create an identity map using symbol id. This representation is optimized
//     // for storage. Analysis passes may expand it into a multi-dimensional map
//     // if desired.
//     AffineMap map = builder.getSymbolIdentityMap();
//     result.addAttribute(boundAttrStrName, AffineMapAttr::get(map));
//     return success();
//   }

//   // Get the attribute location.
//   SMLoc attrLoc = p.getCurrentLocation();

//   Attribute boundAttr;
//   if (p.parseAttribute(boundAttr, builder.getIndexType(), boundAttrStrName,
//                        result.attributes))
//     return failure();

//   // Parse full form - affine map followed by dim and symbol list.
//   if (auto affineMapAttr = llvm::dyn_cast<AffineMapAttr>(boundAttr)) {
//     unsigned currentNumOperands = result.operands.size();
//     unsigned numDims;
//     if (parseDimAndSymbolList(p, result.operands, numDims))
//       return failure();

//     auto map = affineMapAttr.getValue();
//     if (map.getNumDims() != numDims)
//       return p.emitError(
//           p.getNameLoc(),
//           "dim operand count and affine map dim count must match");

//     unsigned numDimAndSymbolOperands =
//         result.operands.size() - currentNumOperands;
//     if (numDims + map.getNumSymbols() != numDimAndSymbolOperands)
//       return p.emitError(
//           p.getNameLoc(),
//           "symbol operand count and affine map symbol count must match");

//     // If the map has multiple results, make sure that we parsed the min/max
//     // prefix.
//     if (map.getNumResults() > 1 && failedToParsedMinMax) {
//       if (isLower) {
//         return p.emitError(attrLoc, "lower loop bound affine map with "
//                                     "multiple results requires 'max' prefix");
//       }
//       return p.emitError(attrLoc, "upper loop bound affine map with multiple "
//                                   "results requires 'min' prefix");
//     }
//     return success();
//   }

//   // Parse custom assembly form.
//   if (auto integerAttr = llvm::dyn_cast<IntegerAttr>(boundAttr)) {
//     result.attributes.pop_back();
//     result.addAttribute(
//         boundAttrStrName,
//         AffineMapAttr::get(builder.getConstantAffineMap(integerAttr.getInt())));
//     return success();
//   }

//   return p.emitError(
//       p.getNameLoc(),
//       "expected valid affine map representation for loop bounds");
// }

// ParseResult AffineForOp::parse(OpAsmParser &parser, OperationState &result) {
//   auto &builder = parser.getBuilder();
//   OpAsmParser::Argument inductionVariable;
//   inductionVariable.type = builder.getIndexType();
//   // Parse the induction variable followed by '='.
//   if (parser.parseArgument(inductionVariable) || parser.parseEqual())
//     return failure();

//   // Parse loop bounds.
//   int64_t numOperands = result.operands.size();
//   if (parseBound(/*isLower=*/true, result, parser))
//     return failure();
//   int64_t numLbOperands = result.operands.size() - numOperands;
//   if (parser.parseKeyword("to", " between bounds"))
//     return failure();
//   numOperands = result.operands.size();
//   if (parseBound(/*isLower=*/false, result, parser))
//     return failure();
//   int64_t numUbOperands = result.operands.size() - numOperands;

//   // Parse the optional loop step, we default to 1 if one is not present.
//   if (parser.parseOptionalKeyword("step")) {
//     result.addAttribute(
//         getStepAttrName(),
//         builder.getIntegerAttr(builder.getIndexType(), /*value=*/1));
//   } else {
//     SMLoc stepLoc = parser.getCurrentLocation();
//     IntegerAttr stepAttr;
//     if (parser.parseAttribute(stepAttr, builder.getIndexType(),
//                               getStepAttrName().data(),
//                               result.attributes))
//       return failure();

//     if (stepAttr.getValue().isNegative())
//       return parser.emitError(
//           stepLoc,
//           "expected step to be representable as a positive signed integer");
//   }

//   // Parse the optional initial iteration arguments.
//   SmallVector<OpAsmParser::Argument, 4> regionArgs;
//   SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

//   // Induction variable.
//   regionArgs.push_back(inductionVariable);

//   if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
//     // Parse assignment list and results type list.
//     if (parser.parseAssignmentList(regionArgs, operands) ||
//         parser.parseArrowTypeList(result.types))
//       return failure();
//     // Resolve input operands.
//     for (auto argOperandType :
//          llvm::zip(llvm::drop_begin(regionArgs), operands, result.types)) {
//       Type type = std::get<2>(argOperandType);
//       std::get<0>(argOperandType).type = type;
//       if (parser.resolveOperand(std::get<1>(argOperandType), type,
//                                 result.operands))
//         return failure();
//     }
//   }

// result.addAttribute("operandSegmentSizes",
//                   builder.getDenseI32ArrayAttr(
//                       {static_cast<int32_t>(outputs.size()),
//                         static_cast<int32_t>(lbOperands.size()),
//                         static_cast<int32_t>(ubOperands.size()),
//                         static_cast<int32_t>(iterArgs.size())}));

//   // Parse the body region.
//   Region *body = result.addRegion();
//   if (regionArgs.size() != result.types.size() + 1)
//     return parser.emitError(
//         parser.getNameLoc(),
//         "mismatch between the number of loop-carried values and results");
//   if (parser.parseRegion(*body, regionArgs))
//     return failure();

//   // AffineForOp::ensureTerminator(*body, builder, result.location);

//   // Parse the optional attribute list.
//   return parser.parseOptionalAttrDict(result.attributes);
// }

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

//===----------------------------------------------------------------------===//
// AllocOp 
//===----------------------------------------------------------------------===//

void AllocSPDOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "alloc_spd");
}