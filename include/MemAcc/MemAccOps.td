//===- MemAccOps.td - MemAcc dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MemAcc_OPS
#define MemAcc_OPS

include "MemAccDialect.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"  // This includes the definition for IntegerAttr and others


//===----------------------------------------------------------------------===//
// packed memory access operations outside of loops
//===----------------------------------------------------------------------===//
class MemAcc_PackedBaseOp<string mnemonic, list<Trait> traits = []> :
     MemAcc_Op<mnemonic, traits # [AttrSizedOperandSegments, AutomaticAllocationScope, 
    DeclareOpInterfaceMethods<LoopLikeOpInterface,
     ["getSingleInductionVar", "getSingleLowerBound", "getSingleStep",
      "getSingleUpperBound", "getYieldedValuesMutable",
      "replaceWithAdditionalYields"]>]>{
  let summary = "Defines a generic memory access pattern (will be hoisted outside of loops)";
  let description = [{
    The MemAcc_PackedBaseOp operation is designed to encapsulate any
    arbitrary memory access pattern, including stride, indirect, and other
    patterns. It will be transformed from a `MemAcc.generic_load` operation inside of
    affine.for. It takes indices array and data array as input, and it will allocate a 
    scratchpad buffer to store the loaded data. 
  }];

  let arguments = (ins Variadic<AnyShaped>:$bufs,
                       Variadic<Index>:$lowerBoundOperands,
                       Variadic<Index>:$upperBoundOperands,
                       Variadic<AnyType>:$inits,
                       AffineMapAttr:$lowerBoundMap,
                       AffineMapAttr:$upperBoundMap,
                       IndexAttr:$step,
                       I64Attr:$indirectionLevel);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$bufs, "int64_t":$lowerBound, "int64_t":$upperBound,
      CArg<"int64_t", "1">:$step, CArg<"ValueRange", "std::nullopt">:$iterArgs,
      CArg<"int64_t", "0">:$indirectionLevel,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>,
    OpBuilder<(ins "ValueRange":$bufs, "ValueRange":$lbOperands, "AffineMap":$lbMap,
      "ValueRange":$ubOperands, "AffineMap":$ubMap, CArg<"int64_t", "1">:$step,
      CArg<"ValueRange", "std::nullopt">:$iterArgs,
      CArg<"int64_t", "0">:$indirectionLevel,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

    let extraClassDeclaration = [{
    /// Defining the function type we use for building the body of affine.for.
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    static StringRef getIndirectionLevelAttrStrName() { return "indirectionLevel"; }
    static StringRef getStepAttrStrName() { return "step"; }
    static StringRef getLowerBoundAttrStrName() { return "lowerBoundMap"; }
    static StringRef getUpperBoundAttrStrName() { return "upperBoundMap"; }

    BlockArgument getInductionVar() { return getBody().getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody().getArguments().drop_front();
    }

    // TODO: provide iterators for the lower and upper bound operands
    // if the current access via getLowerBound(), getUpperBound() is too slow.

    /// Returns operands for the lower and upper bound maps with the operands
    /// for the lower bound map in front of those for the upper bound map.
    operand_range getControlOperands(){
      return {operand_begin(), operand_begin() + getLowerBoundOperands().size() +
                               getUpperBoundOperands().size()};
    }

    /// Set lower bound. The new bound must have the same number of operands as
    /// the current bound map. Otherwise, 'replaceForLowerBound' should be used.
    void setLowerBound(ValueRange lbOperands, AffineMap map){
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
    /// Set upper bound. The new bound must not have more operands than the
    /// current bound map. Otherwise, 'replaceForUpperBound' should be used.
    void setUpperBound(ValueRange ubOperands, AffineMap map){
      assert(ubOperands.size() == map.getNumInputs());
      assert(map.getNumResults() >= 1 && "bound map has at least one result");

      SmallVector<Value, 4> newOperands(getLowerBoundOperands());
      newOperands.append(ubOperands.begin(), ubOperands.end());
      auto iterOperands = getInits();
      newOperands.append(iterOperands.begin(), iterOperands.end());
      (*this)->setOperands(newOperands);

      (*this)->setAttr(getUpperBoundAttrStrName(), AffineMapAttr::get(map));
    }


    /// Returns information about the lower bound as a single object.
    affine::AffineBound getLowerBound();

    /// Returns information about the upper bound as a single object.
    affine::AffineBound getUpperBound();

    /// Returns loop step.
    int64_t getStepAsAPInt() {
      return ::llvm::cast<IntegerAttr>(*(*this)->getInherentAttr(getStepAttrStrName())).getInt();
    }

    /// Set loop step.
    void setStep(int64_t step) {
      assert(step > 0 && "step has to be a positive integer constant");
      auto *context = getLowerBoundMap().getContext();
      (*this)->setAttr(StringAttr::get(context, getStepAttrStrName()),
                       IntegerAttr::get(IndexType::get(context), step));
    }

    /// Returns number of region arguments for loop-carried values.
    unsigned getNumRegionIterArgs() {
      return getBody().getNumArguments() - 1;
    }

    /// Number of operands controlling the loop: lb and ub.
    unsigned getNumControlOperands() { return getOperation()->getNumOperands() - getNumIterOperands(); }

    /// Get the number of loop-carried values.
    unsigned getNumIterOperands();

    /// Returns true if the lower bound is constant.
    bool hasConstantLowerBound(){
      return getLowerBoundMap().isSingleConstant();
    }
    /// Returns true if the upper bound is constant.
    bool hasConstantUpperBound(){
      return getUpperBoundMap().isSingleConstant();
    }
    /// Returns true if both bounds are constant.
    bool hasConstantBounds() {
      return hasConstantLowerBound() && hasConstantUpperBound();
    }
    /// Returns the value of the constant lower bound.
    /// Fails assertion if the bound is non-constant.
    int64_t getConstantLowerBound(){
      return getLowerBoundMap().getSingleConstantResult();
    }
    /// Returns the value of the constant upper bound. The upper bound is
    /// exclusive. Fails assertion if the bound is non-constant.
    int64_t getConstantUpperBound(){
      return getUpperBoundMap().getSingleConstantResult();
    }
    /// Sets the lower bound to the given constant value.
    void setConstantLowerBound(int64_t value){
      setLowerBound({}, AffineMap::getConstantMap(value, getContext()));
    }
    /// Sets the upper bound to the given constant value.
    void setConstantUpperBound(int64_t value){
      setUpperBound({}, AffineMap::getConstantMap(value, getContext()));
    }

    /// Returns true if both the lower and upper bound have the same operand
    /// lists (same operands in the same order).
    bool matchingBoundOperandList(){
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

  }];
}

//===----------------------------------------------------------------------===//
// packed op load outside of loops
//===----------------------------------------------------------------------===//
def MemAcc_PackedGenericLoadOp : MemAcc_PackedBaseOp<"packed_generic_load"> {
  let summary = "Defines a generic gather pattern (will be hoisted outside of loops)";
  let description = [{
    The `MemAcc.packed_generic_load` operation is designed to encapsulate any
    arbitrary load access pattern, including stride, indirect, and other
    patterns. It will be transformed from a `MemAcc.generic_load` operation inside of
    affine.for. It takes indices array and data array as input, and it will allocate a 
    scratchpad buffer to store the loaded data. 

    Example:

    ```mlir
    affine.for %arg5 = 0 to %1 {

      // gather a[idx[i]]
      %5 = memref.load %arg2[%arg5] : memref<?xi32>
      %6 = memref.index_cast %5 : i32 to index
      %7 = memref.load %arg1[%6] : memref<?xf64>

      compute(%7)
      %8 = memref.load %arg2[%arg5] : memref<?xi32> idx[i]
      memref.store %3, %arg0[%8] : memref<?xf64> store a[idx[i]]
    }
    ```

    will be transformed into

    ```mlir
    %4 = memacc.spd_allocate %1 : memref<?xf64>
    memacc.packed_generic_load %i = 0 to %1  outs(%4 : memref<?xf64>) {
        %idx = memref.load %arg2[%i] : memref<?xi32>
        %idx_ = memacc.index_cast %idx : i32 to index
        %5 = memref.load %arg1[%idx_] : memref<?xf64>
        memacc.yield %5 : f64
    }
    affine.for %arg5 = 0 to %1 {
      %3 = memacc.load %4[%arg5] : memref<?xf64>
      memacc.store %3, %arg0[%arg5] : memref<?xf64>
    }
    ```
  }];
}


//===----------------------------------------------------------------------===//
// packed op store outside of loops
//===----------------------------------------------------------------------===//
def MemAcc_PackedGenericStoreOp : MemAcc_PackedBaseOp<"packed_generic_store"> {
  let summary = "Defines a generic scatter pattern (will be sinked outside of loops)";
  let description = [{
    The `MemAcc.packed_generic_store` operation is designed to encapsulate any
    arbitrary scatter pattern, including stride, indirect, and other
    patterns. It will be transformed from a `MemAcc.generic_load` operation inside of
    affine.for. It takes indices array and data array as input, and it will allocate a 
    scratchpad buffer to store the loaded data. 

    Example:TODO: 9Tempest: provide an example
  }];
}

//===----------------------------------------------------------------------===//
// packed op rmw outside of loops
//===----------------------------------------------------------------------===//
def MemAcc_PackedGenericRmwOp : MemAcc_PackedBaseOp<"packed_generic_rmw"> {
  let summary = "Defines a generic scatter pattern (will be sinked outside of loops)";
  let description = [{
    The `MemAcc.packed_generic_rmw` operation is designed to encapsulate any
    arbitrary scatter pattern, including stride, indirect, and other
    patterns. It will be transformed from a `MemAcc.generic_load` operation inside of
    affine.for. It takes indices array and data array as input, and it will allocate a 
    scratchpad buffer to rmw the loaded data. 

    Example:TODO: 9Tempest: provide an example
  }];
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def MemAcc_LoadOp : MemAcc_Op<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "::llvm::cast<MemRefType>($_self).getElementType()">]> {
  let summary = "load operation";
  let description = [{
    The `load` op reads an element from a memref specified by an index list. The
    output of load is a new value with the same type as the elements of the
    memref. The arity of indices is the rank of the memref (i.e., if the memref
    loaded from is of rank 3, then 3 indices are required for the load following
    the memref identifier).

    In an `affine.if` or `affine.for` body, the indices of a load are restricted
    to SSA values bound to surrounding loop induction variables,
    [symbols](Affine.md/#dimensions-and-symbols), results of a
    constant operations, or the result of an
    `affine.apply` operation that can in turn take as arguments all of the
    aforementioned SSA values or the recursively result of such an
    `affine.apply` operation.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemRef().getType());
    }
  }];

  // let hasFolder = 1;
  // let hasVerifier = 1;

  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def MemAcc_StoreOp : MemAcc_Op<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "::llvm::cast<MemRefType>($_self).getElementType()">]> {
  let summary = "store operation";
  let description = [{
    Store a value to a memref location given by indices. The value stored should
    have the same type as the elemental type of the memref. The number of
    arguments provided within brackets need to match the rank of the memref.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       Variadic<Index>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
    }]>];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return ::llvm::cast<MemRefType>(getMemRef().getType());
      }
  }];

  // let hasFolder = 1;
  // let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

def MemAcc_RMWOp : MemAcc_Op<"rmw", [
      AllTypesMatch<["value", "result"]>,
      TypesMatchWith<"value type matches element type of memref",
                     "memref", "value",
                     "::llvm::cast<MemRefType>($_self).getElementType()">
    ]> {
  let summary = "atomic read-modify-write operation";
  let description = [{
    The `MemAcc.rmw` operation provides a way to perform a read-modify-write
    sequence that is free from data races. The kind enumeration specifies the
    modification to perform. The value operand represents the new value to be
    applied during the modification. The memref operand represents the buffer
    that the read and write will be performed against, as accessed by the
    specified indices. The arity of the indices is the rank of the memref. The
    result represents the latest value that was stored.

    Example:

    ```mlir
    %x = memacc.rmw "addf" %value, %I[%i] : (f32, memref<10xf32>) -> f32
    ```
  }];

  let arguments = (ins
      AtomicRMWKindAttr:$kind,
      AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$value,
      Arg<MemRefOf<[AnySignlessInteger, AnyFloat]>, "the reference to read from and write to", [MemRead, MemWrite]>:$memref,
      Variadic<Index>:$indices);
  let results = (outs AnyTypeOf<[AnySignlessInteger, AnyFloat]>:$result);

  let assemblyFormat = [{
    $kind $value `,` $memref `[` $indices `]` attr-dict `:` `(` type($value) `,`
    type($memref) `)` `->` type($result)
  }];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemref().getType());
    }
  }];
  // let hasFolder = 1;
  // let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// MemAcc_YieldOp
//===----------------------------------------------------------------------===//

def MemAcc_YieldOp : MemAcc_Op<"yield", [Terminator, Pure]> {
  let summary = "MemAcc dialect yield operation";
  let description = [{
    The `MemAcc.yield` operation serves as a terminator for blocks within MemAcc
    operations, indicating the end of a block's execution path. It can optionally
    return a memref to the enclosing operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "$operands attr-dict `:` functional-type(operands, results)";
}

// Base class for integer and floating point arithmetic ops. All ops have one
// result, require operands and results to be of the same type, and can accept
// tensors or vectors of integers or floats.
class MemAcc_ArithOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_Op<mnemonic, traits # [SameOperandsAndResultType, NoMemoryEffect] #
    ElementwiseMappable.traits>;

// Base class for unary arithmetic operations.
class MemAcc_UnaryOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_ArithOp<mnemonic, traits # [Pure]> {
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for binary arithmetic operations.
class MemAcc_BinaryOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_ArithOp<mnemonic, traits> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for integer binary operations.
class MemAcc_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_BinaryOp<mnemonic, traits>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)>,
    Results<(outs SignlessIntegerLike:$result)>;

// Base class for integer binary operations without undefined behavior.
class MemAcc_TotalIntBinaryOp<string mnemonic, list<Trait> traits = []> :
    MemAcc_IntBinaryOp<mnemonic, traits # [Pure]>;

// Index transformation operations
//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def MemAcc_AddIOp : MemAcc_TotalIntBinaryOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    Performs N-bit addition on the operands. The operands are interpreted as 
    unsigned bitvectors. The result is represented by a bitvector containing the 
    mathematical value of the addition modulo 2^n, where `n` is the bitwidth. 
    Because `arith` integers use a two's complement representation, this operation 
    is applicable on both signed and unsigned integer operands.
  }];
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def MemAcc_SubIOp : MemAcc_TotalIntBinaryOp<"subi"> {
  let summary = [{
    Integer subtraction operation.
  }];
  let description = [{
    Performs N-bit subtraction on the operands. The operands are interpreted as unsigned
    bitvectors. The result is represented by a bitvector containing the mathematical
    value of the subtraction modulo 2^n, where `n` is the bitwidth. Because `arith`
    integers use a two's complement representation, this operation is applicable on
    both signed and unsigned integer operands.
  }];
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

def MemAcc_MulIOp : MemAcc_TotalIntBinaryOp<"muli", [Commutative]> {
  let summary = [{
    Integer multiplication operation.
  }];
  let description = [{
    Performs N-bit multiplication on the operands. The operands are interpreted as
    unsigned bitvectors. The result is represented by a bitvector containing the
    mathematical value of the multiplication modulo 2^n, where `n` is the bitwidth.
    Because `arith` integers use a two's complement representation, this operation is
    applicable on both signed and unsigned integer operands.
  }];
  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// Index cast can convert between memrefs of signless integers and indices too.
def IndexCastTypeConstraint : TypeConstraint<Or<[
        SignlessIntegerLike.predicate,
        MemRefOf<[AnySignlessInteger, Index]>.predicate]>,
    "signless-integer-like or memref of signless-integer">;

// Base class for arithmetic cast operations. Requires a single operand and
// result. If either is a shaped type, then the other must be of the same shape.
class MemAcc_CastOp<string mnemonic, TypeConstraint From, TypeConstraint To,
                   list<Trait> traits = []> :
    MemAcc_Op<mnemonic, traits # [Pure, SameOperandsAndResultShape,
      DeclareOpInterfaceMethods<CastOpInterface>]>,
    Arguments<(ins From:$in)>,
    Results<(outs To:$out)> {
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

def MemAcc_IndexCastOp
  : MemAcc_CastOp<"index_cast", IndexCastTypeConstraint, IndexCastTypeConstraint> {
  let summary = "cast between index and integer types";
  let description = [{
    Casts between scalar or vector integers and corresponding 'index' scalar or
    vectors. Index is an integer of platform-specific bit width. If casting to
    a wider integer, the value is sign-extended. If casting to a narrower
    integer, the value is truncated.
  }];

  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// AllocLikeOp
//===----------------------------------------------------------------------===//

// Base class for memref allocating ops: alloca and alloc.
//
//   %0 = alloclike(%m)[%s] : memref<8x?xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>>
//
class AllocLikeOp<string mnemonic,
                  Resource resource,
                  list<Trait> traits = []> :
    MemAcc_Op<mnemonic,
    !listconcat([
      AttrSizedOperandSegments
    ], traits)> {

  let arguments = (ins Variadic<Index>:$dynamicSizes,
                       // The symbolic operands (the ones in square brackets)
                       // bind to the symbols of the memref's layout map.
                       Variadic<Index>:$symbolOperands,
                       ConfinedAttr<OptionalAttr<I64Attr>,
                                [IntMinValue<0>]>:$alignment);
  let results = (outs Res<AnyMemRef, "", [MemAlloc<resource>]>:$memref);

  let builders = [
    OpBuilder<(ins "MemRefType":$memrefType,
                  CArg<"IntegerAttr", "IntegerAttr()">:$alignment), [{
      return build($_builder, $_state, memrefType, {}, alignment);
    }]>,
    OpBuilder<(ins "MemRefType":$memrefType, "ValueRange":$dynamicSizes,
                  CArg<"IntegerAttr", "IntegerAttr()">:$alignment), [{
      return build($_builder, $_state, memrefType, dynamicSizes, {}, alignment);
    }]>,
    OpBuilder<(ins "MemRefType":$memrefType, "ValueRange":$dynamicSizes,
                  "ValueRange":$symbolOperands,
                  CArg<"IntegerAttr", "{}">:$alignment), [{
      $_state.types.push_back(memrefType);
      $_state.addOperands(dynamicSizes);
      $_state.addOperands(symbolOperands);
      $_state.addAttribute(getOperandSegmentSizeAttr(),
          $_builder.getDenseI32ArrayAttr({
              static_cast<int32_t>(dynamicSizes.size()),
              static_cast<int32_t>(symbolOperands.size())}));
      if (alignment)
        $_state.addAttribute(getAlignmentAttrStrName(), alignment);
    }]>,
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$sizes, "Type":$elementType,
                   CArg<"Attribute", "{}">:$memorySpace), [{
      SmallVector<int64_t> staticShape;
      SmallVector<Value> dynamicSizes;
      dispatchIndexOpFoldResults(sizes, dynamicSizes, staticShape);
      MemRefLayoutAttrInterface layout;
      MemRefType memrefType = MemRefType::get(staticShape, elementType, layout,
                                              memorySpace);
      return build($_builder, $_state, memrefType, dynamicSizes);
    }]>
  ];

  let extraClassDeclaration = [{
    static StringRef getAlignmentAttrStrName() { return "alignment"; }

    MemRefType getType() { return ::llvm::cast<MemRefType>(getResult().getType()); }

    SmallVector<OpFoldResult> getMixedSizes() {
      SmallVector<OpFoldResult> result;
      unsigned ctr = 0;
      OpBuilder b(getContext());
      for (int64_t i = 0, e = getType().getRank(); i < e; ++i) {
        if (getType().isDynamicDim(i)) {
          result.push_back(getDynamicSizes()[ctr++]);
        } else {
          result.push_back(b.getIndexAttr(getType().getShape()[i]));
        }
      }
      return result;
    }
  }];

  let assemblyFormat = [{
    `(`$dynamicSizes`)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

def MemAcc_AllocSPDOp : AllocLikeOp<"alloc_spd", DefaultResource, [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "memory allocation operation";
  let description = [{
    The `alloc` operation allocates a region of memory, as specified by its
    memref type.

    Example:

    ```mlir
    %0 = memacc.alloc_spd() : memref<8x64xf32, 1>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the ssa value '%d' is
    bound to the second dimension of the memref (which is dynamic).

    ```mlir
    %0 = memacc.alloc_spd(%d) : memref<8x?xf32, 1>
    ```

    The optional list of symbol operands are bound to the symbols of the
    memrefs affine map. In the example below, the ssa value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

    ```mlir
    %0 = memacc.alloc_spd()[%s] : memref<8x64xf32,
                              affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
    ```

    This operation returns a single ssa value of memref type, which can be used
    by subsequent load and store operations.

    The optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary.

    ```mlir
    %0 = memacc.alloc_spd()[%s] {alignment = 8} :
      memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
    ```
  }];
}

#endif // MemAcc_OPS