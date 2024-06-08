#include "MemAcc/Dialect.h"
#include "MemAcc/Passes/MemAccAnalysis.h"
#include "MemAcc/Passes/Passes.h"
#include "PassDetails.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "MemAcc/Passes/MemAccUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "MemAcc/Passes/MemAccUtils.h"

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

llvm::DenseMap<Operation *, DFS::GatherPath> forOpToGatherPath;
llvm::DenseMap<Operation *, DFS::ScatterPath> forOpToScatterPath;
llvm::DenseMap<Operation *, DFS::RMWPath> forOpToRMWPath;
llvm::DenseMap<Operation *, bool> forOpDone;

static inline bool isUnderPackOp(Operation *op) {
  return op->getParentOfType<MemAcc::PackedGenericLoadOp>() ||
         op->getParentOfType<MemAcc::PackedGenericStoreOp>() ||
         op->getParentOfType<MemAcc::PackedGenericRmwOp>();
}

static void getForToIndirectAccess(Operation *op) {
  // DFS to find all gather traces and scatter traces
  // for all AffineForOps
  op->walk([&](Operation *currentOp) {
    if (isa<affine::AffineForOp>(currentOp)) {
      DFS dfs;
      dfs.analyzeLoadOps<affine::AffineForOp>(
          dyn_cast<affine::AffineForOp>(currentOp));
      forOpToGatherPath[currentOp] = dfs.getGatherPath();
      forOpToScatterPath[currentOp] = dfs.getScatterPath();
      forOpToRMWPath[currentOp] = dfs.getRMWPath();
      forOpDone[currentOp] = false;
    }
  });

  // Print forOp -> GatherPath/SactterPath
  for (auto &forOpGatherPath : forOpToGatherPath) {
    PRINT("ForOp:");
    PRINT(*forOpGatherPath.first);

    assert(forOpToGatherPath.count(forOpGatherPath.first) == 1);
    assert(forOpToScatterPath.count(forOpGatherPath.first) == 1);
    forOpGatherPath.second.print();
    forOpToScatterPath[forOpGatherPath.first].print();
    assert(forOpToRMWPath.count(forOpGatherPath.first) == 1);
    forOpToRMWPath[forOpGatherPath.first].print();
  }
}

template <typename SrcOpType, typename DestOpType>
class ConvertArithToMemAccPattern : public OpRewritePattern<SrcOpType> {
public:
  using OpRewritePattern<SrcOpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOpType op,
                                PatternRewriter &rewriter) const override {
    if (isUnderPackOp(op)) {
      rewriter.replaceOpWithNewOp<DestOpType>(op, op.getResult().getType(),
                                              op.getOperands());
      return success();
    }
    return failure();
  }
};

class ConvertArithIndexCastToMemAccIndexCastPattern
    : public OpRewritePattern<arith::IndexCastOp> {
public:
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (isUnderPackOp(op)) {
      rewriter.replaceOpWithNewOp<MemAcc::IndexCastOp>(
          op, op.getResult().getType(), op.getOperand());
      return success();
    }
    return failure();
  }
};

template <typename LoadOpType>
struct StoreOpConversionPattern : public OpRewritePattern<LoadOpType> {
  using OpRewritePattern<LoadOpType>::OpRewritePattern;

  void rewriteStoreOp(LoadOpType storeOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<MemAcc::StoreOp>(
        storeOp, storeOp.getValue(), storeOp.getMemRef(), storeOp.getIndices());
  }

  LogicalResult matchAndRewrite(LoadOpType storeOp,
                                PatternRewriter &rewriter) const override {
    if (isUnderPackOp(storeOp)) {
      rewriteStoreOp(storeOp, rewriter);
      return success();
    } else {
      return failure();
    }
  }
};

template <typename LoadOpType>
struct LoadOpConversionPattern : public OpRewritePattern<LoadOpType> {
  using OpRewritePattern<LoadOpType>::OpRewritePattern;

  void rewriteLoadOp(LoadOpType loadOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<MemAcc::LoadOp>(loadOp, loadOp.getMemRef(),
                                                loadOp.getIndices());
  }

  LogicalResult matchAndRewrite(LoadOpType loadOp,
                                PatternRewriter &rewriter) const override {
    if (isUnderPackOp(loadOp)) {
      rewriteLoadOp(loadOp, rewriter);
      return success();
    } else {
      return failure();
    }
  }
};

// Define the pattern to hoist generic indirect loads outside of loops
class PackGenericLoadOpOutsideLoop
    : public OpRewritePattern<affine::AffineForOp> {
public:
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  Value getForLoopLowerBound(PatternRewriter &rewriter,
                             AffineForOp forOp) const {
    auto lb = forOp.getLowerBoundOperands();
    auto lbMap = forOp.getLowerBoundMap();
    auto lbResult = getAffineMapMax(rewriter, forOp.getLoc(), lbMap, lb);
    return lbResult;
  }

  Value getForLoopUpperBound(PatternRewriter &rewriter,
                             AffineForOp forOp) const {
    auto ub = forOp.getUpperBoundOperands();
    auto ubMap = forOp.getUpperBoundMap();
    auto ubResult = getAffineMapMin(rewriter, forOp.getLoc(), ubMap, ub);
    return ubResult;
  }

  Value getForLoopLength(PatternRewriter &rewriter,Value upperBound, Value lowerBound) const {
    // auto lb = forOp.getLowerBound();
    auto length = rewriter.create<arith::SubIOp>(upperBound.getLoc(), upperBound, lowerBound);
    return length;
  }

  void generatePackedMemAccRmwOp(
      Location loc, PatternRewriter &rewriter,
      llvm::DenseMap<Operation *, mlir::Value> &rmwToAllocs, AffineForOp forOp,
      DFS::RMWPath &rmwPath, unsigned int indirectLevel, Value loopUpperBound, Value loopLowerBound) const {

    const DFS::IndirectChain &indirectChain =
        rmwPath.indirectChain;
    llvm::SmallVector<mlir::Value> alloc_spds;
    for (auto storeToAlloc : rmwToAllocs) {
      alloc_spds.push_back(storeToAlloc.second);
    }
    llvm::ArrayRef<mlir::Value> alloc_spds_ref(alloc_spds);
    // Create PackedOpType outside of the loop
    auto packedRMWOp = rewriter.create<MemAcc::PackedGenericRmwOp>(
        loc, alloc_spds_ref, ValueRange({loopLowerBound}),
        forOp.getLowerBoundMap(), ValueRange({loopUpperBound}),
        forOp.getUpperBoundMap(), forOp.getStep(), forOp.getInits(),
        indirectLevel);

    // for each instruction in GenericLoadOp, clone them into
    // PackedOpType, replace all users of AffineForOp's induction
    // variable with the induction variable of PackedOpType
    auto newInductionVar = packedRMWOp.getInductionVar();
    auto originalInductionVar =
        forOp.getInductionVar(); // Get this from the original AffineForOp
                                 // context
    rewriter.setInsertionPointToStart(&packedRMWOp.getBody().front());
    // record the mapping of old instruction -> new instruction
    DenseMap<Operation *, Value> opToResultMap;
    for (auto [I, condOp, condBranch] : indirectChain) {
      Operation *newI;
      // If the instruction is a store op, replace the store value with the load
      // to new induction variable
      if (isa<memref::StoreOp>(I) || isa<affine::AffineStoreOp>(I)) {
        // 1. create load op to load from modified data
        assert(rmwToAllocs.count(I) &&
               "rmwToAllocs does not contain the store op");
        auto loadOp = rewriter.create<memref::LoadOp>(loc, rmwToAllocs[I],
                                                      newInductionVar);
        // 2. create rmw op
        auto rmwOp = rmwPath.storeToRmwOp[I];
        rewriter.create<MemAcc::RMWOp>(
            loc, rmwOp.opKind, loadOp.getResult(), rmwOp.baseAddress,
            opToResultMap[rmwOp.addressOffset.getDefiningOp()]);
        continue;
      } else {
        newI = rewriter.clone(*I);
      }

      // record the mapping of old instruction -> new instruction
      if (newI->getNumResults() > 0) {
        opToResultMap[I] = newI->getResult(0);
      }

      // replace the operands of the new instruction with the new induction
      // variable/new results
      for (unsigned idx = 0; idx < newI->getNumOperands(); ++idx) {
        // Check if the operand is the original induction variable
        if (newI->getOperand(idx) == originalInductionVar) {
          // Replace the operand with the new induction variable
          newI->setOperand(idx, newInductionVar);
        } else if (opToResultMap.count(newI->getOperand(idx).getDefiningOp())) {
          newI->setOperand(
              idx, opToResultMap[newI->getOperand(idx).getDefiningOp()]);
        }
      }
    }

    // create empty yield op as a terminator
    rewriter.create<MemAcc::YieldOp>(loc, SmallVector<Type>{},
                                     SmallVector<Value>{});
    // PRINT("PackedRMWOp:");
    // PRINT(packedRMWOp);
  }

  void generatePackedMemAccStoreOp(
      Location loc, PatternRewriter &rewriter,
      llvm::DenseMap<Operation *, mlir::Value> &storeToAllocs,
      AffineForOp forOp, const DFS::ScatterPath &scatterPath,
      unsigned int indirectLevel, Value loopUpperBound, Value loopLowerBound) const {
    const DFS::IndirectChain &indirectChain =
        scatterPath.indirectChain;
    llvm::SmallVector<mlir::Value> alloc_spds;
    for (auto storeToAlloc : storeToAllocs) {
      alloc_spds.push_back(storeToAlloc.second);
    }
    llvm::ArrayRef<mlir::Value> alloc_spds_ref(alloc_spds);
    // Create PackedOpType outside of the loop
    auto packedStoreOp = rewriter.create<MemAcc::PackedGenericStoreOp>(
        loc, alloc_spds_ref, ValueRange({loopLowerBound}),
        forOp.getLowerBoundMap(), ValueRange({loopUpperBound}),
        forOp.getUpperBoundMap(), forOp.getStep(), forOp.getInits(),
        indirectLevel);

    // for each instruction in GenericLoadOp, clone them into
    // PackedOpType, replace all users of AffineForOp's induction
    // variable with the induction variable of PackedOpType
    auto newInductionVar = packedStoreOp.getInductionVar();
    auto originalInductionVar =
        forOp.getInductionVar(); // Get this from the original AffineForOp
                                 // context
    rewriter.setInsertionPointToStart(&packedStoreOp.getBody().front());

    // record the mapping of old instruction -> new instruction
    DenseMap<Operation *, Value> opToResultMap;
    for (auto [I, condOp, condBranch] : indirectChain) {
      Operation *newI;
      // If the instruction is a store op, replace the store value with the load
      // to new induction variable
      if (isa<memref::StoreOp>(I) || isa<affine::AffineStoreOp>(I)) {
        // 1. create load op
        assert(storeToAllocs.count(I) &&
               "storeToAllocs does not contain the store op");
        auto loadOp = rewriter.create<memref::LoadOp>(loc, storeToAllocs[I],
                                                      newInductionVar);
        newI = rewriter.clone(*I);
        // 2. replace the store value with the load op
        newI->setOperand(0, loadOp.getResult());
      } else {
        newI = rewriter.clone(*I);
      }

      // record the mapping of old instruction -> new instruction
      if (newI->getNumResults() > 0) {
        opToResultMap[I] = newI->getResult(0);
      }

      // replace the operands of the new instruction with the new induction
      // variable/new results
      for (unsigned idx = 0; idx < newI->getNumOperands(); ++idx) {
        // Check if the operand is the original induction variable
        if (newI->getOperand(idx) == originalInductionVar) {
          // Replace the operand with the new induction variable
          newI->setOperand(idx, newInductionVar);
        } else if (opToResultMap.count(newI->getOperand(idx).getDefiningOp())) {
          newI->setOperand(
              idx, opToResultMap[newI->getOperand(idx).getDefiningOp()]);
        }
      }
    }

    // create empty yield op as a terminator
    rewriter.create<MemAcc::YieldOp>(loc, SmallVector<Type>{},
                                     SmallVector<Value>{});
  }

  void generatePackedMemAccLoadOp(
      Location loc, PatternRewriter &rewriter,
      llvm::DenseMap<Operation *, mlir::Value> &loadToAllocs, AffineForOp forOp,
      const DFS::GatherPath &gatherPath, unsigned int indirectLevel, Value loopUpperBound, Value loopLowerBound) const {

    const DFS::IndirectChain &indirectChain = gatherPath.indirectChain;
    llvm::SmallVector<mlir::Value> alloc_spds;
    for (auto loadToAlloc : loadToAllocs) {
      alloc_spds.push_back(loadToAlloc.second);
    }
    llvm::ArrayRef<mlir::Value> alloc_spds_ref(alloc_spds);
    // Create PackedOpType outside of the loop
    auto packedLoadOp = rewriter.create<MemAcc::PackedGenericLoadOp>(
        loc, alloc_spds_ref, ValueRange({loopLowerBound}),
        forOp.getLowerBoundMap(), ValueRange({loopUpperBound}),
        forOp.getUpperBoundMap(), forOp.getStep(), forOp.getInits(),
        indirectLevel);

    // for each instruction in GenericLoadOp, clone them into
    // PackedOpType, replace all users of AffineForOp's induction
    // variable with the induction variable of PackedOpType
    auto newInductionVar = packedLoadOp.getInductionVar();
    auto originalInductionVar =
        forOp.getInductionVar(); // Get this from the original AffineForOp
                                 // context
    rewriter.setInsertionPointToStart(&packedLoadOp.getBody().front());

    // record the mapping of old instruction -> new instruction
    DenseMap<Operation *, Value> opToResultMap;
    for (auto [I, condOp, condBranch] : indirectChain) {
      auto newI = rewriter.clone(*I);
      opToResultMap[I] = newI->getResult(0);
      for (unsigned idx = 0; idx < newI->getNumOperands(); ++idx) {
        // Check if the operand is the original induction variable
        if (newI->getOperand(idx) == originalInductionVar) {
          // Replace the operand with the new induction variable
          newI->setOperand(idx, newInductionVar);
        } else if (opToResultMap.count(newI->getOperand(idx).getDefiningOp())) {
          newI->setOperand(
              idx, opToResultMap[newI->getOperand(idx).getDefiningOp()]);
        }
      }
    }
    /// Step3: Insert a memacc.yield op at the end of the packed load op
    // result type of the yield is the element type of each spd buffer
    SmallVector<Type, 4> resultTypes;
    SmallVector<Value, 4> resultVals;
    // result val are the keys of external users(deepest load)
    for (auto &loadToAlloc : loadToAllocs) {
      auto loadOp = loadToAlloc.first;
      assert(gatherPath.deepestLoadToExternUsers.count(loadOp) &&
             "externalUsers does not contain the load op");
      auto spdBuffer = loadToAllocs[loadOp];
      if (auto memrefType = spdBuffer.getType().dyn_cast<MemRefType>()) {
        resultTypes.push_back(memrefType.getElementType());
      } else {
        assert(false && "Unexpected type for spd buffer");
      }
      resultVals.push_back(opToResultMap[loadOp]);
    }

    rewriter.create<MemAcc::YieldOp>(loc, resultTypes, resultVals);
  }

  Value getSpdBuffer(Type resultType, PatternRewriter &rewriter,
                     Value loopLength, Location loc) const {
    // Determine the element type, whether resultType is already a memref or
    // not
    SmallVector<int64_t, 4> shape{mlir::ShapedType::kDynamic};
    mlir::Type elementType;
    if (auto memrefType = resultType.dyn_cast<mlir::MemRefType>()) {
      // If resultType is a MemRefType, extract its element type
      elementType = memrefType.getElementType();
      shape.append(memrefType.getShape().begin(), memrefType.getShape().end());
    } else {
      // Otherwise, use resultType directly as the element type
      elementType = resultType;
    }
    // Create a MemRefType with a dynamic size in the first dimension and the
    // obtained element type
    auto memRefType = mlir::MemRefType::get(shape, elementType);
    return rewriter.create<AllocSPDOp>(loc, memRefType,
                                       ValueRange({loopLength}));
  }

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOpDone[forOp]) {
      return failure();
    }
    bool hasGatherPath = (forOpToGatherPath[forOp].indirectChain.size() != 0);
    bool hasScatterPath = (forOpToScatterPath[forOp].indirectChain.size() != 0);
    bool hasRMWPath = (forOpToRMWPath[forOp].indirectChain.size() != 0);
    auto InductionVar = forOp.getInductionVar();
    if (!hasGatherPath && !hasScatterPath && !hasRMWPath) {
      forOpDone[forOp] = true;
      return failure();
    }

    // Step1: Create alloc_spd ops before affine.for loop
    // the size of the alloc_spd is the same as the loop length
    // the type of the alloc_spd is the same as the result type of the generic
    // op also record the result idx that used for store idx
    rewriter.setInsertionPoint(forOp);
    Value loopUpperBound = getForLoopUpperBound(rewriter, forOp);
    Value loopLowerBound = getForLoopLowerBound(rewriter, forOp);
    auto loopLength = getForLoopLength(rewriter, loopUpperBound, loopLowerBound);

    rewriter.setInsertionPoint(&forOp.getBody()->front());
    auto spdIndex = rewriter.create<arith::SubIOp>(forOp.getLoc(), InductionVar, loopLowerBound).getResult();
    llvm::DenseMap<Operation *, Value> spdBufferGather;
    llvm::DenseMap<Operation *, Value> spdBufferScatter;
    llvm::DenseMap<Operation *, Value> spdBufferRMW;
    if (hasGatherPath) {
      // create spd buffer for external users of gather path
      for (auto &opToUserPair :
           forOpToGatherPath[forOp].deepestLoadToExternUsers) {
        rewriter.setInsertionPoint(forOp);
        for (unsigned int i = 0; i < opToUserPair.second.users.size(); i++) {
          auto user = opToUserPair.second.users[i];
          unsigned int operandIdx = opToUserPair.second.operandIdx[i];
          assert(opToUserPair.second.users.size() ==
                     opToUserPair.second.operandIdx.size() &&
                 "Mismatch between users and operandIdx");
          assert(operandIdx < user->getNumOperands() &&
                 "operandIdx out of bound");
          auto operand = user->getOperand(operandIdx);
          Value spdBuffer = getSpdBuffer(operand.getType(), rewriter,
                                         loopLength, forOp->getLoc());
          rewriter.setInsertionPoint(user);
          auto newOperand =
              rewriter
                  .create<memref::LoadOp>(forOp.getLoc(), spdBuffer,
                                          ValueRange({spdIndex}))
                  .getResult();
          user->setOperand(operandIdx, newOperand);
          spdBufferGather[opToUserPair.first] = spdBuffer;
        }
      } // end for

      /// Step2: Create PackedGenericLoadOp and hoist outside of the loop
      rewriter.setInsertionPoint(forOp);
      generatePackedMemAccLoadOp(forOp.getLoc(), rewriter, spdBufferGather,
                                 forOp, forOpToGatherPath[forOp],
                                 forOpToGatherPath[forOp].indirectDepth, loopUpperBound, loopLowerBound);
    }
    if (hasRMWPath) {
      rewriter.setInsertionPoint(forOp);
      /// Step1: create spd buffer for store values of rmw
      for (auto &opToRmwOpPair : forOpToRMWPath[forOp].storeToRmwOp) {
        Value spdBuffer =
            getSpdBuffer(opToRmwOpPair.second.modifiedValue.getType(), rewriter,
                         loopLength, forOp->getLoc());
        // replace the original StoreOp to store to spd buffer
        // assume last instruction is a StoreOp
        spdBufferRMW[opToRmwOpPair.first] = spdBuffer;
      }

      /// Step2: Generate packed store op and sink outside after the loop
      rewriter.setInsertionPointAfter(forOp);
      generatePackedMemAccRmwOp(forOp.getLoc(), rewriter, spdBufferRMW, forOp,
                                forOpToRMWPath[forOp], 1, loopUpperBound, loopLowerBound);
      /// Step3: Change all store operation to store the modified value to spd
      /// buffer
      for (auto &opToRmwOpPair : forOpToRMWPath[forOp].storeToRmwOp) {
        // change value to modifiedValue
        // if modifiedValue is a load op and can map to spd buffer, replace it
        // with load from spd buffer
        auto modifiedValueOp =
            opToRmwOpPair.second.modifiedValue.getDefiningOp();
        rewriter.setInsertionPoint(opToRmwOpPair.first);
        if (spdBufferGather.count(modifiedValueOp)) {
          auto loadOp = rewriter.create<memref::LoadOp>(
              forOp.getLoc(), spdBufferGather[modifiedValueOp], spdIndex);
          opToRmwOpPair.second.modifiedValue = loadOp.getResult();
        }
        opToRmwOpPair.first->setOperand(0, opToRmwOpPair.second.modifiedValue);
        // change base address to InductionVar
        opToRmwOpPair.first->setOperand(1, spdBufferRMW[opToRmwOpPair.first]);
        // change offset to InductionVar
        opToRmwOpPair.first->setOperand(2, spdIndex);
      }
    }
    if (hasScatterPath) {
      rewriter.setInsertionPoint(forOp);
      /// Step1: create spd buffer for store values of scatter path
      for (auto &opToValPair : forOpToScatterPath[forOp].storeOpVals) {
        Value spdBuffer = getSpdBuffer(opToValPair.second.getType(), rewriter,
                                       loopLength, forOp->getLoc());

        // replace the original StoreOp to store to spd buffer
        // assume last instruction is a StoreOp
        spdBufferScatter[opToValPair.first] = spdBuffer;
      }

      /// Step2: Generate packed store op and sink outside after the loop
      rewriter.setInsertionPointAfter(forOp);
      generatePackedMemAccStoreOp(forOp.getLoc(), rewriter, spdBufferScatter,
                                  forOp, forOpToScatterPath[forOp],
                                  forOpToScatterPath[forOp].indirectDepth, loopUpperBound, loopLowerBound);

      /// Step3: Change original store operations to store to spd buffer with
      /// induction var
      for (auto &opToValPair : forOpToScatterPath[forOp].storeOpVals) {
        // change base address to InductionVar
        opToValPair.first->setOperand(1, spdBufferScatter[opToValPair.first]);
        // change offset to InductionVar
        opToValPair.first->setOperand(2, spdIndex);
      }
    } // if

    forOpDone[forOp] = true;
    return success();
  }
};

void MemAccHoistLoadsPass::runOnOperation() {
  mlir::MLIRContext *context = getOperation()->getContext();
  mlir::RewritePatternSet patterns(context);

  getForToIndirectAccess(getOperation());
  GreedyRewriteConfig config;
  patterns.insert<PackGenericLoadOpOutsideLoop>(context);
  patterns.insert<LoadOpConversionPattern<memref::LoadOp>>(context);
  patterns.insert<LoadOpConversionPattern<affine::AffineLoadOp>>(context);
  patterns.insert<StoreOpConversionPattern<memref::StoreOp>>(context);
  patterns.insert<StoreOpConversionPattern<affine::AffineStoreOp>>(context);
  patterns.add<ConvertArithToMemAccPattern<arith::MulIOp, MemAcc::MulIOp>>(
      context);
  patterns.add<ConvertArithToMemAccPattern<arith::AddIOp, MemAcc::AddIOp>>(
      context);
  patterns.add<ConvertArithToMemAccPattern<arith::SubIOp, MemAcc::SubIOp>>(
      context);
  patterns.add<ConvertArithIndexCastToMemAccIndexCastPattern>(context);

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