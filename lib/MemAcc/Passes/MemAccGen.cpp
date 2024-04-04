#include "PassDetails.h"
#include "MemAcc/Dialect.h"
#include "MemAcc/Passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"



// Use LLVM's data structures for convenience and performance
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "memory-access-generation"

using namespace mlir;
using namespace mlir::MemAcc;
using namespace mlir::arith;
using namespace mlir::affine;
// Define the data structures at the beginning of your pass

namespace {
struct MemAccGenPass : public MemAccGenBase<MemAccGenPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace {

llvm::SmallPtrSet<Operation *, 16> deepestLoads;
llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 16>>
    loadOpToIndirectUses;
llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 16>>
    loadOpToIndirectChain;

static void postProcessDeepestLoads() {
  for (auto o : deepestLoads) {
    for (auto i : loadOpToIndirectUses[o]) {
      if (deepestLoads.count(i) > 0) {
        deepestLoads.erase(i);
      }
    }
  }
}

// These unary op legalizations are identical for floating-point
// or quantized types
template <typename SrcOpType, typename DestOpType>
class ConvertArithToMemAccPattern : public OpRewritePattern<SrcOpType> {
public:
  using OpRewritePattern<SrcOpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOpType op,
                                PatternRewriter &rewriter) const override {
    if (op->template getParentOfType<MemAcc::GenericLoadOp>()) {
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
    if (op->template getParentOfType<MemAcc::GenericLoadOp>()) {
      rewriter.replaceOpWithNewOp<MemAcc::IndexCastOp>(
          op, op.getResult().getType(), op.getOperand());
      return success();
    }
    return failure();
  }
};

template <typename StoreOpType>
struct StoreOpConversionPattern : public OpRewritePattern<StoreOpType> {
  using OpRewritePattern<StoreOpType>::OpRewritePattern;

  void rewriteStoreOp(StoreOpType storeOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<MemAcc::StoreOp>(
        storeOp, storeOp.getValueToStore(), storeOp.getMemRef(),
        storeOp.getIndices());
  }

  LogicalResult matchAndRewrite(StoreOpType storeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the storeOp is contained within an affine.for operation
    if (!storeOp->template getParentOfType<AffineForOp>()) {
      return failure();
    }

    rewriteStoreOp(storeOp, rewriter);
    return success();
  }
};

template <typename LoadOpType>
struct LoadOpConversionPattern : public OpRewritePattern<LoadOpType> {
  using OpRewritePattern<LoadOpType>::OpRewritePattern;

  void rewriteLoadOp(LoadOpType loadOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<MemAcc::LoadOp>(loadOp, loadOp.getMemRef(),
                                                loadOp.getIndices());
  }

  SmallVector<Type, 4> getGenericLoadOpResultTypes(Operation *loadOp) const {
    SmallVector<Type, 4> resultTypes;
    auto &indirectLoadUseChain = loadOpToIndirectChain[loadOp];
    loadOpToIndirectUses[loadOp].insert(loadOp);
    for (int i = indirectLoadUseChain.size() - 1; i >= 0; i--) {
      auto I = indirectLoadUseChain[i];
      PRINT(*I);
      for (auto U : I->getUsers()) {
        PRINT("User: " << *U);
        if (loadOpToIndirectUses[loadOp].count(U) == 0) {
          PRINT("External User: " << *U);
          resultTypes.push_back(I->getResult(0).getType());
          break;
        }
      }
    }
    return resultTypes;
  }

  SmallVector<Value, 4> populateGenericLoadOp(
      llvm::SmallVector<Operation *, 16> &indirectLoadUseChain,
      PatternRewriter &rewriter, MemAcc::GenericLoadOp genericLoadOp) const {
    SmallVector<Value, 4> resultVals;
    auto &region = genericLoadOp.getBody();

    // Create a block inside the GenericLoadOp's region
    auto *block = rewriter.createBlock(&region);

    // Move the operations from the indirectLoadUseChain into the block
    for (int i = indirectLoadUseChain.size() - 1; i >= 0; i--) {
      auto clonedOp = rewriter.clone(*indirectLoadUseChain[i]);
      indirectLoadUseChain[i]->getResult(0).replaceAllUsesWith(
          clonedOp->getResult(0));
      rewriter.eraseOp(indirectLoadUseChain[i]);
    }

    // Update external users & generate return values
    int res_idx = 0;
    for (auto &I : *block) {
      bool hasExternalUses = false;
      SmallVector<Operation *, 4> users{I.getUsers().begin(),
                                        I.getUsers().end()};
      for (auto U : users) {
        // PRINT("In populateGenericLoadOp, User: for " << I << " is " << *U);
        if (block != U->getBlock()) {
          hasExternalUses = true;
          for (unsigned int operandIndex = 0;
               operandIndex < U->getNumOperands(); ++operandIndex) {
            if (U->getOperand(operandIndex) == I.getResult(0)) {
              // Update the operand with a new value
              U->setOperand(operandIndex, genericLoadOp->getResult(res_idx));
              hasExternalUses = true;
            }
          } // for
        }   // if
      } // for
      if (hasExternalUses) {
        res_idx++;
        resultVals.push_back(I.getResult(0));
      }
    } // for
    return resultVals;
  }

  LogicalResult matchAndRewrite(LoadOpType loadOp,
                                PatternRewriter &rewriter) const override {
    // Check if the loadOp is contained within an affine.for operation
    if (!loadOp->template getParentOfType<AffineForOp>()) {
      return failure();
    }
    if (loadOp->template getParentOfType<MemAcc::GenericLoadOp>()) {
      rewriteLoadOp(loadOp, rewriter);
      return success();
    }

    // only consider the deepest loads
    if (deepestLoads.count(loadOp) == 0) {
      return failure();
    }

    uint64_t indirectionLevel = 0;
    auto &indirectLoadUseChain = loadOpToIndirectChain[loadOp];
    // Calculate the indirection level based on the number of loads in the
    // indirectLoadUseChain
    for (Operation *op : indirectLoadUseChain) {
      if (isa<memref::LoadOp>(op) || isa<affine::AffineLoadOp>(op)) {
        indirectionLevel++;
      }
    }

    // If there is no indirection, return failure
    if (indirectionLevel < 1){
      return failure();
    }

    SmallVector<Value, 4> resultVals;

    // get result types of generic load op
    auto resultTypes = getGenericLoadOpResultTypes(loadOp);
    // Count the number of loads in the indirectLoadUseChain for indirection
    // level

    auto indirectionAttr = IntegerAttr::get(
        IntegerType::get(rewriter.getContext(), 64), indirectionLevel - 1);

    Location loc = loadOp.getLoc();

    // Start creating the GenericLoadOp
    auto genericLoadOp = rewriter.create<MemAcc::GenericLoadOp>(
        loc, resultTypes, indirectionAttr);

    // Populate the GenericLoadOp with the operations from the
    // indirectLoadUseChain & update use-def chain for external users
    resultVals =
        populateGenericLoadOp(indirectLoadUseChain, rewriter, genericLoadOp);

    rewriter.create<MemAcc::YieldOp>(loc, resultTypes, resultVals);

    return success();
  }
};

// Modified to populate the mapping
void markIndirectLoadUsers(Operation *op,
                           llvm::SmallPtrSetImpl<Operation *> &visited,
                           Operation *originalLoadOp) {

  if (!op || !visited.insert(op).second)
    return;

  if (isa<memref::LoadOp>(op) || isa<affine::AffineLoadOp>(op) ||
      isa<arith::ArithDialect>(op->getDialect())) {
    loadOpToIndirectUses[originalLoadOp].insert(op);
    loadOpToIndirectChain[originalLoadOp].push_back(op);
  } else {
    return;
  }

  for (auto operand : op->getOperands()) {
    markIndirectLoadUsers(operand.getDefiningOp(), visited, originalLoadOp);
  }
}

void analyzeLoadOps(Operation *op,
                    llvm::SmallPtrSet<Operation *, 16> &deepestLoads) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  op->walk([&](Operation *currentOp) {
    if (isa<memref::LoadOp>(currentOp) ||
        isa<affine::AffineLoadOp>(currentOp)) {
      visited.clear();
      loadOpToIndirectChain[currentOp].push_back(currentOp);
      // Check all users of the load operation to see if it indirectly
      // contributes to another load
      for (auto operand : currentOp->getOperands()) {
        markIndirectLoadUsers(operand.getDefiningOp(), visited, currentOp);
      }
      deepestLoads.insert(currentOp);
    }
  });
  postProcessDeepestLoads();
}

void MemAccGenPass::runOnOperation() {
  deepestLoads.clear();
  loadOpToIndirectUses.clear();
  loadOpToIndirectChain.clear();
  mlir::MLIRContext *context = getOperation()->getContext();

  analyzeLoadOps(getOperation(), deepestLoads);

  // context->loadDialect<mlir::MemAcc::MemAccDialect>();
  mlir::RewritePatternSet patterns(context);
  patterns.add<StoreOpConversionPattern<memref::StoreOp>>(context);
  patterns.add<StoreOpConversionPattern<affine::AffineStoreOp>>(context);
  patterns.add<LoadOpConversionPattern<memref::LoadOp>>(context);
  patterns.add<LoadOpConversionPattern<affine::AffineLoadOp>>(context);
  patterns.add<ConvertArithToMemAccPattern<arith::MulIOp, MemAcc::MulIOp>>(
      context);
  patterns.add<ConvertArithToMemAccPattern<arith::AddIOp, MemAcc::AddIOp>>(
      context);
  patterns.add<ConvertArithToMemAccPattern<arith::SubIOp, MemAcc::SubIOp>>(
      context);
  patterns.add<ConvertArithIndexCastToMemAccIndexCastPattern>(context);
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}
} // end anonymous namespace

namespace mlir {
namespace MemAcc {
std::unique_ptr<Pass> createMemAccGenPass() {
  return std::make_unique<MemAccGenPass>();
}
} // namespace MemAcc
} // namespace mlir