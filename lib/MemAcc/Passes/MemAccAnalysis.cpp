#include "MemAcc/Passes/MemAccAnalysis.h"
#include "MemAcc/Dialect.h"
#include "MemAcc/Passes/Passes.h"
#include "PassDetails.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "analysis"

namespace mlir {

static inline bool isLoadOp(Operation *op) {
  return isa<memref::LoadOp>(op) || isa<affine::AffineLoadOp>(op) ||
         isa<MemAcc::LoadOp>(op);
}

static inline bool isStoreOp(Operation *op) {
  return isa<memref::StoreOp>(op) || isa<affine::AffineStoreOp>(op) ||
         isa<MemAcc::StoreOp>(op);
}

static inline bool isRMWOp(Operation *op) {
  return isa<memref::AtomicRMWOp>(op) || isa<MemAcc::RMWOp>(op);
}

static inline bool isLoadAndStoreMatch(Operation *load, Operation *store) {
  assert(isLoadOp(load) && "load op expected\n");
  assert(isStoreOp(store) && "store op expected\n");
  return load->getOperand(0) == store->getOperand(1) &&
         load->getOperand(1) == store->getOperand(2);
}

static inline std::optional<arith::AtomicRMWKind> getRMWKind(Operation *op) {
  std::optional<arith::AtomicRMWKind> maybeKind =
      TypeSwitch<Operation *, std::optional<arith::AtomicRMWKind>>(op)
          .Case([](arith::AddFOp) { return arith::AtomicRMWKind::addf; })
          .Case([](arith::MulFOp) { return arith::AtomicRMWKind::mulf; })
          .Case([](arith::AddIOp) { return arith::AtomicRMWKind::addi; })
          .Case([](arith::AndIOp) { return arith::AtomicRMWKind::andi; })
          .Case([](arith::OrIOp) { return arith::AtomicRMWKind::ori; })
          .Case([](arith::MulIOp) { return arith::AtomicRMWKind::muli; })
          .Case(
              [](arith::MinimumFOp) { return arith::AtomicRMWKind::minimumf; })
          .Case(
              [](arith::MaximumFOp) { return arith::AtomicRMWKind::maximumf; })
          .Case([](arith::MinSIOp) { return arith::AtomicRMWKind::mins; })
          .Case([](arith::MaxSIOp) { return arith::AtomicRMWKind::maxs; })
          .Case([](arith::MinUIOp) { return arith::AtomicRMWKind::minu; })
          .Case([](arith::MaxUIOp) { return arith::AtomicRMWKind::maxu; })
          .Default([](Operation *) -> std::optional<arith::AtomicRMWKind> {
            // TODO: AtomicRMW supports other kinds of reductions this is
            // currently not detecting, add those when the need arises.
            return std::nullopt;
          });
  return maybeKind;
}

// Gather trace must end with a load op
void DFS::GatherPath::verification() {
  if (indirectChain.empty() || !isLoadOp(std::get<0>(indirectChain.back()))) {
    assert(false && "Gather trace must end with a load op\n");
  }
}

void DFS::GatherPath::print() {
  PRINT("Gather Path:");
  PRINT("Indirect chain:");
  for (auto [op, condOp, condBranch] : indirectChain) {
    PRINT("Op  " << *op << " Condition Depends on " << *condOp << " " << condBranch);
  }
  PRINT("External users:");
  for (auto &opToUserPair : deepestLoadToExternUsers) {
    PRINT("  " << *opToUserPair.first << " is used by:");
    for (unsigned int i = 0; i < opToUserPair.second.users.size(); i++) {
      PRINT("    " << *opToUserPair.second.users[i] << " at operand "
                   << opToUserPair.second.operandIdx[i]);
    }
  }
}

void DFS::ScatterPath::print() {
  PRINT("Scatter Path:");
  PRINT("Indirect chain:");
  for (auto [op, condOp, condBranch] : indirectChain) {
    PRINT("Op  " << *op << " Condition Depends on " << *condOp << " " << condBranch);
  }
  PRINT("Store op value:");
  for (auto &opToValPair : storeOpVals) {
    PRINT("  " << *opToValPair.first << " has store val "
               << opToValPair.second);
  }
}

void DFS::filterGatherPath() {
  // Assuming scatter paths are merged
  llvm::SmallVector<Operation *, 16> gatherPathsToRemove;
  for (auto &gatherPathPair : gatherPaths_) {
    auto &gatherPath = gatherPathPair.second;
    auto loadOp = gatherPathPair.first;
    int numUsers = gatherPath.deepestLoadToExternUsers[loadOp].users.size();
    assert(numUsers > 0 && "Gather path must have at least one user\n");
    auto user = gatherPath.deepestLoadToExternUsers[loadOp].users[0];
    // if the user is an index cast op and has only one user, remove the gather
    // path
    if ((isa<arith::IndexCastOp>(user) || isa<MemAcc::IndexCastOp>(user)) &&
        numUsers == 1) {
      gatherPathsToRemove.push_back(loadOp);
    }
    // if the result is used as the modifiable value of a RMW op, remove the
    // gather path
    for (auto &rmwPathPair : resultRMWPath_.storeToRmwOp) {
      auto &rmwPath = rmwPathPair.second;
      if (rmwPath.originalLoadResult == loadOp->getResult(0) && numUsers == 1) {
        gatherPathsToRemove.push_back(loadOp);
      }
    }
  }
  for (int i = gatherPathsToRemove.size() - 1; i >= 0; i--) {
    gatherPaths_.erase(gatherPathsToRemove[i]);
  }
}

void DFS::RMWPath::print() {
  PRINT("RMW Path:");
  PRINT("Indirect chain:");
  for (auto [op, condOp, condBranch] : indirectChain) {
    PRINT("Op  " << *op << " Condition Depends on " << *condOp << " " << condBranch);
  }
  PRINT("RMW ops:");
  for (auto &rmwOpPair : storeToRmwOp) {
    auto &rmwOp = rmwOpPair.second;
    PRINT("  Store op: " << *rmwOpPair.first);
    PRINT("  RMW kind:" << rmwOp.opKind);
    // PRINT("    opKind: " << rmwOp.opKind);
    PRINT("    addressOffset: " << rmwOp.addressOffset);
    PRINT("    baseAddress: " << rmwOp.baseAddress);
    PRINT("    modifiedValue: " << rmwOp.modifiedValue);
  }
}

void DFS::RMWPath::merge(const RMWPath &other) {
  // update indirectChain/set
  for (auto op : other.indirectChain) {
    if (indirectUseSet.count(std::get<0>(op)) == 0) {
      indirectChain.push_back(op);
      indirectUseSet.insert(std::get<0>(op));
    }
  }

  // update RMW ops
  for (auto &storeToRmwOpPair : other.storeToRmwOp) {
    assert(storeToRmwOp.count(storeToRmwOpPair.first) == 0 &&
           "Store op already exists in RMW path\n");
    storeToRmwOp[storeToRmwOpPair.first] = storeToRmwOpPair.second;
  }
}

void DFS::genRMWPath() {
  llvm::SmallVector<Operation *> scatterPathsToRemove;
  // for each scatter path, try to generate RMW path
  for (auto &scatterPathPair : scatterPaths_) {
    auto &scatterPath = scatterPathPair.second;
    auto storeOp = scatterPathPair.first;
    auto storeVal = scatterPath.storeOpVals[storeOp];
    // check if storeVal is a binary arith op
    if (!isa<arith::ArithDialect>(storeVal.getDefiningOp()->getDialect()) &&
        !isa<MemAcc::MemAccDialect>(storeVal.getDefiningOp()->getDialect())) {
    //   PRINT("Store value is not an arith op\n");
      continue;
    }

    auto arithOp = storeVal.getDefiningOp();
    std::optional<arith::AtomicRMWKind> maybeKind = getRMWKind(arithOp);

    if (!maybeKind || arithOp->getNumOperands() != 2) {
    //   PRINT("Store value is not an add op\n" << *arithOp);
      continue;
    }
    // auto arithOpKind = arithOp.getKind();
    // check if one of the operands is a load op
    int loadOpIdx = -1;
    int nonMatchedLoadOpIdx = 1;
    for (int i = 0; i < 2; i++) {
      // if operand is load and base address is the same as the store op's
      // address
      if (isLoadOp(arithOp->getOperand(i).getDefiningOp()) &&
          isLoadAndStoreMatch(arithOp->getOperand(i).getDefiningOp(),
                              storeOp)) {
        loadOpIdx = i;
        nonMatchedLoadOpIdx = (i + 1) % 2;
        // PRINT("loadOpIdx: " << loadOpIdx << "Detected load "
        //                     << *arithOp->getOperand(i).getDefiningOp()
        //                     << " as one of the operands");

        break;
      }
    }

    if (loadOpIdx == -1) {
    //   PRINT("Arith op does not have a load op as one of its operands\n");
      continue;
    }

    // check if the load op has the same value as the address offset of the
    // store op use the address dependency map to find the address offset of the
    // store op
    auto storeAddressOffset = addressDependencyMap_[storeOp];
    auto loadOp = arithOp->getOperand(loadOpIdx).getDefiningOp();
    auto loadAddressOffset = addressDependencyMap_[loadOp];
    if (loadAddressOffset != storeAddressOffset ||
        loadOp->getOperand(0) != storeOp->getOperand(1)) {
    //   PRINT("storeAddressOffset: " << *storeAddressOffset);
    //   PRINT("loadAddressOffset: " << *loadAddressOffset);
    //   PRINT("store base address: " << storeOp->getOperand(1));
    //   PRINT("load base address: " << loadOp->getOperand(0));
    //   PRINT("Load op does not have the same value as the address offset of the "
    //         "store op\n");
      continue;
    }

    // generate RMW path
    RMWOp rmwOp{
        *maybeKind,
        storeOp->getOperand(2),                   // address offset
        storeOp->getOperand(1),                   // base address
        arithOp->getOperand(nonMatchedLoadOpIdx), // modified value
        loadOp->getResult(0)                      // original load op
    };
    resultRMWPath_.merge(
        RMWPath{scatterPath.indirectChain, scatterPath.indirectUseSet,
                llvm::DenseMap<Operation *, RMWOp>{{storeOp, rmwOp}}});
    resultRMWPath_.print();
    scatterPathsToRemove.push_back(storeOp);
  }

  for (int i = scatterPathsToRemove.size() - 1; i >= 0; i--) {
    scatterPaths_.erase(scatterPathsToRemove[i]);
  }

  for (auto &rmwPathPair : rmwPaths_) {
    resultRMWPath_.merge(rmwPathPair.second);
  }
}

void DFS::GatherPath::merge(const GatherPath &other) {
  // update indirectChain/set
  for (auto op : other.indirectChain) {
    if (indirectUseSet.count(std::get<0>(op)) == 0) {
      indirectChain.push_back(op);
      indirectUseSet.insert(std::get<0>(op));
    }
  }

  /// update external users
  // First merge the external users from other
  for (auto &opToUserPair : other.deepestLoadToExternUsers) {
    if (deepestLoadToExternUsers.count(opToUserPair.first) == 0) {
      deepestLoadToExternUsers[opToUserPair.first] = opToUserPair.second;
    }
  }
  // Second remove the external users that exist in indirectUseSet
  llvm::SmallVector<Operation *, 16> toRemove;
  for (auto &opToUserPair : deepestLoadToExternUsers) {
    for (auto &user : opToUserPair.second.users) {
      if (indirectUseSet.count(user) > 0) {
        for (unsigned int i = 0; i < opToUserPair.second.users.size(); i++) {
          if (opToUserPair.second.users[i] == user) {
            opToUserPair.second.users.erase(opToUserPair.second.users.begin() +
                                            i);
            opToUserPair.second.operandIdx.erase(
                opToUserPair.second.operandIdx.begin() + i);
          }
        }
      }
    }
    if (opToUserPair.second.users.empty()) {
      toRemove.push_back(opToUserPair.first);
    }
  }

  for (auto op : toRemove) {
    deepestLoadToExternUsers.erase(op);
  }

  // depth is max of two depths
  indirectDepth = std::max(indirectDepth, other.indirectDepth);
}

void DFS::ScatterPath::merge(const ScatterPath &other) {
  // update indirectChain/set
  for (auto op : other.indirectChain) {
    if (indirectUseSet.count(std::get<0>(op)) == 0) {
      indirectChain.push_back(op);
      indirectUseSet.insert(std::get<0>(op));
    }
  }
  for (auto &opToValPair : other.storeOpVals) {
    assert(storeOpVals.count(opToValPair.first) == 0 &&
           "Store op already exists in scatter path\n");
    storeOpVals[opToValPair.first] = opToValPair.second;
  }

  // update indirectDepth
  indirectDepth = std::max(indirectDepth, other.indirectDepth);
}

/// Scatter trace must end with a store op
void DFS::ScatterPath::verification() {
  if (indirectChain.empty() ||
      (!isa<memref::StoreOp>(std::get<0>(indirectChain.back())) &&
       !isa<affine::AffineStoreOp>(std::get<0>(indirectChain.back())))) {
    assert(false && "Scatter trace must end with a store op\n");
  }
}

void DFS::print_results() {
  // print gather traces
  for (auto &gather : gatherPaths_) {
    PRINT("Gather trace for: " << *gather.first);
    gather.second.verification();
    gather.second.print();
  }
  // print scatter traces
  for (auto &scatter : scatterPaths_) {
    PRINT("Scatter trace for: " << *scatter.first);
    scatter.second.verification();
    scatter.second.print();
  }
  // print RMW traces
  for (auto &rmw : rmwPaths_) {
    PRINT("RMW trace for: " << *rmw.first);
    rmw.second.print();
  }
  // print address dependency map
  PRINT("Address dependency map:");
  for (auto &addressDependency : addressDependencyMap_) {
    PRINT("Op: " << *addressDependency.first
                 << " depends on: " << *addressDependency.second);
  }
}

bool DFS::isAddressTransformationOp(Operation *op){
  return isLoadOp(op) || isStoreOp(op) || isRMWOp(op) || isa<arith::ArithDialect>(op->getDialect()) || isa<MemAcc::MemAccDialect>(op->getDialect());
}

/// Please refer to
//  ```
//  Ainsworth, Sam, and Timothy M. Jones. "Software prefetching for indirect
//  memory accesses." 2017 IEEE/ACM International Symposium on Code Generation
//  and Optimization (CGO). IEEE, 2017.
//  ```
void DFS::solve(Value curr_val, Operation *op, unsigned int depth,
                Operation *addressDependencyOp) {
  // Base case0: if current op has already been processed, return
  if (gatherPaths_.count(op) || scatterPaths_.count(op)) {
    return;
  }

  // Base case1: if current val is used by a load op's index, record the chain
  // for gather trace
  if (isLoadOp(op)) {
    if (op->getOperand(1) == curr_val) {
      if (depth >= 1) {
        GatherUseInfo externalUsers;
        for (auto user : op->getResult(0).getUsers()) {
          externalUsers.users.push_back(user);
          for (unsigned int operandIndex = 0;
               operandIndex < user->getNumOperands(); operandIndex++) {
            if (user->getOperand(operandIndex) == op->getResult(0)) {
              externalUsers.operandIdx.push_back(operandIndex);
            }
          }
        }
        // record the chain only if it's not a streaming memacc
        gatherPaths_[op] = GatherPath{
            currIndChain_, currIndMap_,
            llvm::DenseMap<Operation *, GatherUseInfo>{{op, externalUsers}},
            depth};
      }
      assert(addressDependencyMap_.count(op) == 0 &&
             "Load op already exists in address dependency map\n");
      addressDependencyMap_[op] = addressDependencyOp;
      addressDependencyOp = op;
      curr_val = op->getResult(0);
      depth++;
    }
  }
  // Base case2: if current val is used by a store op, record the chain for
  // scatter trace
  else if (isStoreOp(op)) {
    if (op->getOperand(2) == curr_val && depth >= 1) {
      scatterPaths_[op] = ScatterPath{
          currIndChain_, currIndMap_,
          llvm::DenseMap<Operation *, Value>{{op, op->getOperand(0)}}, depth};
      assert(addressDependencyMap_.count(op) == 0 &&
             "Store op already exists in address dependency map\n");
      addressDependencyMap_[op] = addressDependencyOp;
    }
    return;
  }
  // Base case3: if current op is rmw
  else if (isRMWOp(op)) {
    auto rmwOp = cast<MemAcc::RMWOp>(op);
    if (rmwOp.getOperand(2) == curr_val && depth >= 1) {
      RMWOp rmwOpStruct{
          rmwOp.getKind(),     // rmw kind
          rmwOp.getOperand(2), // address offset
          rmwOp.getOperand(1), // base address
          rmwOp.getOperand(0), // modified value
          curr_val             // original load op
      };
      rmwPaths_[op] =
          RMWPath{currIndChain_, currIndMap_,
                  llvm::DenseMap<Operation *, RMWOp>{{op, rmwOpStruct}}};
      assert(addressDependencyMap_.count(op) == 0 &&
             "RMW op already exists in address dependency map\n");
      addressDependencyMap_[op] = addressDependencyOp;
    }
    return;
  }

  // Base case3: if current val is used by an arith op, populate the current val
  // TODO: Now it only supports unary arith op
  //       When extending to binary arith op, need to add another path for the
  //       second operand
  else if (isa<arith::ArithDialect>(op->getDialect()) ||
           isa<MemAcc::MemAccDialect>(op->getDialect())) {
    // If no result, return
    if (op->getNumResults() == 0)
      return;
    curr_val = op->getResult(0);
    // if current op is not cast op, change addressDependencyOp to current op
    if (!isa<arith::IndexCastOp>(op) && !isa<MemAcc::IndexCastOp>(op)) {
      addressDependencyOp = op;
    }
  }
  // Base case4: if current op is an unsupported operation(i.e. a function call,
  // if expr, ...)
  //             directly return
  else {
    return;
  }

  for (auto user : curr_val.getUsers()) {
    if (currIndMap_.count(user) == 0) { // Prevent infinite recursion
      auto [condOp, condBranch] = getDependentCondition(user);
      currIndChain_.push_back(std::make_tuple(user, condOp, condBranch));
      currIndMap_.insert(user);
      solve(curr_val, user, depth,
            addressDependencyOp); // Update curr_val with user->getResult(0)
      currIndChain_.pop_back();
      currIndMap_.erase(user);
    }
  }
} // DFS::solve
} // namespace mlir