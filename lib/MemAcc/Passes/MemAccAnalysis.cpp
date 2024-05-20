#include "PassDetails.h"
#include "MemAcc/Passes/MemAccAnalysis.h"
#include "MemAcc/Dialect.h"
#include "MemAcc/Passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "analysis"

namespace mlir {

    // Gather trace must end with a load op
    void DFS::GatherPath::verification() {
        if (indirectChain.empty() || (!isa<memref::LoadOp>(indirectChain.back()) && !isa<affine::AffineLoadOp>(indirectChain.back()))) {
            assert(false &&  "Gather trace must end with a load op\n");
        }
    }

    void DFS::GatherPath::print(){
        PRINT("Indirect chain:");
        for (auto op : indirectChain) {
            PRINT("  " << *op);
        }
        PRINT("External users:");
        for (auto& opToUserPair: externUsers) {
            PRINT("  " << *opToUserPair.first << " is used by:");
            for (unsigned int i = 0; i < opToUserPair.second.users.size(); i++) {
                PRINT("    " << *opToUserPair.second.users[i] << " at operand " << opToUserPair.second.operandIdx[i]);
            }
        }
    }

    void DFS::ScatterPath::print(){
        PRINT("Indirect chain:");
        for (auto op : indirectChain) {
            PRINT("  " << *op);
        }
    }


    void DFS::GatherPath::merge(const GatherPath& other){
        // update indirectChain/set
        for (auto op : other.indirectChain){
            if (indirectUseSet.count(op) == 0){
                indirectChain.push_back(op);
                indirectUseSet.insert(op);
            }
        }

        /// update external users
        // First merge the external users from other
        for (auto& opToUserPair: other.externUsers){
            if (externUsers.count(opToUserPair.first) == 0){
                externUsers[opToUserPair.first] = opToUserPair.second;
            }
        }
        // Second remove the external users that exist in indirectUseSet
        llvm::SmallVector<Operation *, 16> toRemove;
        for (auto& opToUserPair: externUsers){
            for (auto& user: opToUserPair.second.users){
                if (indirectUseSet.count(user) > 0){
                    for (unsigned int i = 0; i < opToUserPair.second.users.size(); i++){
                        if (opToUserPair.second.users[i] == user){
                            opToUserPair.second.users.erase(opToUserPair.second.users.begin() + i);
                            opToUserPair.second.operandIdx.erase(opToUserPair.second.operandIdx.begin() + i);
                        }
                    }
                }
            }
            if (opToUserPair.second.users.empty()){
                toRemove.push_back(opToUserPair.first);
            }
        }

        for (auto op: toRemove){
            externUsers.erase(op);
        }
    }

    void DFS::ScatterPath::merge(const ScatterPath& other){
        // update indirectChain/set
        for (auto op : other.indirectChain){
            if (indirectUseSet.count(op) == 0){
                indirectChain.push_back(op);
                indirectUseSet.insert(op);
            }
        }
    }

    /// Scatter trace must end with a store op
    void DFS::ScatterPath::verification() {
        if (indirectChain.empty() || (!isa<memref::StoreOp>(indirectChain.back()) && !isa<affine::AffineStoreOp>(indirectChain.back()))) {
            assert(false &&  "Scatter trace must end with a store op\n");
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
    }

    void DFS::solve(Value curr_val, Operation *op, int depth) {
        // Base case0: if current op has already been processed, return
        if (gatherPaths_.count(op) || scatterPaths_.count(op)) {
            return;
        }

        // Base case1: if current val is used by a load op's index, record the chain for gather trace
        if (isa<memref::LoadOp>(op) || isa<affine::AffineLoadOp>(op)) {
            if (op->getOperand(1) == curr_val) {
                if (depth >= 1){
                    GatherPathOut externalUsers;
                    for (auto user : op->getResult(0).getUsers()) {
                        externalUsers.users.push_back(user);
                        externalUsers.userSet.insert(user);
                        for (unsigned int operandIndex = 0; operandIndex < user->getNumOperands(); operandIndex++) {
                            if (user->getOperand(operandIndex) == op->getResult(0)) {
                                externalUsers.operandIdx.push_back(operandIndex);
                            }
                        }
                    }
                    // record the chain only if it's not a streaming memacc
                    gatherPaths_[op] = GatherPath{
                        currIndChain_,
                        currIndMap_,
                        llvm::DenseMap<Operation *, GatherPathOut>{{op, externalUsers}},
                        depth
                    };
                }
                curr_val = op->getResult(0);
                depth++;
            }
        }
        // Base case2: if current val is used by an arith op, populate the current val
        // TODO: Now it only supports unary arith op
        //       When extending to binary arith op, need to add another path for the second operand
        else if (isa<arith::ArithDialect>(op->getDialect())){
             curr_val = op->getResult(0);
        }
        // Base case3: if current val is used by a store op, record the chain for scatter trace
        else if (isa<memref::StoreOp>(op) || isa<affine::AffineStoreOp>(op)) {
            if (op->getOperand(2) == curr_val && depth >= 1) {
                scatterPaths_[op] = ScatterPath{
                    currIndChain_,
                    currIndMap_
                };
            }
            return;
        }   
        // Base case4: if current op is an unsupported operation(i.e. a function call, if expr, ...)
        //             directly return
        else{
            return;
        }

        for (auto user : curr_val.getUsers()) {
            if (currIndMap_.count(user) == 0) { // Prevent infinite recursion
                currIndChain_.push_back(user);
                currIndMap_.insert(user);
                solve(curr_val, user, depth); // Update curr_val with user->getResult(0)
                currIndChain_.pop_back();
                currIndMap_.erase(user);
            }
        }
    } // DFS::solve
}