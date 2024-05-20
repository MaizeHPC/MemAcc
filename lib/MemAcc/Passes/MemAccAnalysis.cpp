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

    /// Gather trace must end with a load op
    void DFS::GatherPath::verification() {
        if (indirectChain.empty() || !isa<memref::LoadOp>(indirectChain.back()) || !isa<affine::AffineLoadOp>(indirectChain.back())) {
            assert(false &&  "Gather trace must end with a load op\n");
        }
    }

    /// Scatter trace must end with a store op
    void DFS::ScatterPath::verification() {
        if (indirectChain.empty() || !isa<memref::StoreOp>(indirectChain.back()) || !isa<affine::AffineStoreOp>(indirectChain.back())) {
            assert(false &&  "Scatter trace must end with a store op\n");
        }
    }

    void DFS::print_results() {
        // print gather traces
        for (auto &gather : gatherPaths_) {
            PRINT("Gather trace for: " << *gather.first);
            PRINT("Indirect depth: " << gather.second.indirectDepth);
            PRINT("Indirect chain:");
            for (auto op : gather.second.indirectChain) {
                PRINT("  " << *op);
            }
            PRINT("External users:");
            for (size_t i = 0; i < gather.second.externUsers.users.size(); i++) {
                PRINT("  " << *gather.second.externUsers.users[i] << " at operand " << gather.second.externUsers.operandIdx[i]);
            }
        }
        // print scatter traces
        for (auto &scatter : scatterPaths_) {
            PRINT("Scatter trace for: " << *scatter.first);
            PRINT("Indirect chain:");
            for (auto op : scatter.second.indirectChain) {
                PRINT("  " << *op);
            }
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
                        for (unsigned int operandIndex = 0; operandIndex < user->getNumOperands(); operandIndex++) {
                            if (user->getOperand(operandIndex) == op->getResult(0)) {
                                externalUsers.operandIdx.push_back(operandIndex);
                            }
                        }
                    }
                    // record the chain only if it's not a streaming memacc
                    gatherPaths_[op] = GatherPath{
                        currIndChain_,
                        currIndMap,
                        externalUsers,
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
                    currIndMap
                };
            }
        }   
        // Base case4: if current op is an unsupported operation(i.e. a function call, if expr, ...)
        //             directly return
        else{
            return;
        }

        for (auto user : curr_val.getUsers()) {
            if (currIndMap.count(user) == 0) { // Prevent infinite recursion
                currIndChain_.push_back(user);
                currIndMap.insert(user);
                solve(curr_val, user, depth); // Update curr_val with user->getResult(0)
                currIndChain_.pop_back();
                currIndMap.erase(user);
            }
        }
    } // DFS::solve
}