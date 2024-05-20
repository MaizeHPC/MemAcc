//===- PassDetails.h - polygeist pass class details ----------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different polygeist passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_MEMACC_TRANSFORMS_PASSDETAILS_H
#define DIALECT_MEMACC_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"
#include "MemAcc/Ops.h"
#include "MemAcc/Passes/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <iostream>

#define DEBUG
#ifdef DEBUG
#define DEBUG_TYPE "utils"
#define PRINT(x) llvm::errs() << "Pass[" << DEBUG_TYPE << "] " << x << "\n"
#else
#define PRINT(x)
#endif

namespace mlir {
class FunctionOpInterface;
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);
namespace MemAcc {

class PolygeistDialect;

#define GEN_PASS_CLASSES
#include "MemAcc/Passes/Passes.h.inc"

struct DFS{
    struct GatherTrace{
        llvm::SmallVector<Operation *, 16> indirectChain;
        llvm::SmallPtrSet<Operation *, 16> indirectUseSet;
        llvm::SmallVector<std::pair<Operation *, int>, 16> externalUsers;
        int indirectDepth;
    };
    struct ScatterTrace{
        llvm::SmallVector<Operation *, 16> indirectChain;
        llvm::SmallPtrSet<Operation *, 16> indirectUseSet;
    };
    struct RMWTrace{
        // TODO: Implement this!
    };
llvm::DenseMap<Operation *, GatherTrace> gatherTraces;
llvm::DenseMap<Operation *, ScatterTrace> scatterTraces;
llvm::SmallVector<Operation *, 16> curr_ind_chain;
llvm::SmallPtrSet<Operation *, 16> curr_ind_map;

    void print_results() {
        // print gather traces
        for (auto &gather : gatherTraces) {
            PRINT("Gather trace for: " << *gather.first);
            PRINT("Indirect depth: " << gather.second.indirectDepth);
            PRINT("Indirect chain:");
            for (auto op : gather.second.indirectChain) {
                PRINT("  " << *op);
            }
            PRINT("External users:");
            for (auto user : gather.second.externalUsers) {
                PRINT("  " << *user.first << " at operand " << user.second);
            }
        }
        // print scatter traces
        for (auto &scatter : scatterTraces) {
            PRINT("Scatter trace for: " << *scatter.first);
            PRINT("Indirect chain:");
            for (auto op : scatter.second.indirectChain) {
                PRINT("  " << *op);
            }
        }
    }

    void solve(Value curr_val, Operation *op, int depth = 0) {
        if (isa<memref::LoadOp>(op) || isa<affine::AffineLoadOp>(op)) {
            if (op->getOperand(1) == curr_val) {
                if (depth >= 1){
                    // record the chain only if it's not a streaming memacc
                    gatherTraces[op] = GatherTrace{
                        curr_ind_chain,
                        curr_ind_map,
                        {},
                        depth
                    };
                }
                curr_val = op->getResult(0);
                depth++;
            }
        }
        if (isa<arith::ArithDialect>(op->getDialect())){
             curr_val = op->getResult(0);
        }
        if (isa<memref::StoreOp>(op) || isa<affine::AffineStoreOp>(op)) {
            if (op->getOperand(2) == curr_val && depth >= 1) {
                scatterTraces[op] = ScatterTrace{
                    curr_ind_chain,
                    curr_ind_map
                };
            }
        }

        for (auto user : curr_val.getUsers()) {
            if (curr_ind_map.count(user) == 0) { // Prevent infinite recursion
                curr_ind_chain.push_back(user);
                curr_ind_map.insert(user);
                solve(curr_val, user, depth); // Update curr_val with user->getResult(0)
                curr_ind_chain.pop_back();
                curr_ind_map.erase(user);
            }
        }
    }
};

} // namespace polygeist
} // namespace mlir

#endif // DIALECT_POLYGEIST_TRANSFORMS_PASSDETAILS_H
