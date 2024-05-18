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
#include "llvm/ADT/SmallVector.h"
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

// Function to check if opB depends on opA
static bool  dependsOn(Operation *opA, Operation *opB) {
    // Base case: If opB is directly using a result of opA
    for (auto operand : opB->getOperands()) {
        if (operand.getDefiningOp() == opA) {
            return true;
        }
    }

    // Recursive case: Check transitive dependencies
    for (auto result : opA->getResults()) {
        for (auto user : result.getUsers()) {
            if (dependsOn(user, opB)) {
                return true;
            }
        }
    }

    return false;
}

static bool operandIdxDependsOn(Operation *opA, Operation *opB, int operandIdx) {
    // Base case: If opB is directly using a result of opA
    if (opB->getOperand(operandIdx).getDefiningOp() == opA) {
        return true;
    }

    // Recursive case: Check transitive dependencies
    for (auto result : opA->getResults()) {
        for (auto user : result.getUsers()) {
            if (dependsOn(user, opB)) {
                return true;
            }
        }
    }

    return false;
}

template <typename DestType>
static bool typeOperandIdxDependsOn(Operation *opA, int operandIdx) {

    // Recursive case: Check transitive dependencies
    for (auto result : opA->getResults()) {
        for (auto user : result.getUsers()) {
            PRINT("Checking user: " << *user);
            if (isa<DestType>(user) && user->getOperand(operandIdx).getDefiningOp() == opA) {
                PRINT("Found user: " << *user);
                return true;
            }
        }
        for (auto user : result.getUsers()) {
            if (typeOperandIdxDependsOn<DestType>(user, operandIdx)) {
                return true;
            }
        }
    }
    return false;
}

// append the trace of the dependency to the trace vector
template <typename DestType>
static bool typeOperandIdxDependsOn(Operation *opA, int operandIdx, llvm::SmallVector<Operation*, 16>& trace) {

    // Recursive case: Check transitive dependencies
    for (auto result : opA->getResults()) {
        for (auto user : result.getUsers()) {
            PRINT("Checking user: " << *user);
            if (isa<DestType>(user) && user->getOperand(operandIdx).getDefiningOp() == opA) {
                PRINT("Found user: " << *user);
                return true;
            }
        }
        for (auto user : result.getUsers()) {
            trace.push_back(user);
            if (typeOperandIdxDependsOn<DestType>(user, operandIdx, trace)) {
                return true;
            }
            trace.pop_back();
        }
    }
    return false;
}

} // namespace polygeist
} // namespace mlir

#endif // DIALECT_POLYGEIST_TRANSFORMS_PASSDETAILS_H
