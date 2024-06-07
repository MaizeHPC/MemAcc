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
#ifndef DIALECT_MEMACC_UTILS_H
#define DIALECT_MEMACC_UTILS_H

#include "mlir/Pass/Pass.h"
#include "MemAcc/Ops.h"
#include "MemAcc/Passes/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
    namespace MemAcc {
        /// Given a range of values, emit the code that reduces them with "min" or "max"
/// depending on the provided comparison predicate, sgt for max and slt for min.
///
/// Multiple values are scanned in a linear sequence.  This creates a data
/// dependences that wouldn't exist in a tree reduction, but is easier to
/// recognize as a reduction by the subsequent passes.
Value buildMinMaxReductionSeq(Location loc,
                                     arith::CmpIPredicate predicate,
                                     ValueRange values, OpBuilder &builder);

/// Emit instructions that correspond to computing the maximum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
Value getAffineMapMax(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands);

/// Emit instructions that correspond to computing the minimum value among the
/// values of a (potentially) multi-output affine map applied to `operands`.
Value getAffineMapMin(OpBuilder &builder, Location loc, AffineMap map,
                               ValueRange operands);
    }
}

#endif