#ifndef MEMACC_PASSES
#define MEMACC_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def MemAccGen : Pass<"memory-access-generation"> {
  let summary = "convert memory accesses from affine.for to memacc";
  let constructor = "mlir::MemAcc::createMemAccGenPass()";
  let dependentDialects = [
    "MemAcc::MemAccDialect",
  ];
}

def MemAccHoistLoads : Pass<"memory-access-hoist-loads"> {
  let summary = "hoist indirect loads out of loops into scatter or gather operations";
  let constructor = "mlir::MemAcc::createMemAccHoistLoadsPass()";
  let dependentDialects = [
    "MemAcc::MemAccDialect",
  ];
}

#endif // MEMACC_PASSES
