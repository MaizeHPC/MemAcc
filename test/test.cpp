#include <stdio.h>

void gather(double *a, double *b, int *idx, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = b[idx[i]];
  }
}

int main(){
  double a[4], b[4] = {1.0, 2.0, 3.0, 4.0};
  int idx[4] = {3, 2, 1, 0};
  gather(a, b, idx, 4);
  for (int i = 0; i < 4; i++) {
    printf("%f\n", a[i]);
  }
  return 0;
}

// IR
// #map = affine_map<() -> (0)>
// #map1 = affine_map<()[s0] -> (s0)>
// module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
//   func.func @_Z6gatherPdS_Pii(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xi32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
//     %0 = arith.index_cast %arg3 : i32 to index
//     %alloc_spd = memacc.alloc_spd(%0) : memref<?xf64>
//     "memacc.packed_generic_load"(%alloc_spd, %0) <{indirectionLevel = 1 : i64, lowerBoundMap = #map, operandSegmentSizes = array<i32: 1, 0, 1, 0>, step = 1 : index, upperBoundMap = #map1}> ({
//     ^bb0(%arg4: index):
//       %1 = memacc.load %arg2[%arg4] : memref<?xi32>
//       %2 = memacc.index_cast %1 : i32 to index
//       %3 = memacc.load %arg1[%2] : memref<?xf64>
//       %4 = memacc.yield %3 : (f64) -> f64
//     }) : (memref<?xf64>, index) -> ()
//     affine.for %arg4 = 0 to %0 {
//       %1 = memref.load %alloc_spd[%arg4] : memref<?xf64>
//       affine.store %1, %arg0[%arg4] : memref<?xf64>
//     }
//     return
//   }
// }

// LLVM
// module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
//   llvm.func @_Z6gatherPdS_Pii(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) {
//     %0 = llvm.mlir.constant(0 : index) : i64
//     %1 = llvm.mlir.constant(1 : index) : i64
//     %2 = llvm.sext %arg3 : i32 to i64
//     %3 = "llvm.intr.spdalloc"(%2) : (i64) -> !llvm.ptr
//     %4 = llvm.mlir.constant(0 : i64) : i64
//      
//     "llvm.intr.maa.setloopbound"(%4 INT_MAX, %2) : (i64, i64) -> ()   32B binary 
//      ... configre step
//     ... other configs
//     llvm.br ^bb1(%0 : i64)
//   ^bb1(%5: i64):  // 2 preds: ^bb0, ^bb2
//     %6 = llvm.icmp "slt" %5, %2 : i64
//     llvm.cond_br %6, ^bb2, ^bb3
//   ^bb2:  // pred: ^bb1
//     %7 = llvm.getelementptr %3[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f64
//     %8 = llvm.load %7 : !llvm.ptr -> f64
//     %9 = llvm.getelementptr %arg0[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f64
//     llvm.store %8, %9 : f64, !llvm.ptr
//     %10 = llvm.add %5, %1  : i64
//     llvm.br ^bb1(%10 : i64)
//   ^bb3:  // pred: ^bb1
//     llvm.return
//   }
// }
