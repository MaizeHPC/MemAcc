#map = affine_map<() -> (0)>
#map1 = affine_map<()[s0] -> (s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @_Z6gatherPdS_Pii(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xi32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg3 : i32 to index
    %alloc_spd = memacc.alloc_spd(%0) : memref<?xf64>
    "memacc.packed_generic_load"(%alloc_spd, %0) <{indirectionLevel = 1 : i64, lowerBoundMap = #map, operandSegmentSizes = array<i32: 1, 0, 1, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg4: index):
      %1 = memacc.load %arg2[%arg4] : memref<?xi32>
      %2 = memacc.index_cast %1 : i32 to index
      %3 = memacc.load %arg1[%2] : memref<?xf64>
      %4 = memacc.yield %3 : (f64) -> f64
    }) : (memref<?xf64>, index) -> ()
    affine.for %arg4 = 0 to %0 {
      %1 = memref.load %alloc_spd[%arg4] : memref<?xf64>
      affine.store %1, %arg0[%arg4] : memref<?xf64>
    }
    return
  }
}

