polygeist-opt test_memacc.mlir --lower-affine -o test_memacc.mlir.scf
polygeist-opt test_memacc.mlir.scf --convert-polygeist-to-llvm -o test_memacc.mlir.llvm
polygeist-opt test_memacc.mlir.llvm --memory-access-to-llvm --mlir-disable-threading -o test_memacc.mlir.llvm
mlir-opt test_memacc.mlir.llvm --reconcile-unrealized-casts -o test_memacc.mlir.llvm
mlir-translate -mlir-to-llvmir  test_memacc.mlir.llvm -o test_memacc.ll
opt -passes=intrinsic-gen test_memacc.ll -S -o test_memacc.ll
llc --relocation-model=pic -filetype=obj test_memacc.ll -o test_memacc.o
gcc -fPIC -shared maa_lib.c -o libmaa.so
gcc -fPIE -pie test_memacc.o -L. -lmaa -o test_memacc