polygeist-opt test_memacc.mlir --lower-affine -o test_memacc.mlir.scf
polygeist-opt test_memacc.mlir.scf --convert-polygeist-to-llvm -o test_memacc.mlir.llvm
polygeist-opt test_memacc.mlir.llvm --memory-access-to-llvm --mlir-disable-threading -o test_memacc.mlir.llvm
mlir-opt test_memacc.mlir.llvm --reconcile-unrealized-casts -o test_memacc.mlir.llvm