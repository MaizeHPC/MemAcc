polygeist-opt test_memacc.mlir --lower-affine -o test_memacc.mlir.scf
polygeist-opt test_memacc.mlir.scf --convert-polygeist-to-llvm -o test_memacc.mlir.llvm