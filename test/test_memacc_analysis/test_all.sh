export PATH=$PATH:/home/lukezhuz/MemAcc/build/bin/


function test_memacc_analysis() {
    echo "[LOG]: Working on $1"

    echo "[LOG]: Converting C/C++ to MLIR using Polygeist"
    cgeist $1 -raise-scf-to-affine -O0 -S -o $1.mlir -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/usr/include/linux/

    echo "[LOG]: Capturing indirect memory accesses and hoisting them out of loop (hardware-agnostic)"
    polygeist-opt $1.mlir --memory-access-hoist-loads  --canonicalize --mlir-disable-threading -o $1.mlir

    echo "[LOG]: Lowering memacc to llvm"
    polygeist-opt $1.mlir --memory-access-to-llvm --lower-affine --convert-polygeist-to-llvm -o $1.mlir.llvm --mlir-disable-threading
    mlir-opt $1.mlir.llvm --reconcile-unrealized-casts --canonicalize --cse -o $1.mlir.llvm
    mlir-translate -mlir-to-llvmir  $1.mlir.llvm -o $1.ll
    opt -passes=intrinsic-gen $1.ll -S -o $1.ll
    # opt -O2  $1.ll -S -o $1.ll

    echo "[LOG]: Compiling to executable and link target library"
    llc --relocation-model=pic -filetype=obj $1.ll -o $1.o
    cd ../../MAA
    bash make.sh
    cd ../test/test_memacc_analysis

    gcc -fPIE -pie $1.o -L../../MAA -lmaacompiler -o test_exe
    # rm $1.o $1.ll $1.mlir $1.mlir.llvm

    echo "[LOG]: running test"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../MAA
    ./test_exe
}

# for i in $(ls *.cpp); do
#     test_memacc_analysis $i
# done

test_memacc_analysis complex.cpp