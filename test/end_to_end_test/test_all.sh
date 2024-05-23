export PATH=$PATH:/home/lukezhuz/MemAcc/build/bin/

function cleanup() {
    rm test_exe test_exe_ref tmp_compiler.log tmp_ref.log
    rm *.ll *.mlir *.mlir.llvm *.o
}

function abort() {
    echo "[ERROR]: $1"
    exit 1
}

function Usage() {
    echo "Usage: bash test_all.sh [DEBUG|RELEASE] [CLEAN|NOCLEAN]"
    echo "DEBUG: compile with debug information; RELEASE: compile without debug information"
    echo "CLEAN: clean up intermediate files; NOCLEAN: keep intermediate files"

    exit 1
}

function test_memacc_analysis() {
    echo "[LOG]: Working on $1 with debug $2"

    echo "[LOG]: Converting C/C++ to MLIR using Polygeist"
    cgeist $1 -raise-scf-to-affine -O0 -S -o $1.mlir -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/usr/include/linux/
    if [ $? -ne 0 ]; then
        abort "failed to convert C/C++ to MLIR"
    fi

    echo "[LOG]: Capturing indirect memory accesses and hoisting them out of loop (hardware-agnostic)"
    polygeist-opt $1.mlir --memory-access-hoist-loads  --canonicalize --mlir-disable-threading -o $1.mlir
    if [ $? -ne 0 ]; then
        abort "failed to capture indirect memory accesses"
    fi

    echo "[LOG]: Lowering memacc to llvm"
    polygeist-opt $1.mlir --memory-access-to-llvm --lower-affine --convert-polygeist-to-llvm -o $1.mlir.llvm --mlir-disable-threading
    mlir-opt $1.mlir.llvm --reconcile-unrealized-casts --canonicalize --cse -o $1.mlir.llvm
    mlir-translate -mlir-to-llvmir  $1.mlir.llvm -o $1.ll
    opt -passes=intrinsic-gen $1.ll -S -o $1.ll
    if [ $? -ne 0 ]; then
        abort "failed to lower memacc to llvm"
    fi
    # opt -O2  $1.ll -S -o $1.ll

    echo "[LOG]: Compiling to executable and link target library"
    llc --relocation-model=pic -filetype=obj $1.ll -o $1.o
    cd ../../MAA
    bash make.sh $2
    cd ../test/end_to_end_test
    gcc -fPIE -pie $1.o -L../../MAA -lmaacompiler -o test_exe
    if [ $? -ne 0 ]; then
        abort "failed to compile to executable"
    fi

    echo "[LOG]: running test"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../MAA
    ./test_exe 1> tmp_compiler.log
    gcc $1 -o test_exe_ref
    ./test_exe_ref > tmp_ref.log
    diff tmp_compiler.log tmp_ref.log
    if [ $? -ne 0 ]; then
        echo "[ERROR]: test $1 failed"
        echo "compiler output:"
        cat tmp_compiler.log
        echo "expected output:"
        cat tmp_ref.log
        exit 1
    fi
}

if [ $# -ne 2 ]; then
    Usage
fi

for i in $(ls *.cpp); do
    test_memacc_analysis $i $1
done

echo "[LOG]: All tests passed"
if [ $2 == "CLEAN" ]; then
    echo "Cleaning up intermediate files"
    cleanup
fi