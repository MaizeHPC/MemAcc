export PATH=$PATH:/home/lukezhuz/MemAcc/build/bin/


function test_memacc_analysis() {
    echo "[LOG]: Working on $1"

    echo "[LOG]: Converting C/C++ to MLIR using Polygeist"
    cgeist $1 -raise-scf-to-affine -O0 -S -o $1.mlir -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/usr/include/linux/

    echo "[LOG]: Capturing indirect memory accesses and hoisting them out of loop (hardware-agnostic)"
    polygeist-opt $1.mlir --memory-access-generation  --canonicalize --mlir-disable-threading -o $1.mlir
}

for i in $(ls *.cpp); do
    test_memacc_analysis $i
done