export PATH=$PATH:/home/lukezhuz/MemAcc/build/bin/

echo "[LOG]: Converting C/C++ to MLIR using Polygeist"
cgeist test.cpp -raise-scf-to-affine -O0 -S -o test.mlir -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/usr/include/linux/ &> /dev/null

echo "[LOG]: Capturing indirect memory accesses and hoisting them out of loop (hardware-agnostic)"
bash test_memacc.sh &> /dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR]: failed to generate memory access information"
    exit 1
fi

echo "[LOG]: Lowering to generic memory access to target API call (hardware-specific)"
bash lower_to_llvm.sh &> /dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR]: failed to lower to LLVM"
    exit 1
fi

echo "[LOG]: Compiling to executable and link target library"
bash code_gen.sh &> /dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR]: failed to compile to executable"
    exit 1
fi

echo "[LOG]: running test"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../MAA
./test_memacc