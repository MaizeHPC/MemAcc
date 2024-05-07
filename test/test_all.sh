export PATH=$PATH:/home/lukezhuz/MemAcc/build/bin/

echo "[LOG]: converting test.cpp to test.mlir"
cgeist test.cpp -raise-scf-to-affine -O0 -S -o test.mlir -I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/11/include -I/usr/include/linux/ &> /dev/null

echo "[LOG]: generating memory access information"
bash test_memacc.sh &> /dev/null

echo "[LOG]: lowering to LLVM"
bash lower_to_llvm.sh &> /dev/null

echo "[LOG]: running test"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
./test_memacc