llc --relocation-model=pic -filetype=obj test_memacc.ll -o test_memacc.o
cd ../MAA
bash make.sh
cd ../test
gcc -fPIE -pie test_memacc.o -L../MAA -lmaacompiler -o test_memacc
