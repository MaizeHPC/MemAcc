## Indirect Memory Access Accelerator E2E Compiler(MemAcc)
This is the code repo for the compiler that can identify indirect memory access patterns in legacy C code and transform them into packed gather/scatter operations by moving them outside from loop. It builds on top of Polygeist, which is a compiler that can raise C code to MLIR::Affine. 

In general, we made the following contribution:
* We implemented a new MLIR Dialect called `MemAcc`(memory access) to represent arbitrary indirect memory access patterns. [Code Link](include/MemAcc/MemAccOps.td)
* We implemented an MLIR analysis pass that can identify indirect memory accesses(gather, scatter) in `affine.for` operations. [Code Link](lib/MemAcc/Passes/MemAccAnalysis.cpp)
* We implemented an MLIR transformation pass that can hoist identified indirect memory access outside of `affine.for` into `memacc.packed_generic*`. [Code Link](lib/MemAcc/Passes/MemAccHoistLoads.cpp)
* We implemented an MLIR pass that can lower the hoisted packed memory access patterns to target-aware llvm intrinsics. [Code Link](lib/MemAcc/Passes/MemAccToLLVM.cpp)(
Note the llvm intrinsics we added can be found here [Code Link](https://github.com/MaizeHPC/llvm-project/blob/182692a6133d3048b4fb24de98093d39c27e7d90/llvm/include/llvm/IR/Intrinsics.td#L2545-L2569)
mlir::llvm intrinsics we added can be found here [Code Link](https://github.com/MaizeHPC/llvm-project/blob/182692a6133d3048b4fb24de98093d39c27e7d90/mlir/include/mlir/Dialect/LLVMIR/LLVMIntrinsicOps.td#L1266-L1325))
* We implemented an LLVM pass that can transform the llvm intrinsics to external function calls. [Code Link](https://github.com/MaizeHPC/llvm-project/blob/d4db9e67ab825de35460895ba7a18ea6e8130e57/llvm/lib/Transforms/Utils/IntrinsicGen.cpp)
* We implemented a functional simulator that can simulate the behavior of microarchitecture we are designing [Code Link](https://github.com/MaizeHPC/MAA)

## Build the compiler and run our demo
You have to follow the below steps to build our compiler and run our demo
1. Clone the repo and dependent third-party repo
```sh
git clone git@github.com:MaizeHPC/MemAcc.git
git submodule update --init --recursive
```
2. Build the project (if Ninja is not available, change that to "Unix Makefiles")
```sh
mkdir build 
cd build
cmake -G "Unix Makefiles"  ../llvm-project/llvm   -DLLVM_ENABLE_PROJECTS="clang;mlir"   -DLLVM_EXTERNAL_PROJECTS="polygeist"   -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=..   -DLLVM_TARGETS_TO_BUILD="host"   -DLLVM_ENABLE_ASSERTIONS=ON   -DCMAKE_BUILD_TYPE=Release
```
3. Run the end-to-end gather kernel demo(in each step, please note that the IR files will be generated, press enter to continue)
```sh
cd test/demo_may_8
bash test_all.sh
```

## Future work checklist
- [] Implement the scatter kernels
- [] Our indirect memory access pass should support arbitrary loop transformation
- [] Support nested loop
- [] Support conditions inside of loop
- [] Support other backends




## The following are instructions for building Polygeist
## Build instructions

### Requirements 
- Working C and C++ toolchains(compiler, linker)
- cmake
- make or ninja

### 1. Clone Polygeist
```sh
git clone --recursive https://github.com/llvm/Polygeist
cd Polygeist
```

### 2. Install LLVM, MLIR, Clang, and Polygeist

#### Option 1: Using pre-built LLVM, MLIR, and Clang

Polygeist can be built by providing paths to a pre-built MLIR and Clang toolchain.

1. Build LLVM, MLIR, and Clang:
```sh
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```

To enable compilation to cuda add `-DMLIR_ENABLE_CUDA_RUNNER=1` and remove `-DLLVM_TARGETS_TO_BUILD="host"` from the cmake arguments. (You may need to specify `CUDACXX`, `CUDA_PATH`, and/or `-DCMAKE_CUDA_COMPILER`)

To enable the ROCM backend add `-DMLIR_ENABLE_ROCM_RUNNER=1` and remove `-DLLVM_TARGETS_TO_BUILD="host"` from the cmake arguments. (You may need to specify `-DHIP_CLANG_INCLUDE_PATH`, and/or `ROCM_PATH`)

For faster compilation we recommend using `-DLLVM_USE_LINKER=lld`.

2. Build Polygeist:
```sh
mkdir build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

For faster compilation we recommend using `-DPOLYGEIST_USE_LINKER=lld`.

    1. GPU backends

To enable the CUDA backend add `-DPOLYGEIST_ENABLE_CUDA=1`

To enable the ROCM backend add `-DPOLYGEIST_ENABLE_ROCM=1`

    2. Polymer

To enable polymer, add `-DPOLYGEIST_ENABLE_POLYMER=1`

This will cause the cmake invokation to pull and build the dependencies for polymer. To specify a custom directory for the dependencies, specify `-DPOLYMER_DEP_DIR=<absolute-dir>`. The dependencies will be build using the `tools/polymer/build_polymer_deps.sh`.

To run the polymer tests, use `ninja check-polymer`.



#### Option 2: Using unified LLVM, MLIR, Clang, and Polygeist build

Polygeist can also be built as an external LLVM project using [LLVM_EXTERNAL_PROJECTS](https://llvm.org/docs/CMake.html#llvm-related-variables).

1. Build LLVM, MLIR, Clang, and Polygeist:
```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

`ninja check-polygeist-opt` runs the tests in `Polygeist/test/polygeist-opt`
`ninja check-cgeist` runs the tests in `Polygeist/tools/cgeist/Test`

## Citing Polygeist

If you use Polygeist, please consider citing the relevant publications:

``` bibtex
@inproceedings{polygeistPACT,
  title = {Polygeist: Raising C to Polyhedral MLIR},
  author = {Moses, William S. and Chelini, Lorenzo and Zhao, Ruizhe and Zinenko, Oleksandr},
  booktitle = {Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
  numpages = {12},
  location = {Virtual Event},
  series = {PACT '21},
  publisher = {Association for Computing Machinery},
  year = {2021},
  address = {New York, NY, USA},
  keywords = {Polygeist, MLIR, Polyhedral, LLVM, Compiler, C++, Pluto, Polly, OpenScop, Parallel, OpenMP, Affine, Raising, Transformation, Splitting, Automatic-Parallelization, Reduction, Polybench},
}
@inproceedings{10.1145/3572848.3577475,
  author = {Moses, William S. and Ivanov, Ivan R. and Domke, Jens and Endo, Toshio and Doerfert, Johannes and Zinenko, Oleksandr},
  title = {High-Performance GPU-to-CPU Transpilation and Optimization via High-Level Parallel Constructs},
  year = {2023},
  isbn = {9798400700156},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3572848.3577475},
  doi = {10.1145/3572848.3577475},
  booktitle = {Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages = {119–134},
  numpages = {16},
  keywords = {MLIR, polygeist, CUDA, barrier synchronization},
  location = {Montreal, QC, Canada},
  series = {PPoPP '23}
}
```
