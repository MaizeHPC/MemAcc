set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name memacc)
add_public_tablegen_target(MLIRMemAccPassIncGen)

add_mlir_doc(Passes MemAccPasses ./ -gen-pass-doc)
