set(LLVM_TARGET_DEFINITIONS CausalDialect.td)
mlir_tablegen(CausalDialect.h.inc -gen-dialect-decls -dialect=ci)
mlir_tablegen(CausalDialect.cpp.inc -gen-dialect-defs -dialect=ci)
add_public_tablegen_target(CausalDialectIncGen)

set(LLVM_TARGET_DEFINITIONS CausalOps.td)
mlir_tablegen(CausalOps.h.inc -gen-op-decls)
mlir_tablegen(CausalOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CausalOpsIncGen)

add_dependencies(CausalOpsIncGen CausalDialectIncGen)
