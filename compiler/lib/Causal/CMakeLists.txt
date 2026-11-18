add_mlir_dialect_library(CausalDialect
  CausalDialect.cpp
  CausalOps.cpp
  ColumnProjection.cpp
  FusePredictions.cpp
  LowerInterventionPredict.cpp
  LowerTreatmentEffect.cpp

  DEPENDS
  CausalDialectIncGen
  CausalOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
)

target_include_directories(CausalDialect
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}/include"
)
