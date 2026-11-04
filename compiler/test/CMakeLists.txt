add_test(
  NAME ci-column-projection
  COMMAND
    ${CMAKE_COMMAND}
    -DCI_OPT=$<TARGET_FILE:ci-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/column-projection.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check-column-projection.cmake
)

add_test(
  NAME ci-treatment-effect-lowering
  COMMAND
    ${CMAKE_COMMAND}
    -DCI_OPT=$<TARGET_FILE:ci-opt>
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/treatment-effect-lowering.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check-treatment-effect-lowering.cmake
)
