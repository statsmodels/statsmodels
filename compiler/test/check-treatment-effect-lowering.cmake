execute_process(
  COMMAND "${CI_OPT}" --ci-lower-treatment-effect "${INPUT_FILE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)

if(NOT result EQUAL 0)
  message(FATAL_ERROR "ci-opt failed:\n${error}")
endif()

string(FIND "${output}" "\"ci.treatment_effect\"" unlowered_operation_position)
if(NOT unlowered_operation_position EQUAL -1)
  message(FATAL_ERROR "ci.treatment_effect was not lowered:\n${output}")
endif()

string(FIND "${output}" "linalg.generic" generic_position)
if(generic_position EQUAL -1)
  message(FATAL_ERROR "expected lowered IR to contain a linalg.generic op:\n${output}")
endif()
