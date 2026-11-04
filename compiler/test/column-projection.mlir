module {
  func.func @effect(%coefficients: tensor<3xf64>) -> tensor<?xf64> {
    %data = "ci.load_dataset"() {columns = ["z", "x"], path = "users.csv"} : () -> tensor<?x?xf64>
    %effects = "ci.treatment_effect"(%data, %coefficients) {
      control_value = 0.000000e+00 : f64,
      interaction_columns = ["x"],
      interaction_coefficients = array<i64: 2>,
      main_effect_index = 1 : i64,
      treated_value = 1.000000e+00 : f64,
      treatment_name = "A"
    } : (tensor<?x?xf64>, tensor<3xf64>) -> tensor<?xf64>
    return %effects : tensor<?xf64>
  }
}
