include "Causal/CausalDialect.td"

class Causal_Op<string mnemonic, list<Trait> traits = []>
    : Op<Causal_Dialect, mnemonic, traits>;

def Causal_LoadDatasetOp : Causal_Op<"load_dataset"> {
  let summary = "Describe a dataset source and its available columns";
  let description = [{
    Declares a dataset input, including its source path and current column
    projection. The operation is metadata for compiler transformations; it
    requires a runtime implementation before it can be executed.
  }];

  let arguments = (ins StrAttr:$path, ArrayAttr:$columns);
  let results = (outs AnyRankedTensor:$data);

  let hasVerifier = 1;
}

def Causal_TreatmentEffectOp : Causal_Op<"treatment_effect"> {
  let summary = "Compute an OLS treatment contrast from interaction features";
  let description = [{
    For each row, computes (treated_value - control_value) multiplied by the
    fitted treatment main-effect coefficient plus the sum of each interaction
    feature times its fitted interaction coefficient. If data is produced by
    ci.load_dataset, interaction column names select its columns. Otherwise,
    data columns are ordered as interaction_columns.
  }];

  let arguments = (ins
    AnyRankedTensor:$data,
    AnyRankedTensor:$coefficients,
    StrAttr:$treatment_name,
    F64Attr:$control_value,
    F64Attr:$treated_value,
    I64Attr:$main_effect_index,
    ArrayAttr:$interaction_columns,
    DenseI64ArrayAttr:$interaction_coefficients
  );
  let results = (outs AnyRankedTensor:$effects);

  let hasVerifier = 1;
}

def Causal_PredictOp : Causal_Op<"predict"> {
  let summary = "Predict outcome for a single treatment dose";
  let arguments = (ins
    AnyRankedTensor:$data,
    AnyRankedTensor:$coefficients,
    StrAttr:$treatment_name,
    StrAttr:$link,
    F64Attr:$dose,
    DenseI64ArrayAttr:$base_coefficients,
    DenseI64ArrayAttr:$interaction_coefficients,
    I64Attr:$intercept_index,
    I64Attr:$treatment_index
  );
  let results = (outs AnyRankedTensor:$prediction);
  let hasVerifier = 1;
}

def Causal_InterventionPredictOp : Causal_Op<"intervention_predict"> {
  let summary = "Predict outcomes for K treatment doses in one pass";
  let arguments = (ins
    AnyRankedTensor:$data,
    AnyRankedTensor:$coefficients,
    StrAttr:$treatment_name,
    StrAttr:$link,
    DenseF64ArrayAttr:$doses,
    DenseI64ArrayAttr:$base_coefficients,
    DenseI64ArrayAttr:$interaction_coefficients,
    I64Attr:$intercept_index,
    I64Attr:$treatment_index
  );
  let results = (outs AnyRankedTensor:$predictions);
  let hasVerifier = 1;
}
