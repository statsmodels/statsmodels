#include "Causal/CausalDialect.h"
#include "Causal/CausalOps.h"

#include "Causal/CausalDialect.cpp.inc"

void causal::CausalDialect::initialize() {
  addOperations<LoadDatasetOp, TreatmentEffectOp>();
}
