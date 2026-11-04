#ifndef CAUSAL_OPS_H
#define CAUSAL_OPS_H

#include "Causal/CausalDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Causal/CausalOps.h.inc"

#endif // CAUSAL_OPS_H
