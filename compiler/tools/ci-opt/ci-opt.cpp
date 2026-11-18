#include "Causal/CausalDialect.h"
#include "Causal/CausalPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  causal::registerColumnProjectionPass();
  causal::registerFusePredictionsPass();
  causal::registerLowerInterventionPredictPass();
  causal::registerLowerTreatmentEffectPass();

  mlir::DialectRegistry registry;
  registry.insert<
      causal::CausalDialect,
      mlir::arith::ArithDialect,
      mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect,
      mlir::math::MathDialect,
      mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Causal inference optimizer\n", registry));
}
