#include "Causal/CausalOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <cmath>

#define GET_OP_CLASSES
#include "Causal/CausalOps.cpp.inc"

mlir::LogicalResult causal::LoadDatasetOp::verify() {
  auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(getData().getType());
  if (!tensor_type || tensor_type.getRank() != 2) {
    return emitOpError("result must be a rank-2 tensor");
  }
  if (!tensor_type.getElementType().isF64()) {
    return emitOpError("result tensor elements must be f64");
  }
  if (getPath().empty()) {
    return emitOpError("path must not be empty");
  }
  if (!tensor_type.isDynamicDim(1)
      && tensor_type.getDimSize(1) !=
          static_cast<int64_t>(getColumns().size())) {
    return emitOpError("result column count must match the columns attribute");
  }
  if (getColumns().empty()) {
    return emitOpError("requires at least one source column");
  }
  llvm::SmallDenseSet<llvm::StringRef, 8> column_names;
  for (mlir::Attribute column : getColumns()) {
    auto name = llvm::dyn_cast<mlir::StringAttr>(column);
    if (!name) {
      return emitOpError("column names must be strings");
    }
    if (name.getValue().empty()) {
      return emitOpError("column names must not be empty");
    }
    if (!column_names.insert(name.getValue()).second) {
      return emitOpError("column names must be unique");
    }
  }
  return mlir::success();
}

mlir::LogicalResult causal::TreatmentEffectOp::verify() {
  auto data_type = llvm::dyn_cast<mlir::RankedTensorType>(getData().getType());
  auto coefficient_type =
      llvm::dyn_cast<mlir::RankedTensorType>(getCoefficients().getType());
  auto effects_type = llvm::dyn_cast<mlir::RankedTensorType>(getEffects().getType());
  if (!data_type || data_type.getRank() != 2) {
    return emitOpError("data operand must be a rank-2 tensor");
  }
  if (!coefficient_type || coefficient_type.getRank() != 1) {
    return emitOpError("coefficients operand must be a rank-1 tensor");
  }
  if (!effects_type || effects_type.getRank() != 1) {
    return emitOpError("result must be a rank-1 tensor");
  }
  if (!data_type.getElementType().isF64()
      || !coefficient_type.getElementType().isF64()
      || !effects_type.getElementType().isF64()) {
    return emitOpError("all tensor elements must be f64");
  }
  if (getTreatmentName().empty()) {
    return emitOpError("treatment_name must not be empty");
  }
  if (!std::isfinite(getControlValue().convertToDouble())
      || !std::isfinite(getTreatedValue().convertToDouble())
      || getControlValue() == getTreatedValue()) {
    return emitOpError(
        "control_value and treated_value must be distinct finite values");
  }
  if (getMainEffectIndex() < 0
      || (coefficient_type.hasStaticShape()
          && getMainEffectIndex() >= coefficient_type.getDimSize(0))) {
    return emitOpError("main_effect_index is outside the coefficient vector");
  }
  if (getInteractionColumns().size()
      != getInteractionCoefficients().size()) {
    return emitOpError(
        "interaction_columns and interaction_coefficients must have equal "
        "lengths");
  }
  if (getInteractionColumns().empty()) {
    return emitOpError("requires at least one treatment interaction");
  }
  if (!data_type.isDynamicDim(1)
      && data_type.getDimSize(1)
          < static_cast<int64_t>(getInteractionColumns().size())) {
    return emitOpError("data tensor has fewer columns than interactions");
  }
  if (effects_type.hasStaticShape() && data_type.hasStaticShape()
      && effects_type.getDimSize(0) != data_type.getDimSize(0)) {
    return emitOpError("result row count must match data row count");
  }
  llvm::SmallDenseSet<int64_t, 8> coefficient_indices;
  for (int64_t index : getInteractionCoefficients()) {
    if (index < 0
        || (coefficient_type.hasStaticShape()
            && index >= coefficient_type.getDimSize(0))) {
      return emitOpError(
          "interaction coefficient index is outside the coefficient vector");
    }
    if (index == getMainEffectIndex()
        || !coefficient_indices.insert(index).second) {
      return emitOpError(
          "interaction coefficient indices must be distinct from each other "
          "and the main effect");
    }
  }
  llvm::SmallDenseSet<llvm::StringRef, 8> interaction_names;
  for (mlir::Attribute column : getInteractionColumns()) {
    auto name = llvm::dyn_cast<mlir::StringAttr>(column);
    if (!name) {
      return emitOpError("interaction column names must be strings");
    }
    if (!interaction_names.insert(name.getValue()).second) {
      return emitOpError("interaction column names must be unique");
    }
    if (auto source = getData().getDefiningOp<LoadDatasetOp>()) {
      if (!llvm::is_contained(source.getColumns(), name)) {
        return emitOpError("interaction column '")
               << name.getValue() << "' is not provided by the dataset";
      }
    }
  }
  return mlir::success();
}

static mlir::LogicalResult verifyPredictCommon(
    mlir::Operation *op,
    mlir::RankedTensorType dataType,
    mlir::RankedTensorType coeffType,
    llvm::StringRef link,
    llvm::ArrayRef<int64_t> baseCoeffs,
    llvm::ArrayRef<int64_t> intCoeffs,
    int64_t interceptIdx,
    int64_t treatmentIdx) {
  if (!dataType || dataType.getRank() != 2 || !dataType.getElementType().isF64()) {
    return op->emitOpError("data must be a rank-2 f64 tensor");
  }
  if (!coeffType || coeffType.getRank() != 1 || !coeffType.getElementType().isF64()) {
    return op->emitOpError("coefficients must be a rank-1 f64 tensor");
  }
  if (link != "identity" && link != "logit" && link != "log") {
    return op->emitOpError("link must be 'identity', 'logit', or 'log'");
  }
  if (baseCoeffs.size() != intCoeffs.size() || baseCoeffs.empty()) {
    return op->emitOpError("base_coefficients and interaction_coefficients must be non-empty and equal length");
  }
  if (!dataType.isDynamicDim(1)
      && dataType.getDimSize(1) != static_cast<int64_t>(baseCoeffs.size())) {
    return op->emitOpError("data column count must match base_coefficients");
  }
  if (treatmentIdx < 0 || (coeffType.hasStaticShape() && treatmentIdx >= coeffType.getDimSize(0))) {
    return op->emitOpError("treatment_index is outside coefficients");
  }
  if (interceptIdx < -1 || (coeffType.hasStaticShape() && interceptIdx >= coeffType.getDimSize(0))) {
    return op->emitOpError("intercept_index is outside coefficients");
  }
  return mlir::success();
}

mlir::LogicalResult causal::PredictOp::verify() {
  auto dataType = llvm::dyn_cast<mlir::RankedTensorType>(getData().getType());
  auto coeffType = llvm::dyn_cast<mlir::RankedTensorType>(getCoefficients().getType());
  auto outType = llvm::dyn_cast<mlir::RankedTensorType>(getPrediction().getType());
  if (mlir::failed(verifyPredictCommon(
          getOperation(), dataType, coeffType, getLink(),
          getBaseCoefficients(), getInteractionCoefficients(),
          getInterceptIndex(), getTreatmentIndex()))) {
    return mlir::failure();
  }
  if (!outType || outType.getRank() != 1 || !outType.getElementType().isF64()) {
    return emitOpError("result must be a rank-1 f64 tensor");
  }
  if (outType.hasStaticShape() && dataType.hasStaticShape()
      && outType.getDimSize(0) != dataType.getDimSize(0)) {
    return emitOpError("result row count must match data row count");
  }
  return mlir::success();
}

mlir::LogicalResult causal::InterventionPredictOp::verify() {
  auto dataType = llvm::dyn_cast<mlir::RankedTensorType>(getData().getType());
  auto coeffType = llvm::dyn_cast<mlir::RankedTensorType>(getCoefficients().getType());
  auto outType = llvm::dyn_cast<mlir::RankedTensorType>(getPredictions().getType());
  if (mlir::failed(verifyPredictCommon(
          getOperation(), dataType, coeffType, getLink(),
          getBaseCoefficients(), getInteractionCoefficients(),
          getInterceptIndex(), getTreatmentIndex()))) {
    return mlir::failure();
  }
  if (getDoses().empty()) {
    return emitOpError("doses must not be empty");
  }
  llvm::StringRef reduction = getReduction();
  int64_t k = static_cast<int64_t>(getDoses().size());
  if (reduction == "none") {
    if (!outType || outType.getRank() != 2 || !outType.getElementType().isF64()) {
      return emitOpError("result must be a rank-2 f64 tensor");
    }
    if (outType.hasStaticShape() && dataType.hasStaticShape()
        && (outType.getDimSize(0) != dataType.getDimSize(0)
            || outType.getDimSize(1) != k)) {
      return emitOpError("result shape must be (N, K)");
    }
  } else if (reduction == "ate") {
    if (!outType || outType.getRank() != 1 || !outType.getElementType().isF64()) {
      return emitOpError("ate result must be a rank-1 f64 tensor");
    }
    if (outType.hasStaticShape() && outType.getDimSize(0) != k) {
      return emitOpError("ate result length must equal K");
    }
  } else if (reduction == "argmax") {
    if (!outType || outType.getRank() != 1 || !outType.getElementType().isInteger(64)) {
      return emitOpError("argmax result must be a rank-1 i64 tensor");
    }
    if (outType.hasStaticShape() && dataType.hasStaticShape()
        && outType.getDimSize(0) != dataType.getDimSize(0)) {
      return emitOpError("argmax result length must equal N");
    }
  } else {
    return emitOpError("reduction must be 'none', 'ate', or 'argmax'");
  }
  return mlir::success();
}
