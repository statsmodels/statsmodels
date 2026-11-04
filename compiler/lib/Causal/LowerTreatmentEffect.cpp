#include "Causal/CausalOps.h"
#include "Causal/CausalPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace causal {
namespace {

class LowerTreatmentEffectPass
    : public mlir::PassWrapper<LowerTreatmentEffectPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTreatmentEffectPass)

  llvm::StringRef getArgument() const final {
    return "ci-lower-treatment-effect";
  }

  llvm::StringRef getDescription() const final {
    return "Lower ci.treatment_effect to Linalg, tensor, and arith operations";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    llvm::SmallVector<TreatmentEffectOp> effects;
    getOperation().walk(
        [&](TreatmentEffectOp effect) { effects.push_back(effect); });

    for (TreatmentEffectOp effect : effects) {
      lower(effect, rewriter);
    }
  }

private:
  static void lower(TreatmentEffectOp effect, mlir::IRRewriter &rewriter) {
    mlir::Location location = effect.getLoc();
    mlir::Value data = effect.getData();
    mlir::Value coefficients = effect.getCoefficients();
    auto result_type =
        llvm::cast<mlir::RankedTensorType>(effect.getEffects().getType());

    rewriter.setInsertionPoint(effect);
    mlir::Value row_count =
        rewriter.create<mlir::tensor::DimOp>(location, data, 0);
    llvm::SmallVector<mlir::Value> dynamic_sizes;
    if (result_type.isDynamicDim(0)) {
      dynamic_sizes.push_back(row_count);
    }
    mlir::Value output = rewriter.create<mlir::tensor::EmptyOp>(
        location, result_type, dynamic_sizes);
    mlir::ArrayAttr interaction_columns = effect.getInteractionColumns();
    auto interaction_coefficients = effect.getInteractionCoefficients();
    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::AffineMap> indexing_maps;
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::AffineExpr row = mlir::getAffineDimExpr(0, context);

    auto source = effect.getData().getDefiningOp<LoadDatasetOp>();
    auto source_columns = source ? source.getColumns() : mlir::ArrayAttr();
    for (auto indexed_column : llvm::enumerate(interaction_columns)) {
      int64_t feature_position = indexed_column.index();
      if (source) {
        auto column = llvm::cast<mlir::StringAttr>(indexed_column.value());
        auto found = llvm::find_if(
            source_columns, [&](mlir::Attribute attribute) {
              return llvm::cast<mlir::StringAttr>(attribute) == column;
            });
        feature_position = std::distance(source_columns.begin(), found);
      }
      inputs.push_back(data);
      indexing_maps.push_back(mlir::AffineMap::get(
          1,
          0,
          {row, mlir::getAffineConstantExpr(feature_position, context)},
          context));
    }

    llvm::SmallVector<int64_t> coefficient_indices;
    coefficient_indices.push_back(effect.getMainEffectIndex());
    for (int64_t index : interaction_coefficients) {
      coefficient_indices.push_back(index);
    }
    for (int64_t index : coefficient_indices) {
      inputs.push_back(coefficients);
      indexing_maps.push_back(mlir::AffineMap::get(
          1, 0, {mlir::getAffineConstantExpr(index, context)}, context));
    }

    indexing_maps.push_back(mlir::AffineMap::getMultiDimIdentityMap(1, context));
    llvm::SmallVector<mlir::utils::IteratorType> iterator_types{
        mlir::utils::IteratorType::parallel};
    double treatment_delta = effect.getTreatedValue().convertToDouble()
        - effect.getControlValue().convertToDouble();
    auto generic = rewriter.create<mlir::linalg::GenericOp>(
        location,
        mlir::TypeRange{result_type},
        inputs,
        mlir::ValueRange{output},
        indexing_maps,
        iterator_types,
        [&](mlir::OpBuilder &builder,
            mlir::Location body_location,
            mlir::ValueRange arguments) {
          size_t interaction_count = interaction_columns.size();
          mlir::Value accumulator = arguments[interaction_count];
          for (size_t index = 0; index < interaction_count; ++index) {
            mlir::Value product = builder.create<mlir::arith::MulFOp>(
                body_location,
                arguments[index],
                arguments[interaction_count + 1 + index]);
            accumulator = builder.create<mlir::arith::AddFOp>(
                body_location, accumulator, product);
          }
          mlir::Value delta = builder.create<mlir::arith::ConstantOp>(
              body_location,
              builder.getF64Type(),
              builder.getF64FloatAttr(treatment_delta));
          mlir::Value row_effect = builder.create<mlir::arith::MulFOp>(
              body_location, accumulator, delta);
          builder.create<mlir::linalg::YieldOp>(body_location, row_effect);
        });

    rewriter.replaceOp(effect, generic.getResult(0));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerTreatmentEffectPass() {
  return std::make_unique<LowerTreatmentEffectPass>();
}

void registerLowerTreatmentEffectPass() {
  mlir::PassRegistration<LowerTreatmentEffectPass>();
}

} // namespace causal
