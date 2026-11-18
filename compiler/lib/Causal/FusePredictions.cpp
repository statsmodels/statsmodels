#include "Causal/CausalOps.h"
#include "Causal/CausalPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace causal {
namespace {

class FusePredictionsPass
    : public mlir::PassWrapper<FusePredictionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePredictionsPass)

  llvm::StringRef getArgument() const final { return "ci-fuse-predictions"; }

  llvm::StringRef getDescription() const final {
    return "Fuse multiple ci.predict ops on the same model/data into ci.intervention_predict";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    llvm::SmallVector<PredictOp> ops;
    func.walk([&](PredictOp op) { ops.push_back(op); });
    if (ops.size() < 2) {
      return;
    }

    llvm::SmallVector<bool> consumed(ops.size(), false);
    mlir::IRRewriter rewriter(&getContext());

    for (size_t i = 0; i < ops.size(); ++i) {
      if (consumed[i]) {
        continue;
      }
      PredictOp anchor = ops[i];
      llvm::SmallVector<PredictOp> group{anchor};
      consumed[i] = true;

      for (size_t j = i + 1; j < ops.size(); ++j) {
        if (consumed[j]) {
          continue;
        }
        PredictOp cand = ops[j];
        if (cand.getData() == anchor.getData()
            && cand.getCoefficients() == anchor.getCoefficients()
            && cand.getTreatmentName() == anchor.getTreatmentName()
            && cand.getLink() == anchor.getLink()
            && cand.getBaseCoefficients() == anchor.getBaseCoefficients()
            && cand.getInteractionCoefficients()
                   == anchor.getInteractionCoefficients()
            && cand.getInterceptIndex() == anchor.getInterceptIndex()
            && cand.getTreatmentIndex() == anchor.getTreatmentIndex()) {
          group.push_back(cand);
          consumed[j] = true;
        }
      }

      if (group.size() < 2) {
        continue;
      }

      llvm::SmallVector<double> doses;
      doses.reserve(group.size());
      for (PredictOp op : group) {
        doses.push_back(op.getDose().convertToDouble());
      }

      auto dataType =
          llvm::cast<mlir::RankedTensorType>(anchor.getData().getType());
      auto outType = mlir::RankedTensorType::get(
          {dataType.getDimSize(0), static_cast<int64_t>(group.size())},
          rewriter.getF64Type());

      rewriter.setInsertionPoint(anchor);
      auto fused = rewriter.create<InterventionPredictOp>(
          anchor.getLoc(),
          outType,
          anchor.getData(),
          anchor.getCoefficients(),
          anchor.getTreatmentNameAttr(),
          anchor.getLinkAttr(),
          rewriter.getDenseF64ArrayAttr(doses),
          anchor.getBaseCoefficientsAttr(),
          anchor.getInteractionCoefficientsAttr(),
          anchor.getInterceptIndexAttr(),
          anchor.getTreatmentIndexAttr());

      auto sliceType = llvm::cast<mlir::RankedTensorType>(
          anchor.getPrediction().getType());
      mlir::OpFoldResult rowSize = dataType.isDynamicDim(0)
          ? mlir::OpFoldResult(rewriter.create<mlir::tensor::DimOp>(
                anchor.getLoc(), anchor.getData(), 0)
                                   .getResult())
          : mlir::OpFoldResult(rewriter.getIndexAttr(dataType.getDimSize(0)));

      for (auto it : llvm::enumerate(group)) {
        llvm::SmallVector<mlir::OpFoldResult> offsets{
            rewriter.getIndexAttr(0),
            rewriter.getIndexAttr(static_cast<int64_t>(it.index()))};
        llvm::SmallVector<mlir::OpFoldResult> sizes{
            rowSize, rewriter.getIndexAttr(1)};
        llvm::SmallVector<mlir::OpFoldResult> strides{
            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
        auto slice = rewriter.create<mlir::tensor::ExtractSliceOp>(
            it.value().getLoc(),
            sliceType,
            fused.getPredictions(),
            offsets,
            sizes,
            strides);
        rewriter.replaceOp(it.value(), slice.getResult());
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFusePredictionsPass() {
  return std::make_unique<FusePredictionsPass>();
}

void registerFusePredictionsPass() {
  mlir::PassRegistration<FusePredictionsPass>();
}

} // namespace causal
