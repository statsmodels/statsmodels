#include "Causal/CausalDialect.h"
#include "Causal/CausalOps.h"
#include "Causal/CausalPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

namespace causal {
namespace {

class ColumnProjectionPass
    : public mlir::PassWrapper<ColumnProjectionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ColumnProjectionPass)

  llvm::StringRef getArgument() const final { return "ci-project-columns"; }

  llvm::StringRef getDescription() const final {
    return "Project dataset columns to those used by treatment interactions";
  }

  void runOnOperation() override {
    llvm::DenseMap<LoadDatasetOp, llvm::SmallDenseSet<llvm::StringRef, 8>>
        required_by_source;

    getOperation().walk([&](TreatmentEffectOp effect) {
      auto source = effect.getData().getDefiningOp<LoadDatasetOp>();
      if (!source || !source.getData().hasOneUse()) {
        return;
      }

      llvm::SmallDenseSet<llvm::StringRef, 8> &required =
          required_by_source[source];
      for (mlir::Attribute attribute : effect.getInteractionColumns()) {
        auto column = llvm::cast<mlir::StringAttr>(attribute);
        required.insert(column.getValue());
      }
    });

    for (auto &entry : required_by_source) {
      LoadDatasetOp source = entry.first;
      const llvm::SmallDenseSet<llvm::StringRef, 8> &required = entry.second;
      mlir::SmallVector<mlir::Attribute> projected_columns;
      for (mlir::Operation *user : source.getData().getUsers()) {
        auto effect = llvm::dyn_cast<TreatmentEffectOp>(user);
        if (!effect) {
          continue;
        }
        for (mlir::Attribute attribute : effect.getInteractionColumns()) {
          auto column = llvm::cast<mlir::StringAttr>(attribute);
          if (required.contains(column.getValue())
              && !llvm::is_contained(projected_columns, attribute)) {
            projected_columns.push_back(attribute);
          }
        }
      }
      if (projected_columns.size() != required.size()) {
        source.emitError(
            "treatment effect references a column absent from the dataset");
        signalPassFailure();
        return;
      }
      auto old_type =
          llvm::cast<mlir::RankedTensorType>(source.getData().getType());
      auto new_type = mlir::RankedTensorType::get(
          {old_type.getDimSize(0),
           static_cast<int64_t>(projected_columns.size())},
          old_type.getElementType(), old_type.getEncoding());
      source.getData().setType(new_type);
      source->setAttr(
          "columns", mlir::ArrayAttr::get(&getContext(), projected_columns));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createColumnProjectionPass() {
  return std::make_unique<ColumnProjectionPass>();
}

void registerColumnProjectionPass() {
  mlir::PassRegistration<ColumnProjectionPass>();
}

} // namespace causal
