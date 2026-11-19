#include "Causal/CausalOps.h"
#include "Causal/CausalPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
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
    return "Lower ci.treatment_effect to gather + linalg.matvec + linalg.generic";
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
    mlir::Location loc = effect.getLoc();
    mlir::Value data = effect.getData();
    mlir::Value coefficients = effect.getCoefficients();
    auto dataType = llvm::cast<mlir::RankedTensorType>(data.getType());
    auto resultType =
        llvm::cast<mlir::RankedTensorType>(effect.getEffects().getType());

    rewriter.setInsertionPoint(effect);

    // 1. Gather interaction weights into a contiguous 1-D tensor matching
    //    data's column count (mapping through LoadDatasetOp if present).
    int64_t dataCols = dataType.getDimSize(1);
    llvm::SmallVector<int64_t> idxData(dataCols, 0);
    llvm::SmallVector<double> maskData(dataCols, 0.0);

    mlir::ArrayAttr interactionColumns = effect.getInteractionColumns();
    auto interactionCoeffs = effect.getInteractionCoefficients();
    auto source = data.getDefiningOp<LoadDatasetOp>();
    auto sourceColumns = source ? source.getColumns() : mlir::ArrayAttr();

    for (auto indexedCol : llvm::enumerate(interactionColumns)) {
      int64_t pos = static_cast<int64_t>(indexedCol.index());
      if (source) {
        auto colAttr = llvm::cast<mlir::StringAttr>(indexedCol.value());
        auto found = llvm::find_if(sourceColumns, [&](mlir::Attribute attr) {
          return llvm::cast<mlir::StringAttr>(attr) == colAttr;
        });
        pos = std::distance(sourceColumns.begin(), found);
      }
      idxData[pos] = interactionCoeffs[indexedCol.index()];
      maskData[pos] = 1.0;
    }

    auto idxType =
        mlir::RankedTensorType::get({dataCols}, rewriter.getI64Type());
    auto weightType =
        mlir::RankedTensorType::get({dataCols}, rewriter.getF64Type());
    mlir::Value idxConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, idxType,
        mlir::DenseElementsAttr::get(idxType, llvm::ArrayRef<int64_t>(idxData)));
    mlir::Value maskConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, weightType,
        mlir::DenseElementsAttr::get(weightType, llvm::ArrayRef<double>(maskData)));
    mlir::Value wEmpty = rewriter.create<mlir::tensor::EmptyOp>(
        loc, weightType, mlir::ValueRange{});

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::AffineMap id1d = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
    mlir::Value weights =
        rewriter
            .create<mlir::linalg::GenericOp>(
                loc,
                mlir::TypeRange{weightType},
                mlir::ValueRange{idxConst, maskConst},
                mlir::ValueRange{wEmpty},
                llvm::SmallVector<mlir::AffineMap>{id1d, id1d, id1d},
                llvm::SmallVector<mlir::utils::IteratorType>{
                    mlir::utils::IteratorType::parallel},
                [&](mlir::OpBuilder &b, mlir::Location bLoc,
                    mlir::ValueRange args) {
                  mlir::Value idx = b.create<mlir::arith::IndexCastOp>(
                      bLoc, b.getIndexType(), args[0]);
                  mlir::Value val = b.create<mlir::tensor::ExtractOp>(
                      bLoc, coefficients, mlir::ValueRange{idx});
                  mlir::Value masked =
                      b.create<mlir::arith::MulFOp>(bLoc, val, args[1]);
                  b.create<mlir::linalg::YieldOp>(bLoc, masked);
                })
            .getResult(0);

    // 2. Compute data @ weights via linalg.matvec.
    llvm::SmallVector<mlir::Value> dynamicSizes;
    if (resultType.isDynamicDim(0)) {
      dynamicSizes.push_back(
          rewriter.create<mlir::tensor::DimOp>(loc, data, 0));
    }
    mlir::Value emptyInit = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resultType, dynamicSizes);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.0));
    mlir::Value zeroed =
        rewriter
            .create<mlir::linalg::FillOp>(
                loc, mlir::ValueRange{zero}, mlir::ValueRange{emptyInit})
            .getResult(0);
    mlir::Value dot =
        rewriter
            .create<mlir::linalg::MatvecOp>(
                loc, resultType, mlir::ValueRange{data, weights},
                mlir::ValueRange{zeroed})
            .getResult(0);

    // 3. Add main_effect and scale by (treated_value - control_value).
    mlir::Value cMainIdx = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, effect.getMainEffectIndex());
    mlir::Value mainEffect = rewriter.create<mlir::tensor::ExtractOp>(
        loc, coefficients, mlir::ValueRange{cMainIdx});
    double treatmentDelta = effect.getTreatedValue().convertToDouble()
        - effect.getControlValue().convertToDouble();
    mlir::Value delta = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(treatmentDelta));

    mlir::Value emptyOut = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resultType, dynamicSizes);
    auto scaled = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{resultType},
        mlir::ValueRange{dot},
        mlir::ValueRange{emptyOut},
        llvm::SmallVector<mlir::AffineMap>{id1d, id1d},
        llvm::SmallVector<mlir::utils::IteratorType>{
            mlir::utils::IteratorType::parallel},
        [&](mlir::OpBuilder &b, mlir::Location bLoc, mlir::ValueRange args) {
          mlir::Value sum =
              b.create<mlir::arith::AddFOp>(bLoc, args[0], mainEffect);
          mlir::Value rowEffect =
              b.create<mlir::arith::MulFOp>(bLoc, sum, delta);
          b.create<mlir::linalg::YieldOp>(bLoc, rowEffect);
        });

    rewriter.replaceOp(effect, scaled.getResult(0));
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
