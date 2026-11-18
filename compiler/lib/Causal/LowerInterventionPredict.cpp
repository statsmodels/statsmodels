#include "Causal/CausalOps.h"
#include "Causal/CausalPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace causal {
namespace {

class LowerInterventionPredictPass
    : public mlir::PassWrapper<LowerInterventionPredictPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerInterventionPredictPass)

  llvm::StringRef getArgument() const final {
    return "ci-lower-intervention-predict";
  }

  llvm::StringRef getDescription() const final {
    return "Lower ci.intervention_predict to linalg.matmul + fused linalg.generic";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::math::MathDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    llvm::SmallVector<InterventionPredictOp> ops;
    getOperation().walk([&](InterventionPredictOp op) { ops.push_back(op); });
    for (InterventionPredictOp op : ops) {
      lower(op, rewriter);
    }
  }

private:
  static mlir::Value extractOrZero(
      mlir::IRRewriter &rewriter,
      mlir::Location loc,
      mlir::Value coeffs,
      int64_t index,
      mlir::Value zero) {
    if (index < 0) {
      return zero;
    }
    mlir::Value cIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, index);
    return rewriter.create<mlir::tensor::ExtractOp>(
        loc, coeffs, mlir::ValueRange{cIdx});
  }

  static void lower(InterventionPredictOp op, mlir::IRRewriter &rewriter) {
    mlir::Location loc = op.getLoc();
    mlir::Value data = op.getData();
    mlir::Value coeffs = op.getCoefficients();
    auto outType =
        llvm::cast<mlir::RankedTensorType>(op.getPredictions().getType());
    auto dataType = llvm::cast<mlir::RankedTensorType>(data.getType());

    rewriter.setInsertionPoint(op);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.0));

    // 1. Gather W = [beta_x, beta_int] as tensor<Dx2xf64> (row-major).
    auto baseIdx = op.getBaseCoefficients();
    auto intIdx = op.getInteractionCoefficients();
    int64_t nCovs = static_cast<int64_t>(baseIdx.size());
    llvm::SmallVector<mlir::Value> wElements;
    wElements.reserve(nCovs * 2);
    for (int64_t j = 0; j < nCovs; ++j) {
      wElements.push_back(extractOrZero(rewriter, loc, coeffs, baseIdx[j], zero));
      wElements.push_back(extractOrZero(rewriter, loc, coeffs, intIdx[j], zero));
    }
    auto wType = mlir::RankedTensorType::get({nCovs, 2}, rewriter.getF64Type());
    mlir::Value w =
        rewriter.create<mlir::tensor::FromElementsOp>(loc, wType, wElements);

    // 2. Compute shared M = X @ W (Nx2) once via linalg.matmul.
    llvm::SmallVector<mlir::Value> dynRows;
    if (dataType.isDynamicDim(0)) {
      dynRows.push_back(rewriter.create<mlir::tensor::DimOp>(loc, data, 0));
    }
    auto mType = mlir::RankedTensorType::get(
        {dataType.getDimSize(0), 2}, rewriter.getF64Type());
    mlir::Value mEmpty =
        rewriter.create<mlir::tensor::EmptyOp>(loc, mType, dynRows);
    mlir::Value mZero =
        rewriter
            .create<mlir::linalg::FillOp>(
                loc, mlir::ValueRange{zero}, mlir::ValueRange{mEmpty})
            .getResult(0);
    mlir::Value m =
        rewriter
            .create<mlir::linalg::MatmulOp>(
                loc, mType, mlir::ValueRange{data, w}, mlir::ValueRange{mZero})
            .getResult(0);

    // 3. Fused (N, K) epilogue: eta = (m0 + c0) + (m1 + c1) * dose_k, then link^-1.
    mlir::Value c0 = extractOrZero(
        rewriter, loc, coeffs, op.getInterceptIndex(), zero);
    mlir::Value c1 = extractOrZero(
        rewriter, loc, coeffs, op.getTreatmentIndex(), zero);

    auto doses = op.getDoses();
    auto dosesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(doses.size())}, rewriter.getF64Type());
    mlir::Value dosesConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, dosesType, rewriter.getF64TensorAttr(doses));

    mlir::Value outEmpty =
        rewriter.create<mlir::tensor::EmptyOp>(loc, outType, dynRows);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, ctx);
    mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, ctx);
    mlir::AffineMap m0Map =
        mlir::AffineMap::get(2, 0, {d0, mlir::getAffineConstantExpr(0, ctx)}, ctx);
    mlir::AffineMap m1Map =
        mlir::AffineMap::get(2, 0, {d0, mlir::getAffineConstantExpr(1, ctx)}, ctx);
    mlir::AffineMap doseMap = mlir::AffineMap::get(2, 0, {d1}, ctx);
    mlir::AffineMap outMap = mlir::AffineMap::get(2, 0, {d0, d1}, ctx);

    llvm::SmallVector<mlir::AffineMap> maps{m0Map, m1Map, doseMap, outMap};
    llvm::SmallVector<mlir::utils::IteratorType> iters{
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel};

    llvm::StringRef link = op.getLink();
    auto fused = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{m, m, dosesConst},
        mlir::ValueRange{outEmpty},
        maps,
        iters,
        [&](mlir::OpBuilder &b, mlir::Location bLoc, mlir::ValueRange args) {
          mlir::Value base = b.create<mlir::arith::AddFOp>(bLoc, args[0], c0);
          mlir::Value slope = b.create<mlir::arith::AddFOp>(bLoc, args[1], c1);
          mlir::Value scaled =
              b.create<mlir::arith::MulFOp>(bLoc, slope, args[2]);
          mlir::Value eta = b.create<mlir::arith::AddFOp>(bLoc, base, scaled);

          mlir::Value result = eta;
          if (link == "logit") {
            mlir::Value one = b.create<mlir::arith::ConstantOp>(
                bLoc, b.getF64Type(), b.getF64FloatAttr(1.0));
            mlir::Value negEta = b.create<mlir::arith::NegFOp>(bLoc, eta);
            mlir::Value expNeg = b.create<mlir::math::ExpOp>(bLoc, negEta);
            mlir::Value denom = b.create<mlir::arith::AddFOp>(bLoc, one, expNeg);
            result = b.create<mlir::arith::DivFOp>(bLoc, one, denom);
          } else if (link == "log") {
            result = b.create<mlir::math::ExpOp>(bLoc, eta);
          }
          b.create<mlir::linalg::YieldOp>(bLoc, result);
        });

    rewriter.replaceOp(op, fused.getResult(0));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerInterventionPredictPass() {
  return std::make_unique<LowerInterventionPredictPass>();
}

void registerLowerInterventionPredictPass() {
  mlir::PassRegistration<LowerInterventionPredictPass>();
}

} // namespace causal
