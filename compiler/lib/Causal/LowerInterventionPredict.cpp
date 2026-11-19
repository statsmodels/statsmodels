#include "Causal/CausalOps.h"
#include "Causal/CausalPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
                    mlir::math::MathDialect, mlir::scf::SCFDialect,
                    mlir::tensor::TensorDialect>();
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

    // 1. Gather W = [beta_x, beta_int] as tensor<Dx2xf64> via linalg.generic
    //    (avoids IREE stream.async.fill bug with tensor.from_elements at D>=25).
    auto baseIdx = op.getBaseCoefficients();
    auto intIdx = op.getInteractionCoefficients();
    int64_t nCovs = static_cast<int64_t>(baseIdx.size());
    llvm::SmallVector<int64_t> idxData;
    llvm::SmallVector<double> maskData;
    idxData.reserve(nCovs * 2);
    maskData.reserve(nCovs * 2);
    for (int64_t j = 0; j < nCovs; ++j) {
      idxData.push_back(baseIdx[j] >= 0 ? baseIdx[j] : 0);
      maskData.push_back(baseIdx[j] >= 0 ? 1.0 : 0.0);
      idxData.push_back(intIdx[j] >= 0 ? intIdx[j] : 0);
      maskData.push_back(intIdx[j] >= 0 ? 1.0 : 0.0);
    }

    auto idxType = mlir::RankedTensorType::get({nCovs, 2}, rewriter.getI64Type());
    auto wType = mlir::RankedTensorType::get({nCovs, 2}, rewriter.getF64Type());
    mlir::Value idxConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, idxType,
        mlir::DenseElementsAttr::get(idxType, llvm::ArrayRef<int64_t>(idxData)));
    mlir::Value maskConst = rewriter.create<mlir::arith::ConstantOp>(
        loc, wType,
        mlir::DenseElementsAttr::get(wType, llvm::ArrayRef<double>(maskData)));
    mlir::Value wEmpty = rewriter.create<mlir::tensor::EmptyOp>(
        loc, wType, mlir::ValueRange{});

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::AffineMap id2d = mlir::AffineMap::getMultiDimIdentityMap(2, ctx);
    llvm::SmallVector<mlir::AffineMap> gatherMaps{id2d, id2d, id2d};
    llvm::SmallVector<mlir::utils::IteratorType> gatherIters{
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel};
    mlir::Value w =
        rewriter
            .create<mlir::linalg::GenericOp>(
                loc,
                mlir::TypeRange{wType},
                mlir::ValueRange{idxConst, maskConst},
                mlir::ValueRange{wEmpty},
                gatherMaps,
                gatherIters,
                [&](mlir::OpBuilder &b, mlir::Location bLoc,
                    mlir::ValueRange args) {
                  mlir::Value idx = b.create<mlir::arith::IndexCastOp>(
                      bLoc, b.getIndexType(), args[0]);
                  mlir::Value val = b.create<mlir::tensor::ExtractOp>(
                      bLoc, coeffs, mlir::ValueRange{idx});
                  mlir::Value masked =
                      b.create<mlir::arith::MulFOp>(bLoc, val, args[1]);
                  b.create<mlir::linalg::YieldOp>(bLoc, masked);
                })
            .getResult(0);

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
        loc, dosesType,
        mlir::DenseElementsAttr::get(dosesType, llvm::ArrayRef<double>(doses)));

    mlir::Value outEmpty =
        rewriter.create<mlir::tensor::EmptyOp>(loc, outType, dynRows);

    mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, ctx);
    mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, ctx);
    llvm::StringRef link = op.getLink();
    llvm::StringRef reduction = op.getReduction();

    auto emitPred = [&](mlir::OpBuilder &b, mlir::Location bLoc,
                        mlir::Value m0, mlir::Value m1, mlir::Value dose) {
      mlir::Value base = b.create<mlir::arith::AddFOp>(bLoc, m0, c0);
      mlir::Value slope = b.create<mlir::arith::AddFOp>(bLoc, m1, c1);
      mlir::Value scaled = b.create<mlir::arith::MulFOp>(bLoc, slope, dose);
      mlir::Value eta = b.create<mlir::arith::AddFOp>(bLoc, base, scaled);
      if (link == "logit") {
        mlir::Value one = b.create<mlir::arith::ConstantOp>(
            bLoc, b.getF64Type(), b.getF64FloatAttr(1.0));
        mlir::Value negEta = b.create<mlir::arith::NegFOp>(bLoc, eta);
        mlir::Value expNeg = b.create<mlir::math::ExpOp>(bLoc, negEta);
        mlir::Value denom = b.create<mlir::arith::AddFOp>(bLoc, one, expNeg);
        return b.create<mlir::arith::DivFOp>(bLoc, one, denom).getResult();
      }
      if (link == "log") {
        return b.create<mlir::math::ExpOp>(bLoc, eta).getResult();
      }
      return eta;
    };

    if (reduction == "ate") {
      auto kType = mlir::RankedTensorType::get(
          {static_cast<int64_t>(doses.size())}, rewriter.getF64Type());
      mlir::Value kEmpty = rewriter.create<mlir::tensor::EmptyOp>(
          loc, kType, mlir::ValueRange{});
      mlir::Value kZero =
          rewriter
              .create<mlir::linalg::FillOp>(
                  loc, mlir::ValueRange{zero}, mlir::ValueRange{kEmpty})
              .getResult(0);
      mlir::AffineMap m0Map =
          mlir::AffineMap::get(2, 0, {d0, mlir::getAffineConstantExpr(0, ctx)}, ctx);
      mlir::AffineMap m1Map =
          mlir::AffineMap::get(2, 0, {d0, mlir::getAffineConstantExpr(1, ctx)}, ctx);
      mlir::AffineMap doseMap = mlir::AffineMap::get(2, 0, {d1}, ctx);
      mlir::AffineMap kMap = mlir::AffineMap::get(2, 0, {d1}, ctx);
      llvm::SmallVector<mlir::AffineMap> maps{m0Map, m1Map, doseMap, kMap};
      llvm::SmallVector<mlir::utils::IteratorType> iters{
          mlir::utils::IteratorType::reduction,
          mlir::utils::IteratorType::parallel};
      auto sumOp = rewriter.create<mlir::linalg::GenericOp>(
          loc,
          mlir::TypeRange{kType},
          mlir::ValueRange{m, m, dosesConst},
          mlir::ValueRange{kZero},
          maps,
          iters,
          [&](mlir::OpBuilder &b, mlir::Location bLoc, mlir::ValueRange args) {
            mlir::Value p = emitPred(b, bLoc, args[0], args[1], args[2]);
            mlir::Value acc = b.create<mlir::arith::AddFOp>(bLoc, args[3], p);
            b.create<mlir::linalg::YieldOp>(bLoc, acc);
          });

      mlir::Value nRowsIdx = dataType.isDynamicDim(0)
          ? dynRows[0]
          : rewriter.create<mlir::arith::ConstantIndexOp>(
                loc, dataType.getDimSize(0))
                .getResult();
      mlir::Value nRowsI64 = rewriter.create<mlir::arith::IndexCastOp>(
          loc, rewriter.getI64Type(), nRowsIdx);
      mlir::Value nRowsF64 = rewriter.create<mlir::arith::SIToFPOp>(
          loc, rewriter.getF64Type(), nRowsI64);
      mlir::AffineMap id1d = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
      mlir::Value meanEmpty = rewriter.create<mlir::tensor::EmptyOp>(
          loc, outType, mlir::ValueRange{});
      auto meanOp = rewriter.create<mlir::linalg::GenericOp>(
          loc,
          mlir::TypeRange{outType},
          mlir::ValueRange{sumOp.getResult(0)},
          mlir::ValueRange{meanEmpty},
          llvm::SmallVector<mlir::AffineMap>{id1d, id1d},
          llvm::SmallVector<mlir::utils::IteratorType>{
              mlir::utils::IteratorType::parallel},
          [&](mlir::OpBuilder &b, mlir::Location bLoc, mlir::ValueRange args) {
            mlir::Value avg =
                b.create<mlir::arith::DivFOp>(bLoc, args[0], nRowsF64);
            b.create<mlir::linalg::YieldOp>(bLoc, avg);
          });
      rewriter.replaceOp(op, meanOp.getResult(0));
      return;
    }

    if (reduction == "argmax") {
      mlir::AffineMap m0Map =
          mlir::AffineMap::get(1, 0, {d0, mlir::getAffineConstantExpr(0, ctx)}, ctx);
      mlir::AffineMap m1Map =
          mlir::AffineMap::get(1, 0, {d0, mlir::getAffineConstantExpr(1, ctx)}, ctx);
      mlir::AffineMap outMap = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
      auto argmaxOp = rewriter.create<mlir::linalg::GenericOp>(
          loc,
          mlir::TypeRange{outType},
          mlir::ValueRange{m, m},
          mlir::ValueRange{outEmpty},
          llvm::SmallVector<mlir::AffineMap>{m0Map, m1Map, outMap},
          llvm::SmallVector<mlir::utils::IteratorType>{
              mlir::utils::IteratorType::parallel},
          [&](mlir::OpBuilder &b, mlir::Location bLoc, mlir::ValueRange args) {
            mlir::Value cZeroIdx =
                b.create<mlir::arith::ConstantIndexOp>(bLoc, 0);
            mlir::Value cOneIdx =
                b.create<mlir::arith::ConstantIndexOp>(bLoc, 1);
            mlir::Value cKIdx =
                b.create<mlir::arith::ConstantIndexOp>(bLoc, doses.size());
            mlir::Value d0Val = b.create<mlir::tensor::ExtractOp>(
                bLoc, dosesConst, mlir::ValueRange{cZeroIdx});
            mlir::Value initVal = emitPred(b, bLoc, args[0], args[1], d0Val);
            mlir::Value initIdx = b.create<mlir::arith::ConstantOp>(
                bLoc, b.getI64Type(), b.getI64IntegerAttr(0));
            auto loop = b.create<mlir::scf::ForOp>(
                bLoc,
                cOneIdx,
                cKIdx,
                cOneIdx,
                mlir::ValueRange{initVal, initIdx},
                [&](mlir::OpBuilder &lb, mlir::Location lLoc, mlir::Value k,
                    mlir::ValueRange iterArgs) {
                  mlir::Value dk = lb.create<mlir::tensor::ExtractOp>(
                      lLoc, dosesConst, mlir::ValueRange{k});
                  mlir::Value cand = emitPred(lb, lLoc, args[0], args[1], dk);
                  mlir::Value better = lb.create<mlir::arith::CmpFOp>(
                      lLoc, mlir::arith::CmpFPredicate::OGT, cand, iterArgs[0]);
                  mlir::Value kI64 = lb.create<mlir::arith::IndexCastOp>(
                      lLoc, lb.getI64Type(), k);
                  mlir::Value nextVal = lb.create<mlir::arith::SelectOp>(
                      lLoc, better, cand, iterArgs[0]);
                  mlir::Value nextIdx = lb.create<mlir::arith::SelectOp>(
                      lLoc, better, kI64, iterArgs[1]);
                  lb.create<mlir::scf::YieldOp>(
                      lLoc, mlir::ValueRange{nextVal, nextIdx});
                });
            b.create<mlir::linalg::YieldOp>(bLoc, loop.getResult(1));
          });
      rewriter.replaceOp(op, argmaxOp.getResult(0));
      return;
    }

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

    auto fused = rewriter.create<mlir::linalg::GenericOp>(
        loc,
        mlir::TypeRange{outType},
        mlir::ValueRange{m, m, dosesConst},
        mlir::ValueRange{outEmpty},
        maps,
        iters,
        [&](mlir::OpBuilder &b, mlir::Location bLoc, mlir::ValueRange args) {
          mlir::Value result = emitPred(b, bLoc, args[0], args[1], args[2]);
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
