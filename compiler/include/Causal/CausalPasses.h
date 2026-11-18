#ifndef CAUSAL_PASSES_H
#define CAUSAL_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace causal {

std::unique_ptr<mlir::Pass> createColumnProjectionPass();
std::unique_ptr<mlir::Pass> createFusePredictionsPass();
std::unique_ptr<mlir::Pass> createLowerInterventionPredictPass();
std::unique_ptr<mlir::Pass> createLowerTreatmentEffectPass();
void registerColumnProjectionPass();
void registerFusePredictionsPass();
void registerLowerInterventionPredictPass();
void registerLowerTreatmentEffectPass();

}

#endif // CAUSAL_PASSES_H
