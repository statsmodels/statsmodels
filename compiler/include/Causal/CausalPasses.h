#ifndef CAUSAL_PASSES_H
#define CAUSAL_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace causal {

std::unique_ptr<mlir::Pass> createColumnProjectionPass();
std::unique_ptr<mlir::Pass> createLowerTreatmentEffectPass();
void registerColumnProjectionPass();
void registerLowerTreatmentEffectPass();

}

#endif // CAUSAL_PASSES_H
