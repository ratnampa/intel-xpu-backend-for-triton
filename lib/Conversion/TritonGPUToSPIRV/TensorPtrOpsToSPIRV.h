#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_TENSOR_PTR_OPS_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_TENSOR_PTR_OPS_H

#include "TritonGPUToSPIRVBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateTensorPtrOpsToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

#endif
