#ifndef TRITO_INTEL_NGPU_CONVERSION_PASSES_H
#define TRITO_INTEL_NGPU_CONVERSION_PASSES_H

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "triton/Conversion/IntelGPUToSPIRV/TritonIntelGPUToSPIRVPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/IntelGPUToSPIRV/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
