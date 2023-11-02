#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"

//===----------------------------------------------------------------------===//
// Intel MMA Layout Encoding
//===----------------------------------------------------------------------===//

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TTIG_DummyOp : TTIG_Op<"dummy_op"> {
  let summary = "dummy op";

  let description = [{
    dummy
  }];

  let arguments = (ins TT_Ptr:$mbarrier);

  let assemblyFormat = "$mbarrier `,` attr-dict `:` type($mbarrier)";
}

#endif
