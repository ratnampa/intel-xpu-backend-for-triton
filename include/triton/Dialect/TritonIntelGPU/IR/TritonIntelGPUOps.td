#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

def ResultsAreSharedEncoding: NativeOpTrait<"ResultsAreSharedEncoding">;

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TTIG_PrefetchCacheOp : TTIG_Op<"prefetch_cache",
                                  [
                                  //AttrSizedOperandSegments,
                                   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                  // MemoryEffects<[MemRead, MemWrite]>,
                                   //TypesMatchWith<"infer mask type from src type",
                                   //                "src", "mask", "getI1SameShape($_self)",
                                   //                "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
                                   //TypesMatchWith<"infer other type from src type",
                                   //                "src", "other", "getPointeeType($_self)",
                                   //                "($_op.getOperands().size() <= 4) || std::equal_to<>()">
                                                   ]> {
  let summary = "prefetch cache";

  let description = [{
    This is operation to prefetch cache
  }];

  let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr);

  let builders = [
      // A tensor of pointers
      // OpBuilder<(ins "Value":$ptr_johnlu)>,
  ];

  //let assemblyFormat = [{
  //  $src `,` $dst ``
  //  $index, $mask, $other
  //  attr-dict `:` type($src) `->` type($dst)
  //}];

  let extraClassDeclaration = [{
    //static DenseSet<unsigned> getEligibleLoadByteWidth(int computeCapability) {
    //  DenseSet<unsigned> validLoadBytes;
    //  if (computeCapability >= 80) {
    //    validLoadBytes = {4, 8, 16};
    //  }
    //  return validLoadBytes;
    //}
  }];

  let hasCustomAssemblyFormat = 1;
}

#endif
