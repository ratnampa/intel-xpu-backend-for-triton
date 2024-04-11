#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;


def TTIG_AllocOp : TTIG_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory allocation operation";

  let description = [{
    The `alloc` operation allocates a region of memory, as specified by its type.

    Example:

    ```mlir
    %0 = triton_intel_gpu.alloc() : <8x64xf32, 3>
    ```
  }];

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];

  let results = (outs TT_Ptr:$result);
}

def TTIG_PrefetchOp : TTIG_Op<"prefetch", []> {
  let summary = "Prefetch from a tensor pointer";

  let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr, TT_CacheModifierAttr:$cache,
                       TT_EvictionPolicyAttr:$evict, BoolAttr:$isVolatile);

  let results = (outs);

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];
}

def TTIG_Load2DOp : TTIG_Op<"load_2d", [
  SameLoadStoreOperandsAndResultShape,
  SameLoadStoreOperandsAndResultEncoding,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"infer ptr type from result type",
                 "result", "ptr", "$_self",
                 "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">
]> {
    let summary = "Intel load 2d block";

    let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr,
                         OptionalAttr<TT_PaddingOptionAttr>:$padding,
                         //DenseI32ArrayAttr:$strides, //only support contiguous 2d load.
                         TT_CacheModifierAttr:$cache,
                         TT_EvictionPolicyAttr:$evict,
                         BoolAttr:$isVolatile);

    let results = (outs TT_Type:$result);

    let builders = [
        // A tensor of pointers or a pointer to a scalar
        OpBuilder<(ins "Value":$ptr, "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,

        OpBuilder<(ins "Value":$ptr, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor pointer with boundary check and padding
        //OpBuilder<(ins "Value":$ptr, "ArrayRef<int32_t>":$boundaryCheck,
        //               "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
        //               "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor of pointers or a pointer to a scalar with mask
        //OpBuilder<(ins "Value":$ptr, "Value":$mask, "triton::CacheModifier":$cache,
        //               "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor of pointers or a pointer to a scalar with mask and other
        //OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other, "triton::CacheModifier":$cache,
        //               "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A utility function to build the operation with all attributes
        //OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other,
        //               "std::optional<ArrayRef<int32_t>>":$boundaryCheck,
        //               "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
        //               "triton::EvictionPolicy":$evict, "bool":$isVolatile)>
    ];

    let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";

    let hasCanonicalizer = 1;
}

def TTIG_Store2DOp : TTIG_Op<"store_2d", [
  SameLoadStoreOperandsShape,
  SameLoadStoreOperandsEncoding,
  MemoryEffects<[MemWrite<GlobalMemory>]>,
  TypesMatchWith<"infer ptr type from value type",
                 "value", "ptr", "$_self",
                 "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">
]> {
    let summary = "Intel store 2d block";

    let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr, TT_Type:$value,
                         DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
                         DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict);

    let builders = [
        // A tensor of pointers or a pointer to a scalar
        // OpBuilder<(ins "Value":$ptr, "Value":$value, "triton::CacheModifier":$cache, "triton::EvictionPolicy":$evict)>
    ];

    let assemblyFormat = "operands attr-dict `:` type($ptr) `->` type($value)";

    let hasCanonicalizer = 1;
}

#endif
