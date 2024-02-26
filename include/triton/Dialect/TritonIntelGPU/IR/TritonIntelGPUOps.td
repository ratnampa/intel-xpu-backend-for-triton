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
  let summary = "prefetch data to L3 cache";

  let description = [{
    This is operation to prefetch cache
  }];

  let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr);

  let builders = [
      // A tensor of pointers
      // OpBuilder<(ins "Value":$ptr_johnlu)>,
  ];

  let assemblyFormat = "$ptr attr-dict `:` type($ptr)";

  let extraClassDeclaration = [{
  }];

  //let hasCustomAssemblyFormat = 1;
}

def TTIG_Load2DOp : TTIG_Op<"load_2d",
                      [SameLoadStoreOperandsAndResultShape,
                       SameLoadStoreOperandsAndResultEncoding,
                       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                       TypesMatchWith<"infer ptr type from result type",
                                      "result", "ptr", "$_self",
                                      "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">]> {
    let summary = "";

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

    // Format: `tt.load operands attrs : optional(type(ptr)) -> type(result)`
    let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";

    let hasCanonicalizer = 1;
}

#endif
