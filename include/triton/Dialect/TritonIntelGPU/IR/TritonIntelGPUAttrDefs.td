#ifndef TRITON_INTEL_GPU_ATTRDEFS
#define TRITON_INTEL_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
//include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.td"

//===----------------------------------------------------------------------===//
// Intel MMA Layout Encoding
//===----------------------------------------------------------------------===//
// TODO: MMAv1 and MMAv2 should be two instances of the same class

def IntelMmaEncodingAttr : DistributedEncoding<"IntelMmaEncoding", [TritonGPUMMAAttrInterface], TritonIntelGPU_Dialect> {
  let mnemonic = "xmx";

  let description = [{

}];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$shapeA,
    ArrayRefParameter<"unsigned">:$shapeB,
    ArrayRefParameter<"unsigned">:$shapeC,
    ArrayRefParameter<"unsigned">:$packedA,
    ArrayRefParameter<"unsigned">:$packedB,
    ArrayRefParameter<"unsigned">:$packedC,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    "unsigned":$threadsPerWarp
  );

  let builders = [

//    AttrBuilder<(ins "ArrayRef<unsigned>":$shapeA,
//                     "ArrayRef<unsigned>":$shapeB,
//                     "ArrayRef<unsigned>":$shapeC,
//                     "ArrayRef<unsigned>":$packedA,
//                     "ArrayRef<unsigned>":$packedB,
//                     "ArrayRef<unsigned>":$packedC,
//                     "ArrayRef<unsigned>":$warpsPerCTA,
//                     "int":$threadsPerWarp), [{
//      return $_get(context, shapeA, shapeB, shapeC, packedA, packedB, packedC, warpsPerCTA);
//    }]>
  ];

  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool isVolta() const;
    bool isAmpere() const;
    SmallVector<unsigned> getSizePerThread() const;
    SmallVector<unsigned> getShapePerCTATile(ArrayRef<int64_t> tensorShape) const;
    unsigned getTotalElemsPerThreadForOperands(ArrayRef<int64_t> shape, Type eltTy, int opIdx) const;

    SmallVector<int64_t> getXMXRep(ArrayRef<int64_t> shape, int opIdx) const;

    Attribute getCTALayout() const;
  }];

  let hasCustomAssemblyFormat = 1;
}


#endif
