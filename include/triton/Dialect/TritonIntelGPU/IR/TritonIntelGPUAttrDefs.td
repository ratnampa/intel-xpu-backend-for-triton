#ifndef TRITON_INTEL_GPU_ATTRDEFS
#define TRITON_INTEL_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
//include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.td"

//===----------------------------------------------------------------------===//
// TritonIntelGPU Attribute Definitions
//===----------------------------------------------------------------------===//

class TritonIntelGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TritonIntelGPU_Dialect, name, traits, baseCppClass> {

  let description = [{
TritonGPU Tensors differ from usual tensors in that they contain a _layout_ attribute which determines
how the data should be partitioned across CUDA threads. Formally speaking, we define a layout as a function
\mathcal{L} that maps a multi-dimensional tensor index $i \in \mathbb{Z}^d$ to a set of integers T corresponding
to the indices of the CUDA threads allowed to access some data at index $i$.

For example, let us consider the layout function:
\mathcal{L}(0, 0) = {0, 4}
\mathcal{L}(0, 1) = {1, 5}
\mathcal{L}(1, 0) = {2, 6}
\mathcal{L}(1, 1) = {3, 7}

Then, attaching $\mathcal{L} to a tensor $T$ would mean that:
- T[0,0] is owned by both cuda thread 0 and 4
- T[0,1] is owned by both cuda thread 1 and 5
- T[1,0] is owned by both cuda thread 2 and 6
- T[1,1] is owned by both cuda thread 3 and 7

Right now, Triton implements two classes of layouts: shared, and distributed.
  }];

  code extraBaseClassDeclaration = [{
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
//    ::mlir::LogicalResult verifyLayoutForArg(::mlir::Operation* op, unsigned argNo) const;
  }];
}

//===----------------------------------------------------------------------===//
// Distributed Layout Encoding
//===----------------------------------------------------------------------===//

class IntelDistributedEncoding<string name, list<Trait> traits = []> : TritonIntelGPU_Attr<name, traits> {
  let description = [{
Distributed encodings have a layout function that is entirely characterized
by a d-dimensional tensor L. Note that L doesn't need to have the same shape
(or even the same rank) as the tensor it is encoding.

The layout function \mathcal{L} of this layout is then defined, for an
index `i` \in R^D, as follows:

\mathcal{L}(A)[i_d] = L[(i_d + k_d*A.shape[d]) % L.shape[d]] \forall k_d such as i_d + k_d*A.shape[d] < L.shape[d]

For example, for a tensor/layout pair
A = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

Then the data of A would be distributed as follow between the 16 CUDA threads:
L(A) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
  }];

  let extraClassDeclaration = extraBaseClassDeclaration;
}

//===----------------------------------------------------------------------===//
// Intel MMA Layout Encoding
//===----------------------------------------------------------------------===//
// TODO: MMAv1 and MMAv2 should be two instances of the same class

def IntelMmaEncodingAttr : IntelDistributedEncoding<"IntelMmaEncoding", [TritonGPUMMAAttrInterface]> {
  let mnemonic = "xmx";

  let description = [{
An encoding for tensors that have been produced by tensor cores.
It is characterized by two parameters:
- A 'versionMajor' which specifies the generation the tensor cores
whose output is being partitioned: 1 for first-gen tensor cores (Volta),
and 2 for second-gen tensor cores (Turing/Ampere).
- A 'versionMinor' which indicates the specific layout of a tensor core
generation, e.g. for Volta, there might be multiple kinds of layouts annotated
by 0,1,2 and so on.
- A `blockTileSize` to indicate how data should be
partitioned between warps.

// -------------------------------- version = 1 --------------------------- //

For first-gen tensor cores, the implicit warpTileSize is [16, 16].
Note: the layout is different from the recommended in PTX ISA
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.884 section, FP32 accumulator).

For example, when versionMinor=1, the matrix L corresponding to
blockTileSize=[32,16] is:

                               warp 0
--------------------------------/\-------------------------------
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]

                          warp 1 = warp0 + 32
--------------------------------/\-------------------------------
[ 32  32  34  34  40  40  42  42   32  32  34  34  40  40  42  42 ]
[ 33  33  35  35  41  41  43  43   33  33  35  35  41  41  43  43 ]
[ ............................................................... ]


// -------------------------------- version = 2 --------------------------- //

For second-gen tensor cores, the implicit warpTileSize is [16, 8].
Information about this layout can be found in the official PTX documentation
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.16816 section, FP32 accumulator).

For example, the matrix L corresponding to blockTileSize=[32,16] is:
                warp 0                          warp 2
-----------------/\-------------  ----------------/\-------------
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63

              warp 1                           warp 3
----------------/\------------- <MmaEncodingAt  ----------------/\-------------
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127

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

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    bool isVolta() const;
    bool isAmpere() const;
    SmallVector<unsigned> getSizePerThread() const;
    SmallVector<unsigned> getShapePerCTA(ArrayRef<int64_t> tensorShape) const;
  }];

  let hasCustomAssemblyFormat = 1;
}


#endif
