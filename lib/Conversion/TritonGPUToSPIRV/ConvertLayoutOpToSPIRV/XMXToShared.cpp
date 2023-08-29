#include "../ConvertLayoutOpToSPIRV.h"
#include "../Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

using namespace mlir;

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;
using ::mlir::spirv::delinearize;
using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::spirv::getStridesFromShapeAndOrder;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

extern Type getSharedMemPtrTy(Type argType);

void storeXMXToShared(Value src, Value spirvSrc, Value smemBase,
                      Value subGroupID, Location loc,
                      ConversionPatternRewriter &rewriter,
                      TritonGPUToSPIRVTypeConverter *typeConverter) {
  auto srcTy = src.getType().cast<RankedTensorType>();
  auto srcShape = srcTy.getShape();
  assert(srcShape.size() == 2 && "Unexpected rank of storeXMXToShared");
  auto srcXMXLayout =
      srcTy.getEncoding()
          .cast<mlir::triton::gpu::intel::IntelMmaEncodingAttr>();

  auto shapePerCTATile = srcXMXLayout.getShapePerCTATile(srcShape);
  auto order = getOrder(srcXMXLayout);
  unsigned tilesRow = ceil<unsigned>(srcShape[0], shapePerCTATile[0]);
  unsigned tilesCol = ceil<unsigned>(srcShape[1], shapePerCTATile[1]);
  unsigned numElems = tilesRow * tilesCol;

  auto matTy = typeConverter->convertType(srcTy);
  auto elemTy = matTy.cast<spirv::StructType>()
                    .getElementType(0)
                    .cast<spirv::JointMatrixINTELType>();

  auto inVals = typeConverter->unpackLLElements(loc, spirvSrc, rewriter, srcTy);

  // The stride is like the lda.
  assert(elemTy.getMatrixLayout() == spirv::MatrixLayout::RowMajor &&
         "unexpected c layout");
  Value stride = i32_val(srcShape[1]);

  DenseMap<unsigned, Value> sharedPtrs;

  auto multiDimSgIds =
      delinearize(rewriter, loc, subGroupID, shapePerCTATile, order);

  for (unsigned m = 0; m < tilesRow; ++m) {
    for (unsigned n = 0; n < tilesCol; ++n) {
      Value startM = mul(i32_val(tilesRow), i32_val(shapePerCTATile[0]));
      Value startN = mul(i32_val(tilesCol), i32_val(shapePerCTATile[1]));
      Value offsetM = add(startM, mul(multiDimSgIds[0], stride));
      Value offsetN = add(startN, multiDimSgIds[1]);
      Value tileBase = add(mul(startM, stride), startN);
      Value offsetWithinTile = add(offsetM, offsetN);
      // Extract multi dimensional index for current element
      Value currPtr = gep(smemBase, add(tileBase, offsetWithinTile));
      sharedPtrs[m * tilesCol + n] = currPtr;
    }
  }

  auto ctx = rewriter.getContext();
  for (unsigned i = 0; i < numElems; ++i) {
    Value smemAddr = sharedPtrs[i];
    //    rewriter.create<spirv::INTELJointMatrixStoreOp>(
    //        loc, smemAddr, inVals[i], stride,
    //        spirv::MatrixLayoutAttr::get(ctx, elemTy.getMatrixLayout()),
    //        spirv::ScopeAttr::get(ctx, elemTy.getScope()),
    //        spirv::MemoryAccessAttr::get(ctx, spirv::MemoryAccess::None),
    //        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 64));
  }
}

Value storeArg(ConversionPatternRewriter &rewriter, Location loc, Value val,
               Value spirvVal, RankedTensorType sharedType, Value smemBase,
               TritonGPUToSPIRVTypeConverter *typeConverter, Value subGroupID) {
  auto tensorTy = val.getType().cast<RankedTensorType>();
  SharedEncodingAttr dstSharedLayout =
      sharedType.getEncoding().dyn_cast<SharedEncodingAttr>();
  auto outOrd = dstSharedLayout.getOrder();
  auto dstShapePerCTA = triton::gpu::getShapePerCTA(sharedType);

  Type smemPtrTy = getSharedMemPtrTy(tensorTy.getElementType());
  smemBase = bitcast(smemBase, smemPtrTy);

  storeXMXToShared(val, spirvVal, smemBase, subGroupID, loc, rewriter,
                   typeConverter);

  auto smemObj =
      SharedMemoryObject(smemBase, dstShapePerCTA, outOrd, loc, rewriter);
  auto retVal =
      ConvertTritonGPUOpToSPIRVPatternBase::getStructFromSharedMemoryObject(
          loc, smemObj, rewriter);
  return retVal;
}

namespace XMXToShared {
Value convertLayout(ConversionPatternRewriter &rewriter, Location loc,
                    Value val, Value spirvVal, RankedTensorType sharedType,
                    Value smemBase,
                    TritonGPUToSPIRVTypeConverter *typeConverter,
                    Value subGroupID) {
  return storeArg(rewriter, loc, val, spirvVal, sharedType, smemBase,
                  typeConverter, subGroupID);
}
} // namespace XMXToShared
