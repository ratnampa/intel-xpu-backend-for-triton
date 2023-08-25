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

void storeXMXToShared(Value src, Value spirvSrc, Value smemBase, Location loc,
                      ConversionPatternRewriter &rewriter,
                      TritonGPUToSPIRVTypeConverter *typeConverter) {
  auto srcTy = src.getType().cast<RankedTensorType>();
  auto srcShape = srcTy.getShape();
  assert(srcShape.size() == 2 && "Unexpected rank of storeXMXToShared");
  auto srcXMXLayout =
      srcTy.getEncoding()
          .cast<mlir::triton::gpu::intel::IntelMmaEncodingAttr>();

  auto shapePerCTA = srcXMXLayout.getShapePerCTATile(srcShape);
  unsigned tilesRow = ceil<unsigned>(srcShape[0], shapePerCTA[0]);
  unsigned tilesCol = ceil<unsigned>(srcShape[1], shapePerCTA[1]);
  unsigned numElems = tilesRow * tilesCol;

  auto matTy = typeConverter->convertType(srcTy);
  auto elemTy = matTy.cast<spirv::StructType>()
                    .getElementType(0)
                    .cast<spirv::JointMatrixINTELType>();

  auto inVals = typeConverter->unpackLLElements(loc, spirvSrc, rewriter, srcTy);

  SmallVector<SmallVector<Value>> srcIndices;
  ArrayRef<Value> dstStrides;
  DenseMap<unsigned, Value> sharedPtrs;

  llvm::outs() << "johnlu smemBase:" << smemBase << "\n";
  llvm::outs().flush();
  for (unsigned elemIdx = 0; elemIdx < numElems; ++elemIdx) {
    Value offset = i32_val(0);
    // Extract multi dimensional index for current element
    Value currPtr = gep(srcTy.getElementType(), smemBase, offset);
    sharedPtrs[elemIdx] = offset;
  }

#if 0
DenseMap<unsigned, Value>
getSwizzledSharedPtrs(Location loc, unsigned inVec, RankedTensorType srcTy,
                      triton::gpu::SharedEncodingAttr resSharedLayout,
                      Type resElemTy, SharedMemoryObject smemObj,
                      ConversionPatternRewriter &rewriter,
                      SmallVectorImpl<Value> &offsetVals,
                      SmallVectorImpl<Value> &srcStrides) const {
  // This utililty computes the pointers for accessing the provided swizzled
  // shared memory layout `resSharedLayout`. More specifically, it computes,
  // for all indices (row, col) of `srcEncoding` such that idx % inVec = 0,
  // the pointer: ptr[(row, col)] = base + (rowOff * strides[ord[1]] +
  // colOff) where :
  //   phase = (row // perPhase) % maxPhase
  //   rowOff = row
  //   colOff = colOffSwizzled + colOffOrdered
  //     colOffSwizzled = ((col // outVec) ^ phase) * outVec
  //     colOffOrdered = (col % outVec) // minVec * minVec
  //
  // Note 1:
  // -------
  // Because swizzling happens at a granularity of outVec, we need to
  // decompose the offset into a swizzled factor and a non-swizzled
  // (ordered) factor
  //
  // Note 2:
  // -------
  // If we have x, y, z of the form:
  // x = 0b00000xxxx
  // y = 0byyyyy0000
  // z = 0b00000zzzz
  // then (x + y) XOR z = 0byyyyxxxx XOR 0b00000zzzz = (x XOR z) + y
  // This means that we can use some immediate offsets for shared memory
  // operations.
  auto dstPtrTy = ptr_ty(resElemTy, spirv::StorageClass::Workgroup);
  auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
  Value dstPtrBase = gep(dstPtrTy, smemObj.base, dstOffset);

  auto srcEncoding = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto srcShapePerCTA = triton::gpu::getShapePerCTA(srcTy);
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
  // swizzling params as described in TritonGPUAttrDefs.td
  unsigned outVec = resSharedLayout.getVec();
  unsigned perPhase = resSharedLayout.getPerPhase();
  unsigned maxPhase = resSharedLayout.getMaxPhase();
  // Order
  auto inOrder = triton::gpu::getOrder(srcEncoding);
  auto outOrder = triton::gpu::getOrder(resSharedLayout);
  // Tensor indices held by the current thread, as SPIRV values
  auto srcIndices = emitIndices(loc, rewriter, srcEncoding, srcTy, false);
  // Swizzling with leading offsets
  unsigned swizzlingByteWidth = 0;
  if (resSharedLayout.getHasLeadingOffset()) {
    if (perPhase == 4 && maxPhase == 2)
      swizzlingByteWidth = 32;
    else if (perPhase == 2 && maxPhase == 4)
      swizzlingByteWidth = 64;
    else if (perPhase == 1 && maxPhase == 8)
      swizzlingByteWidth = 128;
    else
      llvm::report_fatal_error("Unsupported shared layout.");
  }
  unsigned numElemsPerSwizzlingRow =
      swizzlingByteWidth * 8 / resElemTy.getIntOrFloatBitWidth();
  Value numElemsPerSwizzlingRowVal = i32_val(numElemsPerSwizzlingRow);
  unsigned leadingDimOffset =
      numElemsPerSwizzlingRow * srcShapePerCTA[outOrder[1]];
  Value leadingDimOffsetVal = i32_val(leadingDimOffset);
  // Return values
  DenseMap<unsigned, Value> ret;
  // cache for non-immediate offsets
  DenseMap<unsigned, Value> cacheCol, cacheRow;
  unsigned minVec = std::min(outVec, inVec);
  for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
    Value offset = i32_val(0);
    // Extract multi dimensional index for current element
    auto idx = srcIndices[elemIdx];
    Value idxCol = idx[outOrder[0]]; // contiguous dimension
    Value idxRow = idx[outOrder[1]]; // discontiguous dimension
    Value strideCol = srcStrides[outOrder[0]];
    Value strideRow = srcStrides[outOrder[1]];
    // compute phase = (row // perPhase) % maxPhase
    Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
    // extract dynamic/static offset for immediate offsetting
    unsigned immedateOffCol = 0;
    unsigned immedateOffRow = 0;
    if (leadingDimOffset) {
      // hopper
      offset =
          mul(udiv(idxCol, numElemsPerSwizzlingRowVal), leadingDimOffsetVal);
      // Shrink by swizzling blocks
      idxCol = urem(idxCol, numElemsPerSwizzlingRowVal);
      strideRow = numElemsPerSwizzlingRowVal;
    } else {
      if (auto add = dyn_cast_or_null<spirv::IAddOp>(idxCol.getDefiningOp()))
        if (auto _cst = dyn_cast_or_null<spirv::ConstantOp>(
                add.getOperand2().getDefiningOp())) {
          unsigned cst =
              _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
          unsigned key = cst % (outVec * maxPhase);
          cacheCol.insert({key, idxCol});
          idxCol = cacheCol[key];
          immedateOffCol = cst / (outVec * maxPhase) * (outVec * maxPhase);
        }
      if (auto add = dyn_cast_or_null<spirv::IAddOp>(idxRow.getDefiningOp()))
        if (auto _cst = dyn_cast_or_null<spirv::ConstantOp>(
                add.getOperand2().getDefiningOp())) {
          unsigned cst =
              _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
          unsigned key = cst % (perPhase * maxPhase);
          cacheRow.insert({key, idxRow});
          idxRow = cacheRow[key];
          immedateOffRow =
              cst / (perPhase * maxPhase) * (perPhase * maxPhase);
        }
    }
    // row offset is simply row index
    Value rowOff = mul(idxRow, strideRow);
    // because swizzling happens at a granularity of outVec, we need to
    // decompose the offset into a swizzled factor and a non-swizzled
    // (ordered) factor: colOffSwizzled = ((col // outVec) ^ phase) * outVec
    // colOffOrdered = (col % outVec) // minVec * minVec
    Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
    colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
    Value colOffOrdered = urem(idxCol, i32_val(outVec));
    colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
    colOffOrdered = mul(colOffOrdered, i32_val(minVec));
    Value colOff = add(colOffSwizzled, colOffOrdered);
    // compute non-immediate offset
    offset = add(offset, add(rowOff, mul(colOff, strideCol)));
    Value currPtr = gep(dstPtrTy, dstPtrBase, offset);
    // compute immediate offset
    Value immedateOff =
        add(mul(i32_val(immedateOffRow), srcStrides[outOrder[1]]),
            i32_val(immedateOffCol));
    ret[elemIdx] = gep(dstPtrTy, currPtr, immedateOff);
  }
  return ret;
}
#endif
  //  DenseMap<unsigned, Value> sharedPtrs =
  //      getSwizzledSharedPtrs(loc, inVec, srcTy, resSharedLayout, resElemTy,
  //                            smemObj, rewriter, offsetVals, srcStrides);

  //  auto dstStrides =
  //      getStridesFromShapeAndOrder(dstShapePerCTA, outOrd, loc, rewriter);

  auto ctx = rewriter.getContext();
  for (unsigned i = 0; i < numElems; ++i) {
    Value smemAddr = sharedPtrs[i];
    rewriter.create<spirv::INTELJointMatrixStoreOp>(
        loc, smemAddr, inVals[i], dstStrides[0],
        spirv::MatrixLayoutAttr::get(ctx, elemTy.getMatrixLayout()),
        spirv::ScopeAttr::get(ctx, elemTy.getScope()),
        spirv::MemoryAccessAttr::get(ctx, spirv::MemoryAccess::None),
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 64));
  }
}

Value storeArg(ConversionPatternRewriter &rewriter, Location loc, Value val,
               Value spirvVal, RankedTensorType sharedType, Value smemBase,
               TritonGPUToSPIRVTypeConverter *typeConverter, Value thread) {
  auto tensorTy = val.getType().cast<RankedTensorType>();

  int bitwidth = tensorTy.getElementTypeBitWidth();
  SharedEncodingAttr dstSharedLayout =
      sharedType.getEncoding().dyn_cast<SharedEncodingAttr>();
  auto mmaLayout =
      tensorTy.getEncoding().cast<triton::gpu::intel::IntelMmaEncodingAttr>();
  auto outOrd = dstSharedLayout.getOrder();
  auto dstShapePerCTA = triton::gpu::getShapePerCTA(sharedType);

  storeXMXToShared(val, spirvVal, smemBase, loc, rewriter, typeConverter);

  auto smemObj =
      SharedMemoryObject(smemBase, dstShapePerCTA, outOrd, loc, rewriter);
  auto retVal =
      ConvertTritonGPUOpToSPIRVPatternBase::getStructFromSharedMemoryObject(
          loc, smemObj, rewriter);
  return retVal;
#if 0

    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    auto dstShapePerCTA = triton::gpu::getShapePerCTA(dstTy);
    assert(srcShape.size() == 2 &&
           "Unexpected rank of ConvertLayout(blocked->shared)");
    auto srcLayout = srcTy.getEncoding();
    auto dstSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
    auto inOrd = getOrder(srcLayout);
    auto outOrd = dstSharedLayout.getOrder();
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy),
                            spirv::StorageClass::Workgroup);
#endif
#if 0
  int bitwidth = tensorTy.getElementTypeBitWidth();
  DotOperandEncodingAttr encoding =
      dotOpType.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  auto mmaLayout =
      encoding.getParent().cast<triton::gpu::intel::IntelMmaEncodingAttr>();

  SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                             tensorTy.getShape().end());

  ValueTable vals;
  int mmaInstrM = 8, mmaInstrN = 8, mmaInstrK = 4 * 64 / bitwidth;
  int matShapeM = 8, matShapeN = 8, matShapeK = 4 * 64 / bitwidth;

  auto numRep = getXMXRep(encoding, tensorTy.getShape(), bitwidth);
  int kWidth = encoding.getKWidth();

  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto order = triton::gpu::getOrder(mmaLayout);

  auto threadsPerWarp =
      product<unsigned>(triton::gpu::getThreadsPerWarp(mmaLayout));
  Value warp = udiv(thread, i32_val(threadsPerWarp));
  Value lane = urem(thread, i32_val(threadsPerWarp));

  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warp, warpsPerCTA, order);
  Value warpM = urem(multiDimWarpId[0], i32_val(shape[0] / mmaInstrM));
  Value warpN = urem(multiDimWarpId[1], i32_val(shape[1] / mmaInstrN));

  int warpsPerTile;
  if (isA)
    warpsPerTile = std::min<int>(warpsPerCTA[0], shape[0] / mmaInstrK);
  else
    warpsPerTile = std::min<int>(warpsPerCTA[1], shape[1] / mmaInstrK);

  std::function<void(int, int)> loadFn;
  if (isA)
    loadFn = getLoadMatrixFn(
        tensor, dotOpType, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/,
        1 /*kOrder*/, kWidth, {mmaInstrM, mmaInstrK} /*instrShape*/,
        {matShapeM, matShapeK} /*matShape*/, warpM /*warpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);
  else
    loadFn = getLoadMatrixFn(
        tensor, dotOpType, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/,
        0 /*kOrder*/, kWidth, {mmaInstrK, mmaInstrN} /*instrShape*/,
        {matShapeK, matShapeN} /*matShape*/, warpN /*warpId*/, lane /*laneId*/,
        vals /*vals*/, isA /*isA*/, typeConverter /* typeConverter */,
        rewriter /*rewriter*/, loc /*loc*/);

  // Perform loading.
  int numRepOuter = isA ? numRep[0] : numRep[1];
  int numRepK = isA ? numRep[1] : numRep[0];
  for (int m = 0; m < numRepOuter; ++m)
    for (int k = 0; k < numRepK; ++k)
      loadFn(m, k);

  // Format the values to LLVM::Struct to passing to mma codegen.
  return composeValuesToDotOperandLayoutStruct(vals, numRepOuter, numRepK,
                                               typeConverter, loc, rewriter);
#endif
}

namespace XMXToShared {
Value convertLayout(ConversionPatternRewriter &rewriter, Location loc,
                    Value val, Value spirvVal, RankedTensorType sharedType,
                    Value smemBase,
                    TritonGPUToSPIRVTypeConverter *typeConverter,
                    Value thread) {
  return storeArg(rewriter, loc, val, spirvVal, sharedType, smemBase,
                  typeConverter, thread);
}
} // namespace XMXToShared
