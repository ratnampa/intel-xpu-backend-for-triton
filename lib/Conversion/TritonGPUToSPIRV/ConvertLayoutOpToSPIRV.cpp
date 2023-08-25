#include "ConvertLayoutOpToSPIRV.h"
#include "Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

using ::mlir::spirv::delinearize;
using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::spirv::getStridesFromShapeAndOrder;
using ::mlir::spirv::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Forward declarations
namespace SharedToDotOperandXMX {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, RankedTensorType dotOpTypee,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToSPIRVTypeConverter *typeConverter, Value thread);
}

namespace SharedToDotOperandFMA {
Value convertLayout(int opIdx, Value B, Value llB, BlockedEncodingAttr dLayout,
                    Value thread, Location loc,
                    TritonGPUToSPIRVTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter);
}

namespace XMXToShared {
Value convertLayout(ConversionPatternRewriter &rewriter, Location loc,
                    Value val, Value spirvVal, RankedTensorType sharedType,
                    Value smemBase,
                    TritonGPUToSPIRVTypeConverter *typeConverter, Value thread);
}

struct ConvertLayoutOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    // forwarding on mma->shared, lower distributed->shared otherwise
    if (srcLayout.isa<triton::gpu::intel::IntelMmaEncodingAttr>() &&
        dstLayout.isa<SharedEncodingAttr>()) {
      return lowerMMAToShared(op, adaptor, rewriter);
    }
    if (isaDistributedLayout(srcLayout) &&
        dstLayout.isa<SharedEncodingAttr>()) {
      return lowerDistributedToShared(op, adaptor, rewriter);
    }
    if (srcLayout.isa<SharedEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerSharedToDotOperand(op, adaptor, rewriter);
    }
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (srcLayout.isa<triton::gpu::intel::IntelMmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }
    if (srcLayout.isa<SharedEncodingAttr>() &&
        isaDistributedLayout(dstLayout)) {
      return lowerSharedToDistributed(op, adaptor, rewriter);
    }
    // TODO: to be implemented
    llvm_unreachable("unsupported layout conversion");
    return failure();
  }

private:
  SmallVector<Value>
  getMultiDimOffset(Attribute layout, Location loc,
                    ConversionPatternRewriter &rewriter, unsigned elemId,
                    RankedTensorType type,
                    ArrayRef<unsigned> multiDimCTAInRepId,
                    ArrayRef<unsigned> shapePerCTATile) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
      auto multiDimOffsetFirstElem =
          emitBaseIndexForLayout(loc, rewriter, blockedLayout, type, false);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] =
            add(multiDimOffsetFirstElem[d],
                i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentSizePerThread = getSizePerThread(parentEncoding);
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                            parentEncoding);
      auto offsets = emitOffsetForLayout(layout, type);
      auto parentOffset = emitOffsetForLayout(parentEncoding, parentTy);
      SmallVector<int> idxs;
      for (SmallVector<unsigned> off : offsets) {
        off.insert(off.begin() + dim, 0);
        auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
        idxs.push_back(std::distance(parentOffset.begin(), it));
      }
      auto multiDimOffsetParent = getMultiDimOffset(
          parentEncoding, loc, rewriter, idxs[elemId], parentTy,
          sliceLayout.paddedShape(multiDimCTAInRepId),
          sliceLayout.paddedShape(shapePerCTATile));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    if (auto mmaLayout =
            layout.dyn_cast<triton::gpu::intel::IntelMmaEncodingAttr>()) {
      SmallVector<Value> mmaColIdx(4);
      SmallVector<Value> mmaRowIdx(2);
      Value threadId = getThreadId(rewriter, loc);
      Value warpSize = i32_val(32);
      Value laneId = urem(threadId, warpSize);
      Value warpId = udiv(threadId, warpSize);
      // TODO: fix the bug in MMAEncodingAttr document
      auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
      auto order = triton::gpu::getOrder(mmaLayout);
      SmallVector<Value> multiDimWarpId =
          delinearize(rewriter, loc, warpId, warpsPerCTA, order);
      Value _1 = i32_val(1);
      Value _2 = i32_val(2);
      Value _4 = i32_val(4);
      Value _8 = i32_val(8);
      Value _16 = i32_val(16);
      multiDimWarpId[0] = urem(multiDimWarpId[0], i32_val(shape[0] / 16));
      multiDimWarpId[1] = urem(multiDimWarpId[1], i32_val(shape[1] / 8));
      Value mmaGrpId = udiv(laneId, _4);
      Value mmaGrpIdP8 = add(mmaGrpId, _8);
      Value mmaThreadIdInGrp = urem(laneId, _4);
      Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
      Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
      Value rowWarpOffset = mul(multiDimWarpId[0], _16);
      mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
      mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
      Value colWarpOffset = mul(multiDimWarpId[1], _8);
      mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
      mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);

      assert(rank == 2);
      SmallVector<Value> multiDimOffset(rank);
      multiDimOffset[0] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
      multiDimOffset[1] = elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
      multiDimOffset[0] = add(multiDimOffset[0], i32_val(multiDimCTAInRepId[0] *
                                                         shapePerCTATile[0]));
      multiDimOffset[1] = add(multiDimOffset[1], i32_val(multiDimCTAInRepId[1] *
                                                         shapePerCTATile[1]));
      return multiDimOffset;
    }
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  SmallVector<Value>
  getWrappedMultiDimOffset(ConversionPatternRewriter &rewriter, Location loc,
                           ArrayRef<Value> multiDimOffset,
                           ArrayRef<unsigned> shape,
                           SmallVector<unsigned> shapePerCTATile,
                           SmallVector<int64_t> shapePerCTA) const {
    unsigned rank = shape.size();
    SmallVector<Value> multiDimOffsetWrapped(rank);
    for (unsigned d = 0; d < rank; ++d) {
      if (shapePerCTATile[d] > shapePerCTA[d])
        multiDimOffsetWrapped[d] = urem(multiDimOffset[d], i32_val(shape[d]));
      else
        multiDimOffsetWrapped[d] = multiDimOffset[d];
    }
    return multiDimOffsetWrapped;
  }

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> origRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {

    llvm::outs() << "johnlu total extracted value size: " << vals.size()
                 << "\n";

    llvm::outs() << "johnlu numCTAsEachRep: ";
    for (auto &i : numCTAsEachRep) {
      llvm::outs() << " " << i;
    }
    llvm::outs() << "\n";
    llvm::outs() << "johnlu multiDimRepId: ";
    for (auto &i : multiDimRepId) {
      llvm::outs() << " " << i;
    }
    llvm::outs() << "\n";
    llvm::outs().flush();
    llvm::outs() << "johnlu paddedRepShape: ";
    for (auto &i : paddedRepShape) {
      llvm::outs() << " " << i;
    }
    llvm::outs() << "\n";
    llvm::outs().flush();

    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTATiles(rank);
    auto shapePerCTATile = getShapePerCTATile(layout);
    auto shapePerCTA = getShapePerCTA(layout, type.getShape());
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTATiles[d] = ceil<unsigned>(shapePerCTA[d], shapePerCTATile[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = elemTy.isa<triton::PointerType>();
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTATiles, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
            rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile,
            shapePerCTA);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
        auto elemPtrTy = ptr_ty(llvmElemTy, spirv::StorageClass::Workgroup);
        Value ptr = gep(elemPtrTy, bitcast(smemBase, elemPtrTy), offset);
        if (vec == 1) {
          if (stNotRd) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread];
            if (isInt1) {
              // spriv::UConvert doesn't support i1
              currVal = select(currVal, int_val(8, 1), int_val(8, 0));
            } else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            store(currVal, ptr);
          } else {
            Value currVal = load(ptr);
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<spirv::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread] = currVal;
          }
        } else {
          auto vecTy = vec_ty(llvmElemTy, vec);
          ptr = bitcast(ptr, ptr_ty(vecTy, spirv::StorageClass::Workgroup));
          if (stNotRd) {
            Value valVec = undef(vecTy);
            for (unsigned v = 0; v < vec; ++v) {
              auto currVal =
                  vals[elemId + linearCTAId * accumSizePerThread + v];
              if (isInt1) {
                // spriv::UConvert doesn't support i1
                currVal = select(currVal, int_val(8, 1), int_val(8, 0));
              } else if (isPtr)
                currVal = ptrtoint(llvmElemTy, currVal);
              valVec = insert_element(vecTy, valVec, currVal, idx_val(v));
            }
            store(valVec, ptr);
          } else {
            Value valVec = load(ptr);
            for (unsigned v = 0; v < vec; ++v) {
              Value currVal = extract_element(llvmElemTy, valVec, idx_val(v));
              if (isInt1)
                currVal = icmp_ne(
                    currVal, rewriter.create<spirv::ConstantOp>(
                                 loc, i8_ty, rewriter.getI8IntegerAttr(0)));
              else if (isPtr)
                currVal = inttoptr(llvmElemTyOrig, currVal);
              vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
            }
          }
        }
      }
    }
  }

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(llvmElemTy, spirv::StorageClass::Workgroup);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout, shape);
    auto shapePerCTA = getShapePerCTA(srcLayout, shape);

    if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
      //      isSrcMmaV1 = sliceLayout.getParent().isa<MmaEncodingAttr>() &&
      //                   sliceLayout.getParent().cast<MmaEncodingAttr>().isVolta();
    }
    if (auto mmaLayout = dstLayout.dyn_cast<MmaEncodingAttr>()) {
      //      isDstMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = dstLayout.dyn_cast<SliceEncodingAttr>()) {
      //      isDstMmaV1 = sliceLayout.getParent().isa<MmaEncodingAttr>() &&
      //                   sliceLayout.getParent().cast<MmaEncodingAttr>().isVolta();
    }

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA =
          std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA =
          std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }
    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals = getTypeConverter()->unpackLLElements(loc, adaptor.getSrc(),
                                                     rewriter, srcTy);
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto origRepShape = getRepShapeForCvtLayout(op);
    auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0)
        barrier();
      if (srcLayout.isa<BlockedEncodingAttr>() ||
          srcLayout.isa<SliceEncodingAttr>() ||
          srcLayout.isa<triton::gpu::intel::IntelMmaEncodingAttr>()) {
        if (auto xmxLayout =
                srcLayout
                    .dyn_cast<triton::gpu::intel::IntelMmaEncodingAttr>()) {
          llvm::outs() << "johnlu op: " << op << "\n";
          llvm::outs() << "johnlu source operands: " << adaptor.getSrc()
                       << "\n";
          llvm::outs().flush();
          SmallVector<Value> temp;
          for (auto &val : vals) {
            for (int i = 0; i < 2; i++) {
              auto scalar = rewriter.create<spirv::VectorExtractDynamicOp>(
                  loc, llvmElemTy, val, i32_val(i));
              temp.push_back(scalar);
            }
          }
          vals = temp;
        }
          processReplica(loc, rewriter, /*stNotRd*/ true, srcTy,
                         inNumCTAsEachRep, multiDimRepId, inVec, paddedRepShape,
                         origRepShape, outOrd, vals, smemBase);
      } else {
        assert(0 && "ConvertLayout with input layout not implemented");
        return failure();
      }

      barrier();
      if (dstLayout.isa<BlockedEncodingAttr>() ||
          dstLayout.isa<SliceEncodingAttr>() ||
          dstLayout.isa<MmaEncodingAttr>()) {
//        if (isDstMmaV1)
//          processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ false, dstTy,
//                                 multiDimRepId, outVec, paddedRepShape, outOrd,
//                                 outVals, smemBase, shape, /*isDestMma=*/true);
//        else
          processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                         outNumCTAsEachRep, multiDimRepId, outVec,
                         paddedRepShape, origRepShape, outOrd, outVals,
                         smemBase);
      } else {
        assert(0 && "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    Value result =
        getTypeConverter()->packLLElements(loc, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  LogicalResult
  lowerSharedToDistributed(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() == 2 &&
           "Unexpected rank of ConvertLayout(shared->blocked)");
    auto srcSharedLayout = srcTy.getEncoding().cast<SharedEncodingAttr>();
    auto dstLayout = dstTy.getEncoding();
    auto inOrd = getOrder(srcSharedLayout);

    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), rewriter);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());

    auto srcStrides =
        getStridesFromShapeAndOrder(srcShape, inOrd, loc, rewriter);
    auto dstIndices = emitIndices(loc, rewriter, dstLayout, dstTy);

    SmallVector<Value> outVals = loadSharedToDistributed(
        dst, dstIndices, src, smemObj, elemTy, loc, rewriter);

    Value result =
        getTypeConverter()->packLLElements(loc, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  // work around for joint matrix
  LogicalResult lowerMMAToShared(triton::gpu::ConvertLayoutOp op,
                                 OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    Value smemBase = getSharedMemoryBase(loc, rewriter, dst);
    auto dstTy = dst.getType().cast<RankedTensorType>();

    auto thread = getThreadId(rewriter, loc);
    Value res =
        XMXToShared::convertLayout(rewriter, loc, src, adaptor.getSrc(), dstTy,
                                   smemBase, getTypeConverter(), thread);

    rewriter.replaceOp(op, res);
    return success();
  }

  // blocked -> shared.
  // Swizzling in shared memory to avoid bank conflict. Normally used for
  // A/B operands of dots.
  LogicalResult
  lowerDistributedToShared(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShapePerCTA = triton::gpu::getShapePerCTA(dstTy);
    assert(srcShape.size() == 2 &&
           "Unexpected rank of ConvertLayout(blocked->shared)");
    auto srcLayout = srcTy.getEncoding();
    auto dstSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
    auto inOrd = getOrder(srcLayout);
    auto outOrd = dstSharedLayout.getOrder();
    Value smemBase = getSharedMemoryBase(loc, rewriter, dst);
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy),
                            spirv::StorageClass::Workgroup);
    smemBase = bitcast(smemBase, elemPtrTy);

    auto dstStrides =
        getStridesFromShapeAndOrder(dstShapePerCTA, outOrd, loc, rewriter);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy, false);
    storeDistributedToShared(src, adaptor.getSrc(), dstStrides, srcIndices, dst,
                             smemBase, elemTy, loc, rewriter);
    auto smemObj =
        SharedMemoryObject(smemBase, dstShapePerCTA, outOrd, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

  // shared -> mma_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = dst.getType().cast<RankedTensorType>();
    auto srcTensorTy = src.getType().cast<RankedTensorType>();
    auto dotOperandLayout =
        dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto sharedLayout = srcTensorTy.getEncoding().cast<SharedEncodingAttr>();

    bool isOuter{};
    int K{};
    if (dotOperandLayout.getOpIdx() == 0) // $a
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[1]];
    isOuter = K == 1;

    Value res;
    if (auto mmaLayout =
            dotOperandLayout.getParent()
                .dyn_cast_or_null<triton::gpu::intel::IntelMmaEncodingAttr>()) {
      res = lowerSharedToDotOperandMMA(op, adaptor, rewriter, mmaLayout,
                                       dstTensorTy, isOuter);
    } else if (auto blockedLayout =
                   dotOperandLayout.getParent()
                       .dyn_cast_or_null<BlockedEncodingAttr>()) {
      auto dotOpLayout =
          dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
      auto thread = getThreadId(rewriter, loc);
      res = SharedToDotOperandFMA::convertLayout(
          dotOpLayout.getOpIdx(), src, adaptor.getSrc(), blockedLayout, thread,
          loc, getTypeConverter(), rewriter);
    } else {
      assert(false && "Unsupported dot operand layout found");
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    assert(0 && "no mma support yet");
  }

  // shared -> dot_operand if the result layout is mma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter,
      const triton::gpu::intel::IntelMmaEncodingAttr &mmaLayout,
      const RankedTensorType &dotOpType, bool isOuter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();

    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), rewriter);
    Value res;

    DotOperandEncodingAttr dotOperandLayout =
        dotOpType.getEncoding().dyn_cast<DotOperandEncodingAttr>();
    res = SharedToDotOperandXMX::convertLayout(
        dotOperandLayout.getOpIdx(), rewriter, loc, src, dotOpType, smemObj,
        getTypeConverter(), tid_val());

    return res;
  }
}; // namespace triton::gpu::ConvertLayoutOp

void populateConvertLayoutOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpSPIRVConversion>(
      typeConverter, context, allocation, indexCacheInfo, benefit);
}
