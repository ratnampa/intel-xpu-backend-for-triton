#include "../ConvertLayoutOpToSPIRV.h"
#include "../Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#define DEBUG_PRINT 0

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

// Data loader for XMX instruction.
class JointMatrixMatmulLoader {
public:
  JointMatrixMatmulLoader(ArrayRef<uint32_t> order, int warpsPerTile,
                          uint32_t kDim, int repeatCount, int systolicDepth,
                          int exeCutionSize, int opsPerChan, int threadsPerWarp,
                          ArrayRef<Value> smemStrides, ArrayRef<int> instrShape,
                          int perPhase, int maxPhase, int vec, int elemBytes,
                          ConversionPatternRewriter &rewriter,
                          TritonGPUToSPIRVTypeConverter *typeConverter,
                          const Location &loc);

  // lane = thread % 32
  // warp = (thread/32)
  SmallVector<Value> computeOffsets(Value warp, Value lane,
                                    Value cSwizzleOffset) {
    return computeLdsMatOffs(warp, lane, cSwizzleOffset);
  }

#if 0
  // Compute the offset to the matrix this thread(indexed by warpOff and lane)
  // mapped to.
  SmallVector<Value> computeLdmatrixMatOffs(Value warpId, Value lane,
                                            Value cSwizzleOffset);
#endif
  // compute matrix loads offset.
  SmallVector<Value> computeLdsMatOffs(Value warpOff, Value lane,
                                       Value cSwizzleOffset);

  // Load the matrix value.
  Value operator()(int mat0, int mat1, ArrayRef<Value> ptrs,
                   ArrayRef<Value> offs, Value subGroupID, Value lane,
                   Type matTy, Value cSwizzleOffset) const;

  int getNumPtrs() const { return numPtrs; }

private:
  // XMX instruction meta.
  int repeatCount;
  int systolicDepth;
  int exeCutionSize;
  int opsPerChan;
  int threadsPerWarp;

  // Share local memory meta.
  SmallVector<uint32_t> order;
  SmallVector<int> instrShape;
  SmallVector<Value> smemStrides;
  int vec;
  int perPhase;
  int maxPhase;
  int elemBytes;

  int kDim;
  bool needTrans;
  Value repNonKDimStride;
  Value repKDimStride;
  // Stride in number of matrices to increment on non-k dim across warps
  Value warpMatStride;
  int numPtrs;

  // code gen struct.
  ConversionPatternRewriter &rewriter;
  const Location &loc;
  MLIRContext *ctx{};
};

#if 0
SmallVector<Value>
JointMatrixMatmulLoader::computeLdmatrixMatOffs(Value warpId, Value lane,
                                                Value cSwizzleOffset) {
  // 4x4 matrices
  Value rowInMat = urem(lane, i32_val(8)); // row in the 8x8 matrix
  Value matIndex =
      udiv(lane, i32_val(8)); // linear index of the matrix in the 2x2 matrices

  // Decompose matIndex => s_0, s_1, that is the coordinate in 2x2 matrices in a
  // warp
  Value s0 = urem(matIndex, i32_val(2));
  Value s1 = udiv(matIndex, i32_val(2));

  // We use different orders for a and b for better performance.
  Value kMatArr = kOrder == 1 ? s1 : s0;  // index of matrix on the k dim
  Value nkMatArr = kOrder == 1 ? s0 : s1; // index of matrix on the non-k dim

  // Matrix coordinates inside a CTA,
  // the matrix layout is [2warpsPerTile[0], 2] for A and [2, 2warpsPerTile[1]]
  // for B. e.g., Setting warpsPerTile=4, the data layout for A(kOrder=1) is
  //   |0 0|  -> 0,1,2,3 are the warpids
  //   |0 0|
  //   |1 1|
  //   |1 1|
  //   |2 2|
  //   |2 2|
  //   |3 3|
  //   |3 3|
  //
  // for B(kOrder=0) is
  //   |0 1 2 3 0 1 2 3| -> 0,1,2,3 are the warpids
  //   |0 1 2 3 0 1 2 3|
  // Note, for each warp, it handles a 2x2 matrices, that is the coordinate
  // address (s0,s1) annotates.

  Value matOff[2];
  matOff[kOrder ^ 1] = add(
      mul(warpId, i32_val(warpMatOffset)), // warp offset (kOrder=1)
      mul(nkMatArr,
          i32_val(inWarpMatOffset))); // matrix offset inside a warp (kOrder=1)
  matOff[kOrder] = kMatArr;

  // Physical offset (before swizzling)
  Value contiguousMatIndex = matOff[order[0]];
  Value stridedMatIndex = matOff[order[1]];
  // Add the offset of the slice
  Value contiguousSliceMatOffset =
      udiv(cSwizzleOffset, i32_val(contiguousMatShape));

  SmallVector<Value> offs(1 /*numPtrs*/);
  Value phase = urem(udiv(rowInMat, i32_val(perPhase)), i32_val(maxPhase));
  // To prevent out-of-bound access of B when warpsPerTile * 16 > tile_size.
  // In such a case, we need to wrap around the offset of B.
  // |0 1 2 3 0 1 2 3| -> | 0(0) 1(1) 2(2) 3(3) |
  // |0 1 2 3 0 1 2 3|    | 0(0) 1(1) 2(2) 3(3) |
  //          ~~~~~~~ out-of-bound access

  Value rowOffset =
      urem(add(rowInMat, mul(stridedMatIndex, i32_val(stridedMatShape))),
           i32_val(tileShape[order[1]]));
  auto contiguousTileNumMats = tileShape[order[0]] / matShape[order[0]];

  for (int i = 0; i < 1 /*numPtrs*/; ++i) {
    Value contiguousIndex =
        add(contiguousMatIndex, i32_val(i * contiguousLoadMatOffset));
    if (warpsPerCTA[order[0]] > contiguousTileNumMats ||
        contiguousTileNumMats % warpsPerCTA[order[0]] != 0)
      contiguousIndex = urem(contiguousIndex, i32_val(contiguousTileNumMats));
    contiguousIndex = add(contiguousIndex, contiguousSliceMatOffset);
    Value contiguousIndexSwizzled = xor_(contiguousIndex, phase);
    offs[i] = add(mul(contiguousIndexSwizzled, i32_val(contiguousMatShape)),
                  mul(rowOffset, stridedSmemOffset));
  }

  return offs;
}
#endif
// clang-format off
// Value layout example for warp size 32.
// For A operand:
//                                   systolic depth = 8
// <------------------------------------------------------------------------------------------------->
// opsPerChan
// <--------->
// t0  ...  t0   t1  ... t1   t2  ... t2  t3  ... t3  t4  ... t4   t5  ... t5  t6  ... t6  t7  ... t7    ^
// t8  ...  t8   t9  ... t9   t10 ... t10 t11 ... t11 t12 ... t12  t13 ... t13 t14 ... t14 t15 ... t15   |
// t16 ...  t16  t17 ... t17  t18 ... t18 t19 ... t19 t20 ... t20  t21 ... t21 t22 ... t22 t23 ... t23   |
// t24 ...  t24  t25 ... t25  t26 ... t26 t27 ... t27 t28 ... t28  t29 ... t29 t30 ... t30 t31 ... t31   | repeat count <= 8
// t0  ...  t0   t1  ... t1   t2  ... t2  t3  ... t3  t4  ... t4   t5  ... t5  t6  ... t6  t7  ... t7    |
// t8  ...  t8   t9  ... t9   t10 ... t10 t11 ... t11 t12 ... t12  t13 ... t13 t14 ... t14 t15 ... t15   |
// t16 ...  t16  t17 ... t17  t18 ... t18 t19 ... t19 t20 ... t20  t21 ... t21 t22 ... t22 t23 ... t23   |
// t24 ...  t24  t25 ... t25  t26 ... t26 t27 ... t27 t28 ... t28  t29 ... t29 t30 ... t30 t31 ... t31   v
//
// For B operand:
//               execution size = 16
// <------------------------------------------------------------->
// t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15     ^             ^
// .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .       | opsPerChan  |
// t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15     v             |
// t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |
// .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
// t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |  systolic depth = 8
// t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15                   |
// .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
// t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15                   |
// t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |
// .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
// t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   v
//
// This pattern repeats every warpsPerTile[0] (resp. warpsPerTile[1]) blocks
// along the row (resp. col) dimension.
// clang-format on
SmallVector<Value>
JointMatrixMatmulLoader::computeLdsMatOffs(Value warp, Value lane,
                                           Value cSwizzleOffset) {
  SmallVector<Value> offs(numPtrs);

  int repRowsPerInst;
  int rowsPerWarp;
  Value laneRowIndex;
  Value laneColIndex;
  if (kDim == 1) /*A*/ {
    rowsPerWarp = threadsPerWarp / systolicDepth;
    repRowsPerInst = repeatCount / rowsPerWarp;
    laneRowIndex = udiv(lane, i32_val(systolicDepth));
    laneColIndex = urem(lane, i32_val(systolicDepth));
    laneColIndex = mul(laneColIndex, i32_val(opsPerChan));
  } else /*B*/ {
    rowsPerWarp = threadsPerWarp / exeCutionSize;
    repRowsPerInst = systolicDepth / rowsPerWarp;
    rowsPerWarp = rowsPerWarp * opsPerChan;
    laneRowIndex = udiv(lane, i32_val(exeCutionSize));
    laneRowIndex = mul(laneRowIndex, i32_val(opsPerChan));
    laneColIndex = urem(lane, i32_val(exeCutionSize));
  }

  // outer index offset
  Value iOff = mul(warp, warpMatStride);

  int index = 0;
  Value iBase;
  Value jBase;
  for (int rep = 0; rep < repRowsPerInst; ++rep) {
    Value repRowIndex = mul(i32_val(rep), i32_val(rowsPerWarp));
    for (int opsIdx = 0; opsIdx < opsPerChan; ++opsIdx) {
      // inner index base
      jBase = laneColIndex;
      // outer index base
      iBase = add(repRowIndex, laneRowIndex);
      if (kDim == 1) /*A*/ {
        jBase = add(jBase, i32_val(opsIdx));
      } else /*B*/ {
        iBase = add(iBase, i32_val(opsIdx));
      }
#if 0
      // swizzle
      if (!needTrans) {
        Value phase = urem(udiv(iBase, i32_val(perPhase)), i32_val(maxPhase));
        jOff = add(jOff, udiv(cSwizzleOffset, i32_val(quadWidth)));
        jOff = xor_(jOff, phase);
      } else {
        Value phase = urem(udiv(jBase, i32_val(perPhase)), i32_val(maxPhase));
        iOff = add(iOff, udiv(cSwizzleOffset, i32_val(quadHeight)));
        iOff = xor_(iOff, phase);
      }
      // To prevent out-of-bound access when tile is too small.
      Value i = add(iBase, mul(iOff, i32_val(quadHeight)));
      Value j = add(jBase, mul(jOff, i32_val(quadWidth)));
      // wrap around the bounds
      // i = urem(i, i32_val(cTileShape));
      // j = urem(j, i32_val(sTileShape));
      if (needTrans) {
        offs[idx] = add(i, mul(j, stridedSmemOffset));
      } else {
        offs[idx] = add(mul(i, stridedSmemOffset), j);
      }
#endif
      // inner index offset
      Value jOff = i32_val(0);
      // swizzle: col_swizzled = (col / vec) ^ phase * vec
      Value phase = urem(udiv(iBase, i32_val(perPhase)), i32_val(maxPhase));
      jOff = add(jOff, udiv(cSwizzleOffset, i32_val(vec)));
      jOff = mul(xor_(jOff, phase), i32_val(vec));

      Value i = add(mul(iBase, smemStrides[0]), iOff);
      Value j = add(mul(jBase, smemStrides[1]), jOff);

      offs[index++] = add(i, j);
    }
  }

  return offs;
}

Value JointMatrixMatmulLoader::operator()(
    int repOutter, int repInner, ArrayRef<Value> ptrs, ArrayRef<Value> offs,
    Value subGroupID, Value lane, Type matTy, Value cSwizzleOffset) const {
  // The struct should have exactly the same element types.
  auto structTy = matTy.cast<spirv::StructType>();
  auto elemNum = structTy.getNumElements();
  auto elemTy = structTy.getElementType(0);

  Value offsetOutter = mul(i32_val(repOutter), repNonKDimStride);
  Value offsetInner = mul(i32_val(repInner), repKDimStride);
  Value offset = add(offsetOutter, offsetInner);

#if DEBUG_PRINT
#if 1
  auto printFuncTy = mlir::FunctionType::get(
      rewriter.getContext(), {i32_ty, i32_ty, i32_ty, i32_ty, f16_ty},
      TypeRange());

  NamedAttrList attributes;
  attributes.set("libname",
                 StringAttr::get(rewriter.getContext(), "libdevice"));
  attributes.set("libpath", StringAttr::get(rewriter.getContext(), ""));
  auto linkageTypeAttr = rewriter.getAttr<::mlir::spirv::LinkageTypeAttr>(
      spirv::LinkageType::Import);
  auto linkageAttr = rewriter.getAttr<::mlir::spirv::LinkageAttributesAttr>(
      "print_scalar", linkageTypeAttr);
  attributes.set("linkage_attributes", linkageAttr);
  spirv::appendOrGetFuncOp(loc, rewriter, "print_scalar", printFuncTy,
                           spirv::FunctionControl::Inline, attributes);
#else

  auto printFuncTy = mlir::FunctionType::get(
      rewriter.getContext(),
      {i32_ty, i32_ty, i32_ty, ptr_ty(f16_ty, spirv::StorageClass::Workgroup),
       f16_ty},
      TypeRange());

  NamedAttrList attributes;
  attributes.set("libname",
                 StringAttr::get(rewriter.getContext(), "libdevice"));
  attributes.set("libpath", StringAttr::get(rewriter.getContext(), ""));
  auto linkageTypeAttr = rewriter.getAttr<::mlir::spirv::LinkageTypeAttr>(
      spirv::LinkageType::Import);
  auto linkageAttr = rewriter.getAttr<::mlir::spirv::LinkageAttributesAttr>(
      "print_scalar2", linkageTypeAttr);
  attributes.set("linkage_attributes", linkageAttr);
  spirv::appendOrGetFuncOp(loc, rewriter, "print_scalar2", printFuncTy,
                           spirv::FunctionControl::Inline, attributes);
#endif
#endif

  Value ret = rewriter.create<spirv::UndefOp>(loc, structTy);
  for (int i = 0; i < elemNum; i++) {
    Value readPtr = gep(ptrs[i], offset);
    Value val = rewriter.create<spirv::LoadOp>(loc, readPtr);
#if DEBUG_PRINT
    if (kDim == 1)
      rewriter.create<spirv::FunctionCallOp>(
          loc, TypeRange(), "print_scalar",
          ValueRange{subGroupID, lane, i32_val(i), offs[i], val});
#endif
    ret = insert_val(structTy, val, ret, rewriter.getI32ArrayAttr(i));
  }
  return ret;
}

JointMatrixMatmulLoader::JointMatrixMatmulLoader(
    ArrayRef<uint32_t> order, int warpsPerTile, uint32_t kDim, int repeatCount,
    int systolicDepth, int exeCutionSize, int opsPerChan, int threadsPerWarp,
    ArrayRef<Value> smemStrides, ArrayRef<int> instrShape, int perPhase,
    int maxPhase, int vec, int elemBytes, ConversionPatternRewriter &rewriter,
    TritonGPUToSPIRVTypeConverter *typeConverter, const Location &loc)
    : order(order.begin(), order.end()), kDim(kDim), repeatCount(repeatCount),
      systolicDepth(systolicDepth), exeCutionSize(exeCutionSize),
      opsPerChan(opsPerChan), threadsPerWarp(threadsPerWarp),
      instrShape(instrShape.begin(), instrShape.end()),
      smemStrides(smemStrides.begin(), smemStrides.end()), perPhase(perPhase),
      maxPhase(maxPhase), vec(vec), elemBytes(elemBytes), rewriter(rewriter),
      loc(loc), ctx(rewriter.getContext()) {

  // if the k dim is contiguous, then load the value just as packed.
  needTrans = kDim != order[0];

  repKDimStride = mul(i32_val(instrShape[kDim]), smemStrides[kDim]);
  repNonKDimStride =
      mul(i32_val(instrShape[kDim ^ 1] * warpsPerTile), smemStrides[kDim ^ 1]);
  warpMatStride = mul(i32_val(instrShape[kDim ^ 1]), smemStrides[kDim ^ 1]);

  if (kDim == 1 /*A*/) {
    int rowsPerWarp = threadsPerWarp / systolicDepth;
    numPtrs = (repeatCount / rowsPerWarp) * opsPerChan;
  } else /*B*/ {
    int rowsPerWarp = threadsPerWarp / exeCutionSize;
    numPtrs = (repeatCount / rowsPerWarp) * opsPerChan;
  }
}

Value composeValuesToDotOperandLayoutStruct(
    const ValueTable &vals, int n0, int n1,
    TritonGPUToSPIRVTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter) {
  std::vector<Value> elems;
  for (int m = 0; m < n0; ++m)
    for (int k = 0; k < n1; ++k) {
      auto matVal = vals.at({m, k});
      auto matType = matVal.getType().cast<spirv::StructType>();
      auto valTy = matType.getElementType(0);
      for (int i = 0; i < matType.getNumElements(); ++i) {
        auto val = extract_val(valTy, matVal, rewriter.getI32ArrayAttr(i));
        elems.push_back(val);
      }
    }

  assert(!elems.empty());

  Type elemTy = elems[0].getType();
  Type structTy =
      spirv::StructType::get(SmallVector<Type>(elems.size(), elemTy));
  auto result = typeConverter->packLLElements(loc, elems, rewriter, structTy);
  return result;
}

std::function<void(int, int)>
getLoadMatrixFn(Value tensor, const RankedTensorType &dstType,
                const SharedMemoryObject &smemObj,
                triton::gpu::intel::IntelMmaEncodingAttr mmaLayout,
                int warpsPerTile, uint32_t kDim, SmallVector<int> instrShape,
                Value warp, Value subGroupID, Value lane, ValueTable &vals,
                TritonGPUToSPIRVTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, Location loc) {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  Type eltTy = tensorTy.getElementType();
  // We assumes that the input operand of Dot should be from shared layout.
  // TODO(Superjomn) Consider other layouts if needed later.
  auto sharedLayout = tensorTy.getEncoding().cast<SharedEncodingAttr>();
  const int perPhase = sharedLayout.getPerPhase();
  const int maxPhase = sharedLayout.getMaxPhase();
  const int vec = sharedLayout.getVec();
  const int elemBytes = tensorTy.getElementTypeBitWidth() / 8;
  auto order = sharedLayout.getOrder();

  // (a, b) is the coordinate.
  auto load = [=, &rewriter, &vals](int a, int b) {
    JointMatrixMatmulLoader loader(
        sharedLayout.getOrder(), warpsPerTile, kDim, mmaLayout.getRepeatCount(),
        mmaLayout.getSystolicDepth(), mmaLayout.getExecutionSize(),
        mmaLayout.getOpsPerChan(), mmaLayout.getSugGroupSize(), smemObj.strides,
        instrShape, perPhase, maxPhase, vec, elemBytes, rewriter, typeConverter,
        loc);

    // Offset of a slice within the original tensor in shared memory
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    auto offs = loader.computeOffsets(subGroupID, lane, cSwizzleOffset);
    // initialize pointers
    const int numPtrs = loader.getNumPtrs();
    SmallVector<Value> ptrs(numPtrs);

    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
    Type smemPtrTy = spirv::getSharedMemPtrTy(eltTy);
#if 0
    llvm::outs() << "johnlu JointMatrixMatmulLoader smemBase:" << smemBase
                 << "\n";
    llvm::outs().flush();
    llvm::outs() << "johnlu matTy eltTy:" << eltTy << "\n";
    llvm::outs().flush();
    llvm::outs() << "johnlu matTy smemPtrTy:" << smemPtrTy << "\n";
    llvm::outs().flush();
#endif
    for (int i = 0; i < numPtrs; ++i)
      ptrs[i] = bitcast(gep(smemPtrTy, smemBase, offs[i]), smemPtrTy);

      // actually load from shared memory
#if 0
    llvm::outs() << "johnlu matTy dstType:" << dstType << "\n";
    llvm::outs().flush();
#endif
    auto totalElem = product<int>(instrShape);
    auto threadsPerWarp = mmaLayout.getSugGroupSize();
    auto matTy = spirv::StructType::get(SmallVector<Type>(
        totalElem / threadsPerWarp, typeConverter->convertType(eltTy)));
#if 0
    llvm::outs() << "johnlu matTy matTy:" << matTy << "\n";
    llvm::outs().flush();
    llvm::outs() << "load a: " << a << " b:" << b << "\n";
    llvm::outs().flush();
#endif
    auto matrix = loader(a /*mat0*/, b /*mat1*/, ptrs, offs, warp, lane, matTy,
                         cSwizzleOffset);

    vals[{a, b}] = matrix;
  };

  return load;
}

Value loadArg(ConversionPatternRewriter &rewriter, Location loc, Value tensor,
              RankedTensorType dotOpType, const SharedMemoryObject &smemObj,
              TritonGPUToSPIRVTypeConverter *typeConverter, Value thread,
              bool isA) {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();

  DotOperandEncodingAttr encoding =
      dotOpType.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  auto mmaLayout =
      encoding.getParent().cast<triton::gpu::intel::IntelMmaEncodingAttr>();

  SmallVector<int64_t> shape(tensorTy.getShape().begin(),
                             tensorTy.getShape().end());

  ValueTable vals;
  auto shapePerCTATile = mmaLayout.getShapePerCTATile(shape);
  auto shapeA = mmaLayout.getShapeA();
  auto shapeB = mmaLayout.getShapeB();
  int mmaInstrM = shapeA[0], mmaInstrN = shapeB[1], mmaInstrK = shapeA[1];

  auto numRep = mmaLayout.getXMXRep(tensorTy.getShape(), encoding.getOpIdx());

  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto order = triton::gpu::getOrder(mmaLayout);

  auto threadsPerWarp =
      product<unsigned>(triton::gpu::getThreadsPerWarp(mmaLayout));
  Value warp = udiv(thread, i32_val(threadsPerWarp));
  Value lane = urem(thread, i32_val(threadsPerWarp));

  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warp, warpsPerCTA, order);
  Value warpM = urem(multiDimWarpId[0], i32_val(ceil(shape[0] / mmaInstrM)));
  Value warpN = urem(multiDimWarpId[1], i32_val(ceil(shape[1] / mmaInstrN)));

  int warpsPerTile;
  if (isA)
    warpsPerTile = std::min<int>(warpsPerCTA[0], ceil(shape[0] / mmaInstrM));
  else
    warpsPerTile = std::min<int>(warpsPerCTA[1], ceil(shape[1] / mmaInstrN));

#if 0
  llvm::outs() << "johnlu load joint matrix operand " << (isA ? "A:" : "B:")
               << tensor.getType() << "\n";
  llvm::outs() << "johnlu load joint matrix operand " << dotOpType << "\n";
  llvm::outs() << "johnlu load joint matrix warpsPerTile " << warpsPerTile
               << "\n";

  llvm::outs() << "johnlu load joint matrix mmaInstrM " << mmaInstrM << "\n";
  llvm::outs() << "johnlu load joint matrix mmaInstrN " << mmaInstrN << "\n";
  llvm::outs() << "johnlu load joint matrix mmaInstrK " << mmaInstrK << "\n";

  llvm::outs() << "johnlu load joint matrix warpsPerCTA: ";

  for (auto &i : warpsPerCTA)
    llvm::outs() << " " << i;
  llvm::outs() << "\n";

  llvm::outs() << "johnlu load joint matrix tensor shape: ";
  for (auto &i : shape)
    llvm::outs() << " " << i;
  llvm::outs() << "\n";

  llvm::outs() << "johnlu load joint matrix shapePerCTATile: ";
  for (auto &i : shapePerCTATile)
    llvm::outs() << " " << i;
  llvm::outs() << "\n";

  llvm::outs() << "johnlu load joint matrix tileShape of operands (numRep): ";
  for (auto &i : numRep)
    llvm::outs() << " " << i;
  llvm::outs() << "\n";
  llvm::outs().flush();

#endif

  std::function<void(int, int)> loadFn;
  if (isA)
    loadFn = getLoadMatrixFn(
        tensor, dotOpType, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/,
        1 /*kDim*/, {mmaInstrM, mmaInstrK} /*instrShape*/, warp,
        warpM /*warpId*/, lane /*laneId*/, vals /*vals*/,
        typeConverter /* typeConverter */, rewriter /*rewriter*/, loc /*loc*/);
  else
    loadFn = getLoadMatrixFn(
        tensor, dotOpType, smemObj, mmaLayout, warpsPerTile /*warpsPerTile*/,
        0 /*kDim*/, {mmaInstrK, mmaInstrN} /*instrShape*/, warp,
        warpN /*warpId*/, lane /*laneId*/, vals /*vals*/,
        typeConverter /* typeConverter */, rewriter /*rewriter*/, loc /*loc*/);

  // Perform loading.
  int numRepOuter = isA ? numRep[0] : numRep[1];
  int numRepK = isA ? numRep[1] : numRep[0];
  for (int m = 0; m < numRepOuter; ++m)
    for (int k = 0; k < numRepK; ++k)
      loadFn(m, k);

  // Format the values to LLVM::Struct to passing to mma codegen.
  return composeValuesToDotOperandLayoutStruct(vals, numRepOuter, numRepK,
                                               typeConverter, loc, rewriter);
}

namespace SharedToDotOperandXMX {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, RankedTensorType dotOpType,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToSPIRVTypeConverter *typeConverter,
                    Value thread) {
  if (opIdx == 0)
    return loadArg(rewriter, loc, tensor, dotOpType, smemObj, typeConverter,
                   thread, true);
  else {
    assert(opIdx == 1);
    return loadArg(rewriter, loc, tensor, dotOpType, smemObj, typeConverter,
                   thread, false);
  }
}
} // namespace SharedToDotOperandXMX
