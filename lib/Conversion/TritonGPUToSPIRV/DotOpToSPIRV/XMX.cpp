#include "../DotOpToSPIRV.h"
#include "../Utility.h"
#include "GenIntrinsics.h"
#include "triton/Conversion/TritonGPUToSPIRV/ESIMDHelper.h"
#include "triton/Conversion/TritonGPUToSPIRV/VCIntrinsicHelper.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::intel;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;

using ValueTableV2 = std::map<std::pair<unsigned, unsigned>, Value>;

Value loadC(Value tensor, Value spirvTensor,
            TritonGPUToSPIRVTypeConverter *typeConverter, Location loc,
            ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = tensor.getContext();
  auto tensorTy = tensor.getType().cast<RankedTensorType>();
  //  size_t fcSize = triton::gpu::getTotalElemsPerThread(tensor.getType());

  //  llvm::outs() << "johnlu tensor " << tensor << "\n";
  //  llvm::outs() << "johnlu spirvTensor " << spirvTensor << "\n";
  //  llvm::outs().flush();

  assert(
      tensorTy.getEncoding().isa<triton::gpu::intel::IntelMmaEncodingAttr>() &&
      "Currently, we only support $c with a mma layout.");
  // Load a normal C tensor with mma layout, that should be a
  // LLVM::struct with fcSize elements.
  //  auto structTy = spirvTensor.getType().cast<spirv::StructType>();
  //  assert(structTy.getElementTypes().size() == fcSize &&
  //         "DotOp's $c operand should pass the same number of values as $d in
  //         " "mma layout.");

  return spirvTensor;
#if 0
  auto numMmaRets = tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  assert(numMmaRets == 4 || numMmaRets == 2);
  if (numMmaRets == 4) {
    return spirvTensor;
  } else if (numMmaRets == 2) {
    auto cPack = SmallVector<Value>();
    auto cElemTy = tensorTy.getElementType();
    int numCPackedElem = 4 / numMmaRets;
    Type cPackTy = vec_ty(cElemTy, numCPackedElem);
    for (int i = 0; i < fcSize; i += numCPackedElem) {
      Value pack = rewriter.create<spirv::UndefOp>(loc, cPackTy);
      for (int j = 0; j < numCPackedElem; ++j) {
        pack = insert_element(
            cPackTy, pack,
            extract_val(cElemTy, spirvTensor, rewriter.getI32ArrayAttr(i + j)),
            i32_val(j));
      }
      cPack.push_back(pack);
    }

    Type structTy =
        spirv::StructType::get(SmallVector<Type>(cPack.size(), cPackTy));
    Value result =
        typeConverter->packLLElements(loc, cPack, rewriter, structTy);
    return result;
  }

  return spirvTensor;
#endif
}

ValueTableV2 getValuesFromDotOperandLayoutStruct(
    TritonGPUToSPIRVTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter, Value value, int n0, int n1,
    RankedTensorType type, Type dotOperandType) {

  auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);
  int offset{};
  ValueTableV2 vals;
  auto totalElems = elems.size();
  auto numElemsPerOperand = totalElems / (n0 * n1);
  auto elemTy = elems[0].getType();
  auto matTy = vec_ty(elemTy, numElemsPerOperand);

  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      Value matVal = rewriter.create<spirv::UndefOp>(loc, matTy);
      for (int k = 0; k < numElemsPerOperand; ++k) {
        matVal = insert_element(matTy, matVal, elems[offset++], i32_val(k));
      }
      vals[{i, j}] = bitcast(matVal, dotOperandType);
    }
  }
  return vals;
}

static Value composeValuesToDotOperandLayoutStruct(
    const ValueTableV2 &vals, int n0, int n1,
    TritonGPUToSPIRVTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter
#if 1
    ,
    Value warp, Value lane
#endif
) {
  std::vector<Value> elems;
#if 0
  auto printFuncTy = mlir::FunctionType::get(
      rewriter.getContext(), {i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, f32_ty}, TypeRange());

  NamedAttrList attributes;
  attributes.set("libname", StringAttr::get(rewriter.getContext(), "libdevice"));
  attributes.set("libpath", StringAttr::get(rewriter.getContext(), ""));
  auto linkageTypeAttr =
      rewriter.getAttr<::mlir::spirv::LinkageTypeAttr>(spirv::LinkageType::Import);
  auto linkageAttr = rewriter.getAttr<::mlir::spirv::LinkageAttributesAttr>(
      "print_mm", linkageTypeAttr);
  attributes.set("linkage_attributes", linkageAttr);
  spirv::appendOrGetFuncOp(loc, rewriter, "print_mm", printFuncTy,
                           spirv::FunctionControl::Inline, attributes);
#endif
  for (int m = 0; m < n0; ++m)
    for (int k = 0; k < n1; ++k) {
      auto matVal = vals.at({m, k});
      auto vecType = matVal.getType().cast<mlir::VectorType>();
      auto valTy = vecType.getElementType();
      for (int i = 0; i < vecType.getNumElements(); ++i) {
        auto val = extract_element(valTy, matVal, i32_val(i));
#if 0
        rewriter.create<spirv::FunctionCallOp>(
            loc, TypeRange(), "print_mm",
            ValueRange{warp, lane, i32_val(m), i32_val(k), i32_val(i), val});
#endif
        elems.push_back(val);
      }
    }

  assert(!elems.empty());

  Type elemTy = elems[0].getType();
  MLIRContext *ctx = elemTy.getContext();
  Type structTy =
      spirv::StructType::get(SmallVector<Type>(elems.size(), elemTy));
  auto result = typeConverter->packLLElements(loc, elems, rewriter, structTy);
  return result;
}

std::tuple<Type, Type, Type, Type>
getMmaOperandsType(DPASEngineType mmaType, MLIRContext *ctx,
                   mlir::triton::gpu::intel::IntelMmaEncodingAttr layout) {
  Type fp32Ty = type::f32Ty(ctx);
  Type fp16Ty = type::f16Ty(ctx);
  Type bf16Ty = type::bf16Ty(ctx);
  Type i32Ty = type::i32Ty(ctx);
  Type i16Ty = type::i16Ty(ctx);

  auto threadsPerWarp = layout.getSugGroupSize();
  auto opsPerChan = layout.getOpsPerChan();
  auto shapeC = layout.getShapeC();
  auto elemNumC = product<unsigned>(shapeC) / threadsPerWarp;
  auto shapeA = layout.getShapeA();
  auto elemNumA = product<unsigned>(shapeA) / threadsPerWarp;
  auto shapeB = layout.getShapeB();
  auto elemNumB = product<unsigned>(shapeB) / threadsPerWarp;
  switch (mmaType) {
  case DPASEngineType::FP32_FP32_FP16_FP16: {
    Type cTy = vec_ty(fp32Ty, elemNumC);
    Type aTy;
    if (threadsPerWarp == 32) {
      aTy = vec_ty(i16Ty, elemNumA);
    } else {
      aTy = vec_ty(i32Ty, elemNumA / opsPerChan); // pack scalar to i32.
    }
    Type bTy = vec_ty(i32Ty, elemNumB / opsPerChan); // pack scalar to i32.
    return {cTy, cTy, aTy, bTy};
  }
  case DPASEngineType::FP16_FP16_FP16_FP16: {
    Type cTy = vec_ty(fp16Ty, elemNumC);
    Type aTy;
    if (threadsPerWarp == 32) {
      aTy = vec_ty(i16Ty, elemNumA);
    } else {
      aTy = vec_ty(i32Ty, elemNumA / opsPerChan); // pack scalar to i32.
    }
    Type bTy = vec_ty(i32Ty, elemNumB / opsPerChan); // pack scalar to i32.
    return {cTy, cTy, aTy, bTy};
  }
  case DPASEngineType::FP32_FP32_BF16_BF16: {
    Type cTy = vec_ty(fp32Ty, elemNumC);
    Type aTy;
    if (threadsPerWarp == 32) {
      aTy = vec_ty(i16Ty, elemNumA);
    } else {
      aTy = vec_ty(i32Ty, elemNumA / opsPerChan); // pack scalar to i32.
    }
    Type bTy = vec_ty(i32Ty, elemNumB / opsPerChan); // pack scalar to i32.
    return {cTy, cTy, aTy, bTy};
  }
  case DPASEngineType::BF16_BF16_BF16_BF16: {
    Type cTy = vec_ty(bf16Ty, elemNumC);
    Type aTy;
    if (threadsPerWarp == 32) {
      aTy = vec_ty(i16Ty, elemNumA);
    } else {
      aTy = vec_ty(i32Ty, elemNumA / opsPerChan); // pack scalar to i32.
    }
    Type bTy = vec_ty(i32Ty, elemNumB / opsPerChan); // pack scalar to i32.
    return {cTy, cTy, aTy, bTy};
  }
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return std::make_tuple<Type, Type, Type, Type>({}, {}, {}, {});
}

DPASEngineType getMmaType(triton::DotOp op) {
  Value A = op.getA();
  Value B = op.getB();
  auto aTy = A.getType().cast<RankedTensorType>();
  auto bTy = B.getType().cast<RankedTensorType>();
  // d = a*b + c
  auto dTy = op.getD().getType().cast<RankedTensorType>();

  if (dTy.getElementType().isF32()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return DPASEngineType::FP32_FP32_FP16_FP16;
    if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
      return DPASEngineType::FP32_FP32_BF16_BF16;
    if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        op.getAllowTF32())
      return DPASEngineType::FP32_FP32_TF32_TF32;
  } else if (dTy.getElementType().isInteger(32)) {
    // TODO: add integer dpas.
  } else if (dTy.getElementType().isF16()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return DPASEngineType::FP16_FP16_FP16_FP16;
  }

  return DPASEngineType::NOT_APPLICABLE;
}

LogicalResult convertDot(TritonGPUToSPIRVTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value a, Value b, Value c, Value d, Value loadedA,
                         Value loadedB, Value loadedC, DotOp op,
                         DotOpAdaptor adaptor, Value tid) {
  auto aTensorTy = a.getType().cast<RankedTensorType>();
  auto bTensorTy = b.getType().cast<RankedTensorType>();
  auto cTensorTy = c.getType().cast<RankedTensorType>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();

  auto aShapePerCTA = triton::gpu::getShapePerCTA(aTensorTy);
  auto bShapePerCTA = triton::gpu::getShapePerCTA(bTensorTy);
  auto dShapePerCTA = triton::gpu::getShapePerCTA(dTensorTy);

  auto dotOpA = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto repA = dotOpA.getParent()
                  .cast<triton::gpu::intel::IntelMmaEncodingAttr>()
                  .getXMXRep(aShapePerCTA, dotOpA.getOpIdx());
  auto dotOpB = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto repB = dotOpB.getParent()
                  .cast<triton::gpu::intel::IntelMmaEncodingAttr>()
                  .getXMXRep(bShapePerCTA, dotOpB.getOpIdx());

  assert(repA[1] == repB[0]);
  int repM = repA[0], repN = repB[1], repK = repA[1];
#if 0
  llvm::outs() << "johnlu repM: " << repM << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu repN: " << repN << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu repK: " << repK << "\n";
  llvm::outs().flush();

  llvm::outs() << "johnlu aTensorTy: " << aTensorTy << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu loadedA: " << loadedA << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu bTensorTy: " << bTensorTy << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu loadedB: " << loadedB << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu loadedC: " << loadedC << "\n";
  llvm::outs().flush();
#endif
  auto mmaType = getMmaType(op);
  auto srcXMXLayout =
      dTensorTy.getEncoding()
          .cast<mlir::triton::gpu::intel::IntelMmaEncodingAttr>();
  Type aTy, bTy, cTy, dTy;
  std::tie(dTy, cTy, aTy, bTy) =
      getMmaOperandsType(mmaType, op.getContext(), srcXMXLayout);

  // shape / shape_per_cta
  auto ha = getValuesFromDotOperandLayoutStruct(
      typeConverter, loc, rewriter, loadedA, repM, repK, aTensorTy, aTy);
  auto hb = getValuesFromDotOperandLayoutStruct(
      typeConverter, loc, rewriter, loadedB, repN, repK, bTensorTy, bTy);
  auto fc = getValuesFromDotOperandLayoutStruct(
      typeConverter, loc, rewriter, loadedC, repM, repN, cTensorTy, cTy);

  auto shapePerCTA = mlir::triton::gpu::getShapePerCTA(dTensorTy.getEncoding(),
                                                       dTensorTy.getShape());
  auto shapePerCTATile = mlir::triton::gpu::getShapePerCTATile(
      dTensorTy.getEncoding(), dTensorTy.getShape());
#if 0
  llvm::outs() << "johnlu shapePerCTA: ";
  for (auto &i : shapePerCTA)
    llvm::outs() << " " << i;
  llvm::outs() << "\n";
  llvm::outs() << "johnlu shapePerCTATile: ";
  for (auto &i : shapePerCTATile)
    llvm::outs() << " " << i;
  llvm::outs() << "\n";
  llvm::outs() << "johnlu ha.size(): " << ha.size() << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu hb.size(): " << hb.size() << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu fc.size(): " << fc.size() << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu dTensorTy: " << dTensorTy << "\n";
  llvm::outs().flush();
#endif
  auto mod = op->getParentOfType<ModuleOp>();
  unsigned threadsPerWarp =
      triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

  //  llvm::outs() << "johnlu aTy: " << aTy << "\n";
  //  llvm::outs().flush();
  //  llvm::outs() << "johnlu bTy: " << bTy << "\n";
  //  llvm::outs().flush();
  //  llvm::outs() << "johnlu cTy: " << cTy << "\n";
  //  llvm::outs().flush();
  //  llvm::outs() << "johnlu dTy: " << dTy << "\n";
  //  llvm::outs().flush();

  auto callMma = [&](unsigned m, unsigned n, unsigned k) {
    auto valA = ha.at({m, k});
    auto valB = hb.at({n, k});
    auto valc = fc.at({m, n});
    IntrinsicBuilder builder(threadsPerWarp, rewriter);
    GenISA_DPAS dpas2Intrinsic = *builder.create<GenISA_DPAS>(
        mmaType, srcXMXLayout.getSystolicDepth(), srcXMXLayout.getRepeatCount(),
        dTy, cTy, aTy, bTy);
    auto ret = dpas2Intrinsic(rewriter, loc, valc, valA, valB);
//    auto ret = rewriter.create<spirv::FunctionCallOp>(
//        loc, TypeRange{dTy},
//        "llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32",
//        ValueRange{valc, valA, valB, i32_val(10), i32_val(10), i32_val(8),
//                   i32_val(8), int_val(1, 0)});
//    auto mmaType = typeConverter->convertType(dTensorTy);
//    Type elemTy = mmaType.cast<spirv::StructType>().getElementType(0);
#if 0
    llvm::outs() << "johnlu XMX.cpp m: " << m << " n:" << n << " k:" << k
                 << "\n";
    llvm::outs() << "johnlu XMX.cpp mmaType: " << mmaType << "\n";
    llvm::outs() << "johnlu XMX.cpp elemTy: " << elemTy << "\n";
    llvm::outs().flush();
#endif
    //    fc.at({m, n}) = ret.getReturnValue();
    fc.at({m, n}) = ret;
    //    fc.at({m, n}) = rewriter.create<spirv::UndefOp>(loc, dTy);
  };

  for (int k = 0; k < repK; ++k)
    for (int m = 0; m < repM; ++m)
      for (int n = 0; n < repN; ++n)
        callMma(m, n, k);

  Type resElemTy = typeConverter->convertType(dTensorTy);

#if 0
  llvm::outs() << "johnlu resElemTy: " << resElemTy << "\n";
  llvm::outs().flush();
#endif
  // Format the values to LLVM::Struct to passing to mma codegen.
  Value warp = udiv(tid, i32_val(threadsPerWarp));
  Value lane = urem(tid, i32_val(threadsPerWarp));
  Value res = composeValuesToDotOperandLayoutStruct(
      fc, repM, repN, typeConverter, loc, rewriter, warp, lane);

  rewriter.replaceOp(op, res);

  return success();
}

// Convert to xmx
LogicalResult convertXMXDot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToSPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter, Value tid) {

  Value A = op.getA();
  Value B = op.getB();
  Value C = op.getC();

  auto ATensorTy = A.getType().cast<RankedTensorType>();
  auto BTensorTy = B.getType().cast<RankedTensorType>();

  assert(ATensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
         BTensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  Value loadedA, loadedB, loadedC;
  loadedA = adaptor.getA();
  loadedB = adaptor.getB();
  loadedC =
      loadC(op.getC(), adaptor.getC(), typeConverter, op.getLoc(), rewriter);

  return convertDot(typeConverter, rewriter, op.getLoc(), A, B, C, op.getD(),
                    loadedA, loadedB, loadedC, op, adaptor, tid);
}
