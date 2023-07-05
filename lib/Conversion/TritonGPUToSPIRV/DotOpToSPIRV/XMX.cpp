#include "../DotOpToSPIRV.h"
#include "../Utility.h"
#include "triton/Conversion/TritonGPUToSPIRV/ESIMDHelper.h"
#include "triton/Conversion/TritonGPUToSPIRV/VCIntrinsicHelper.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

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
    RankedTensorType type) {

  auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);
  int offset{};
  ValueTableV2 vals;
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; j++) {
      vals[{i, j}] = elems[offset++];
    }
  }
  return vals;
}

enum class TensorCoreType : uint8_t {
  // floating-point tensor core instr
  FP32_FP16_FP16_FP32 = 0, // default
  FP32_BF16_BF16_FP32,
  FP32_TF32_TF32_FP32,
  FP16_FP16_FP16_FP16,
  // integer tensor core instr
  INT32_INT1_INT1_INT32, // Not implemented
  INT32_INT4_INT4_INT32, // Not implemented
  INT32_INT8_INT8_INT32, // Not implemented
  //
  NOT_APPLICABLE,
};

Type getMmaRetType(TensorCoreType mmaType, MLIRContext *ctx) {
  Type fp32Ty = type::f32Ty(ctx);
  Type fp16Ty = type::f16Ty(ctx);
  Type i32Ty = type::i32Ty(ctx);
  Type fp32x4Ty = spirv::StructType::get(SmallVector<Type>(4, fp32Ty));
  Type i32x4Ty = spirv::StructType::get(SmallVector<Type>(4, i32Ty));
  Type fp16x2Pack2Ty =
      spirv::StructType::get(SmallVector<Type>(2, vec_ty(fp16Ty, 2)));
  switch (mmaType) {
  case TensorCoreType::FP32_FP16_FP16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_BF16_BF16_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP32_TF32_TF32_FP32:
    return fp32x4Ty;
  case TensorCoreType::FP16_FP16_FP16_FP16:
    return fp16x2Pack2Ty;
  case TensorCoreType::INT32_INT8_INT8_INT32:
    return i32x4Ty;
  default:
    llvm::report_fatal_error("Unsupported mma type found");
  }

  return Type{};
}

TensorCoreType getMmaType(triton::DotOp op) {
  Value A = op.getA();
  Value B = op.getB();
  auto aTy = A.getType().cast<RankedTensorType>();
  auto bTy = B.getType().cast<RankedTensorType>();
  // d = a*b + c
  auto dTy = op.getD().getType().cast<RankedTensorType>();

  if (dTy.getElementType().isF32()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP32_FP16_FP16_FP32;
    if (aTy.getElementType().isBF16() && bTy.getElementType().isBF16())
      return TensorCoreType::FP32_BF16_BF16_FP32;
    if (aTy.getElementType().isF32() && bTy.getElementType().isF32() &&
        op.getAllowTF32())
      return TensorCoreType::FP32_TF32_TF32_FP32;
  } else if (dTy.getElementType().isInteger(32)) {
    if (aTy.getElementType().isInteger(8) && bTy.getElementType().isInteger(8))
      return TensorCoreType::INT32_INT8_INT8_INT32;
  } else if (dTy.getElementType().isF16()) {
    if (aTy.getElementType().isF16() && bTy.getElementType().isF16())
      return TensorCoreType::FP16_FP16_FP16_FP16;
  }

  return TensorCoreType::NOT_APPLICABLE;
}

inline static const std::map<TensorCoreType, std::string> mmaInstrPtx = {
    {TensorCoreType::FP32_FP16_FP16_FP32,
     "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"},
    {TensorCoreType::FP32_BF16_BF16_FP32,
     "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"},
    {TensorCoreType::FP32_TF32_TF32_FP32,
     "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"},

    {TensorCoreType::INT32_INT1_INT1_INT32,
     "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc"},
    {TensorCoreType::INT32_INT4_INT4_INT32,
     "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32"},
    {TensorCoreType::INT32_INT8_INT8_INT32,
     "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"},

    {TensorCoreType::FP16_FP16_FP16_FP16,
     "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"},
};

LogicalResult convertDot(TritonGPUToSPIRVTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Value a, Value b, Value c, Value d, Value loadedA,
                         Value loadedB, Value loadedC, DotOp op,
                         DotOpAdaptor adaptor) {
  MLIRContext *ctx = c.getContext();
  auto aTensorTy = a.getType().cast<RankedTensorType>();
  auto bTensorTy = b.getType().cast<RankedTensorType>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();

  auto aShapePerCTA = triton::gpu::getShapePerCTA(aTensorTy);
  auto bShapePerCTA = triton::gpu::getShapePerCTA(bTensorTy);
  auto dShapePerCTA = triton::gpu::getShapePerCTA(dTensorTy);

  int bitwidth = aTensorTy.getElementType().getIntOrFloatBitWidth();
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

  llvm::outs() << "johnlu loadedA: " << loadedA << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu loadedB: " << loadedB << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu loadedC: " << loadedC << "\n";
  llvm::outs().flush();
  // shape / shape_per_cta
  auto ha = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                loadedA, repM, repK, aTensorTy);
  auto hb = getValuesFromDotOperandLayoutStruct(typeConverter, loc, rewriter,
                                                loadedB, repN, repK, bTensorTy);
  auto fc = typeConverter->unpackLLElements(loc, loadedC, rewriter, dTensorTy);

  //  auto mmaType = getMmaType(op);
  auto mmaType = typeConverter->convertType(dTensorTy);

  llvm::outs() << "johnlu ha.size(): " << ha.size() << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu hb.size(): " << hb.size() << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu fc.size(): " << fc.size() << "\n";
  llvm::outs().flush();
  llvm::outs() << "johnlu mmaType: " << mmaType << "\n";
  llvm::outs().flush();

  //  std::string libName = "libdevice";
  //  std::string curPrint = "print_cur_float";
  //  std::string accPrint = "print_acc_float";
  //  std::string outPrint = "print_output_float";
  //  std::string writeIndexPrint = "print_write_index";
  //  appendOrGetFuncOp(rewriter, op, libName, writeIndexPrint,
  //                    mlir::FunctionType::get(rewriter.getContext(), {i32_ty,
  //                    i32_ty, i32_ty}, TypeRange()));

  //  spirv::FuncOp func = spirv::appendOrGetFuncOp(loc, rewriter, "libdevice",
  //  "llvm.genx.dpas",
  //   mlir::FunctionType::get(rewriter.getContext(), {i32_ty, i32_ty, i32_ty},
  //   TypeRange()),
  //                                                  spirv::FunctionControl::Inline,attributes);

  std::string simdFunc = "SIMDwrapper";
#if 1
  auto valueTy = i32_ty;
  auto valSIMDTy = mlir::VectorType::get({(int64_t)32}, valueTy);

  auto aTy = mlir::VectorType::get(128, f16_ty);
  auto bTy = mlir::VectorType::get(128, i32_ty);
  auto cTy = mlir::VectorType::get(128, i32_ty);
  auto dTy = mlir::VectorType::get(128, i32_ty);
  SmallVector<mlir::Type *, 4> funcTys;
  funcTys.push_back(&dTy);
  funcTys.push_back(&aTy);
  funcTys.push_back(&bTy);
  funcTys.push_back(&cTy);

  //  auto xmxIntrinsic =
  //  triton::intel::getGenXName(llvm::GenXIntrinsic::ID::genx_dpas, funcTys);
  //  llvm::outs() << "johnlu xmxIntrinsic name:" << xmxIntrinsic << "\n";

  auto simdFunTy =
      mlir::FunctionType::get(rewriter.getContext(), {valSIMDTy}, {dTy});
  spirv::FuncOp wrapper;
  {
    // SIMD function
    NamedAttrList attributes;
    wrapper = mlir::triton::intel::appendOrGetESIMDFunc(rewriter, simdFunc,
                                                        simdFunTy, loc);
    if (wrapper.empty()) {
      auto block = wrapper.addEntryBlock();
      auto entryBlock = &wrapper.getFunctionBody().front();

      OpBuilder rewriter(entryBlock, entryBlock->begin());

      Value retVal = rewriter.create<spirv::UndefOp>(loc, bTy);

      rewriter.create<spirv::ReturnValueOp>(loc, TypeRange(),
                                            ValueRange{retVal});
    }
  }
  auto funPtrTy = spirv::FunctionPointerINTELType::get(
      simdFunTy, spirv::StorageClass::Function);
  spirv::INTELConstantFunctionPointerOp funValue =
      rewriter.create<spirv::INTELConstantFunctionPointerOp>(loc, funPtrTy,
                                                             simdFunc);

  //  std::cout << "johnlu address of" << std::endl;
  //  (*this)->print(llvm::outs());
  //  llvm::outs().flush();
  //  std::cout << std::endl;
#if 0
  auto symbolOp = SymbolTable::lookupNearestSymbolFrom(funValue->getParentOp(),
                                     funValue.getVariableAttr());
    if(symbolOp) {
      std::cout << "start symbolTableOp" << std::endl;
      symbolOp->print(llvm::outs());
      llvm::outs().flush();
      std::cout << std::endl;
      std::cout << "end symbolTableOp" << std::endl;
      auto funcOp = dyn_cast_or_null<spirv::FuncOp>(symbolOp);
      std::cout << "start symbolOp type" << std::endl;
      funcOp.getFunctionType().print(llvm::outs());
      llvm::outs().flush();
      std::cout << std::endl;
      std::cout << "end symbolOp type" << std::endl;
      std::cout << "start pointer type" << std::endl;
      funValue->getResult(0).print(llvm::outs());
//      funValue.getPointer().print(llvm::outs());
      llvm::outs().flush();
      std::cout << std::endl;
      std::cout << "end pointer type" << std::endl;
    }
#endif

  auto simtDTy = mlir::VectorType::get(128 / 32, i32_ty);
  std::string intfSIMDFunc = "_Z33__regcall3____builtin_invoke_simd" + simdFunc;
  auto simtToSIMDFunTy = mlir::FunctionType::get(
      rewriter.getContext(), {funPtrTy, valueTy}, {simtDTy});
  {
    // SIMT -> SIMD calling abi
    auto simtWrapper =
        spirv::appendOrGetFuncOp(loc, rewriter, intfSIMDFunc, simtToSIMDFunTy,
                                 spirv::FunctionControl::Inline);

    simtWrapper->setAttr(
        spirv::SPIRVDialect::getAttributeName(
            spirv::Decoration::LinkageAttributes),
        spirv::LinkageAttributesAttr::get(
            rewriter.getContext(), intfSIMDFunc,
            spirv::LinkageTypeAttr::get(rewriter.getContext(),
                                        spirv::LinkageType::Import)));
  }
#endif
  auto callMma = [&](unsigned m, unsigned n, unsigned k) {
    //    Type elemTy = getMmaRetType(mmaType, op.getContext())
    //                      .cast<spirv::StructType>()
    //                      .getElementType(0);
    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange{simtDTy},
                                           intfSIMDFunc,
                                           ValueRange{funValue, i32_val(0)});
    Type elemTy = mmaType.cast<spirv::StructType>().getElementType(0);
    llvm::outs() << "johnlu elemTy: " << elemTy << "\n";
    llvm::outs().flush();
    fc[m * repN + n] = rewriter.create<spirv::UndefOp>(loc, elemTy);
  };

  for (int k = 0; k < repK; ++k)
    for (int m = 0; m < repM; ++m)
      for (int n = 0; n < repN; ++n)
        callMma(m, n, k);

  Type resElemTy = typeConverter->convertType(dTensorTy);

  llvm::outs() << "johnlu resElemTy: " << resElemTy << "\n";
  llvm::outs().flush();
  SmallVector<Value> results(fc.size());
  for (int i = 0; i < fc.size(); ++i) {
    results[i] = fc[i];
  }
  Value res = typeConverter->packLLElements(loc, results, rewriter, resElemTy);

  rewriter.replaceOp(op, res);

  return success();
}

// Convert to xmx
LogicalResult convertXMXDot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToSPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto mmaLayout = op.getResult()
                       .getType()
                       .cast<RankedTensorType>()
                       .getEncoding()
                       .cast<triton::gpu::intel::IntelMmaEncodingAttr>();

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
                    loadedA, loadedB, loadedC, op, adaptor);
}
