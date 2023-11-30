//
// Created by chengjun on 9/6/23.
//
#include "Utility.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "triton/Conversion/TritonGPUToSPIRV/ESIMDHelper.h"

namespace mlir {
namespace triton {
namespace intel {

using namespace mlir;

mlir::spirv::FuncOp appendOrGetESIMDFunc(OpBuilder &builder,
                                         std::string simdFuncName,
                                         FunctionType simdFuncTy, Location loc,
                                         const NamedAttrList &extraAttrs) {
  spirv::FuncOp wrapper;

  // SIMD function
  NamedAttrList attributes(extraAttrs);

  attributes.set(spirv::SPIRVDialect::getAttributeName(
                     spirv::Decoration::VectorComputeFunctionINTEL),
                 UnitAttr::get(builder.getContext()));
  attributes.set(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::StackCallINTEL),
      UnitAttr::get(builder.getContext()));
  attributes.set(spirv::SPIRVDialect::getAttributeName(
                     spirv::Decoration::ReferencedIndirectlyINTEL),
                 UnitAttr::get(builder.getContext()));

  attributes.set(spirv::SPIRVDialect::getAttributeName(
                     spirv::Decoration::LinkageAttributes),
                 spirv::LinkageAttributesAttr::get(
                     builder.getContext(), simdFuncName,
                     spirv::LinkageTypeAttr::get(builder.getContext(),
                                                 spirv::LinkageType::Export)));

  // SIMD function
  wrapper = spirv::appendOrGetFuncOp(loc, builder, simdFuncName, simdFuncTy,
                                     spirv::FunctionControl::None, attributes);

  //  wrapper->setAttr(spirv::SPIRVDialect::getAttributeName(
  //                       spirv::Decoration::VectorComputeFunctionINTEL),
  //                   UnitAttr::get(builder.getContext()));
  //  wrapper->setAttr(spirv::SPIRVDialect::getAttributeName(
  //                       spirv::Decoration::StackCallINTEL),
  //                   UnitAttr::get(builder.getContext()));
  //  wrapper->setAttr(spirv::SPIRVDialect::getAttributeName(
  //                       spirv::Decoration::ReferencedIndirectlyINTEL),
  //                   UnitAttr::get(builder.getContext()));
  //
  //  wrapper->setAttr(spirv::SPIRVDialect::getAttributeName(spirv::Decoration::LinkageAttributes),
  //                   spirv::LinkageAttributesAttr::get(builder.getContext(),
  //                                                     simdFuncName,
  //                                                     spirv::LinkageTypeAttr::get(builder.getContext(),
  //                                                     spirv::LinkageType::Export)));

  return wrapper;
}

} // namespace intel
} // namespace triton
} // namespace mlir

#if 0

class SIMDAdd
    : public ConvertTritonGPUOpToSPIRVPattern<arith::AddIOp> {
public:
  using OpAdaptor = typename arith::AddIOp::Adaptor;

  explicit SIMDAdd(
      TritonGPUToSPIRVTypeConverter &converter, MLIRContext *context,
      PatternBenefit benefit = 1, bool use_INTELConvertFToBF16Op = false)
      : ConvertTritonGPUOpToSPIRVPattern<arith::AddIOp>(converter, context, benefit),
        use_INTELConvertFToBF16Op(use_INTELConvertFToBF16Op) {}

  bool use_INTELConvertFToBF16Op = false;
  LogicalResult
  matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = op.getType();
    // element type
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<Value> resultVals;
    Location loc = op->getLoc();
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
    auto valSIMDTy = mlir::VectorType::get({(int64_t)32}, elemTy);
    auto simdFunTy =
        mlir::FunctionType::get(rewriter.getContext(), {valSIMDTy, valSIMDTy}, {valSIMDTy});
    spirv::FuncOp wrapper;
    {
      // SIMD function
      NamedAttrList attributes;
      wrapper = spirv::appendOrGetFuncOp(loc, rewriter, "", simdFunc, simdFunTy,
                                         spirv::FunctionControl::None);
      if (wrapper.empty()) {
        wrapper->setAttr(spirv::SPIRVDialect::getAttributeName(
                             spirv::Decoration::VectorComputeFunctionINTEL),
                         UnitAttr::get(rewriter.getContext()));
        wrapper->setAttr(spirv::SPIRVDialect::getAttributeName(
                             spirv::Decoration::StackCallINTEL),
                         UnitAttr::get(rewriter.getContext()));
        wrapper->setAttr(spirv::SPIRVDialect::getAttributeName(
                             spirv::Decoration::ReferencedIndirectlyINTEL),
                         UnitAttr::get(rewriter.getContext()));

        wrapper->setAttr(spirv::SPIRVDialect::getAttributeName(spirv::Decoration::LinkageAttributes),
                         spirv::LinkageAttributesAttr::get(rewriter.getContext(),
                                                           simdFunc,
                                                           spirv::LinkageTypeAttr::get(rewriter.getContext(), spirv::LinkageType::Export)));
        auto block = wrapper.addEntryBlock();

        OpBuilder rewriter(block, block->begin());
        //    rewriter.create<spirv::FunctionCallOp>(loc, TypeRange(),
        //    "llvm.genx.dpas",
        //                                          ValueRange{i32_val(0),
        //                                          i32_val(0), i32_val(0)});


        Value retVal = rewriter.create<spirv::IAddOp>(loc, block->getArgument(0), block->getArgument(1));

        rewriter.create<spirv::ReturnValueOp>(loc, TypeRange(), ValueRange{retVal});
      }
    }
    auto funPtrTy =
        spirv::FunctionPointerINTELType::get(simdFunTy, spirv::StorageClass::Function);
    spirv::INTELConstantFunctionPointerOp funValue =
        rewriter.create<spirv::INTELConstantFunctionPointerOp>(loc, funPtrTy, simdFunc);

    //  std::cout << "johnlu address of" << std::endl;
    //  (*this)->print(llvm::outs());
    //  llvm::outs().flush();
    //  std::cout << std::endl;

    std::string intfSIMDFunc = "_Z33__regcall3____builtin_invoke_simd" + simdFunc;
    auto simtToSIMDFunTy = mlir::FunctionType::get(
        rewriter.getContext(), {funPtrTy, elemTy, elemTy}, {elemTy});
    {
      // SIMT -> SIMD calling abi
      //    NamedAttrList attributes;
      //    attributes.set("libname",
      //                   StringAttr::get(rewriter.getContext(), "libdevice"));
      //    attributes.set("libpath", StringAttr::get(rewriter.getContext(), ""));
      //    attributes.set(
      //        "linkage_attributes",
      //        ArrayAttr::get(rewriter.getContext(),
      //                       {
      //                           StringAttr::get(rewriter.getContext(), intfSIMDFunc),
      //                           StringAttr::get(rewriter.getContext(), "Import"),
      //                       }));
      auto simtWrapper = spirv::appendOrGetFuncOp(loc, rewriter, "", intfSIMDFunc, simtToSIMDFunTy,
                                                  spirv::FunctionControl::Inline);

      simtWrapper->setAttr(spirv::SPIRVDialect::getAttributeName(spirv::Decoration::LinkageAttributes),
                           spirv::LinkageAttributesAttr::get(rewriter.getContext(),
                                                             intfSIMDFunc,
                                                             spirv::LinkageTypeAttr::get(rewriter.getContext(), spirv::LinkageType::Import)));

    }
#endif
    //
    SmallVector<SmallVector<Value>> allOperands;
    auto operands = adaptor.getOperands();
    for (const auto &operand : operands) {
      auto argTy = op->getOperand(0).getType();
      auto sub_operands = this->getTypeConverter()->unpackLLElements(
          loc, operand, rewriter, argTy);
      sub_operands = unpackI32(sub_operands, argTy, rewriter, loc,
                               this->getTypeConverter());
      allOperands.resize(sub_operands.size());
      auto vs = llvm::enumerate(sub_operands);
      for (const auto &v : vs)
        allOperands[v.index()].push_back(v.value());
    }
    if (allOperands.size() == 0)
      allOperands.push_back({});
    for (const SmallVector<Value> &operands : allOperands) {
      auto curr_ = rewriter.create<spirv::FunctionCallOp>(loc, TypeRange{elemTy},
                                                          intfSIMDFunc,
                                                          ValueRange{funValue, operands[0], operands[1]});
      Value curr = curr_.getReturnValue();
      if (!bool(curr))
        return failure();
      resultVals.push_back(curr);
    }
    if (op->getNumOperands() > 0) {
      auto argTy = op->getOperand(0).getType();
      resultVals = reorderValues(resultVals, argTy, resultTy);
    }
    resultVals =
        packI32(resultVals, resultTy, rewriter, loc, this->getTypeConverter());
    Value view = this->getTypeConverter()->packLLElements(loc, resultVals,
                                                          rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }
};

#endif
