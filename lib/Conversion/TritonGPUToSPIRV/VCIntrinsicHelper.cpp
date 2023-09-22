//
// Created by chengjun on 9/6/23.
//

#include "triton/Conversion/TritonGPUToSPIRV/VCIntrinsicHelper.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

namespace mlir {
namespace triton {
namespace intel {

std::string getGenXName(llvm::GenXIntrinsic::ID id,
                        ArrayRef<mlir::Type *> mlirTys) {
  SmallVector<llvm::Type *, 4> llvmTys;
  llvm::LLVMContext llvmContext;
  mlir::LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
  for (mlir::Type *ty : mlirTys) {
    llvmTys.push_back(typeTranslator.translateType(*ty));
  }
  return llvm::GenXIntrinsic::getGenXName(id, llvmTys);
}

mlir::FunctionType getGenXType(mlir::MLIRContext &context,
                               llvm::GenXIntrinsic::ID id,
                               ArrayRef<mlir::Type *> mlirTys) {
  SmallVector<llvm::Type *, 4> llvmTys;
  llvm::LLVMContext llvmContext;
  mlir::LLVM::TypeToLLVMIRTranslator llvmToMLIR(llvmContext);
  for (mlir::Type *ty : mlirTys) {
    llvmTys.push_back(llvmToMLIR.translateType(*ty));
  }
  auto llvmFuncType =
      llvm::GenXIntrinsic::getGenXType(llvmContext, id, llvmTys);

  LLVM::TypeFromLLVMIRTranslator mlirToLLVM(context);
  auto mlirType = mlirToLLVM.translateType(llvmFuncType);
  auto mlirLLVMFuncType = mlirType.cast<mlir::LLVM::LLVMFunctionType>();

  auto funcTy = mlir::FunctionType::get(&context, mlirLLVMFuncType.getParams(),
                                        mlirLLVMFuncType.getReturnTypes());
  return funcTy;
}

// The code convert the function attribute from the original here:
// https://github.com/llvm/llvm-project/blob/e575b7cb7a64297583d6382c16ce264d9fe45d08/mlir/lib/Target/LLVMIR/ModuleImport.cpp#L1547
static void processPassthroughAttrs(mlir::MLIRContext &context,
                                    NamedAttrList &attributes,
                                    const llvm::AttributeSet &funcAttrs) {
  for (llvm::Attribute attr : funcAttrs) {
    StringRef attrName;
    if (attr.isStringAttribute())
      attrName = attr.getKindAsString();
    else
      attrName = llvm::Attribute::getNameFromAttrKind(attr.getKindAsEnum());
    auto keyAttr = StringAttr::get(&context, attrName);

    if (attr.isStringAttribute()) {
      StringRef val = attr.getValueAsString();
      attributes.set(keyAttr, StringAttr::get(&context, val));
      continue;
    }
    if (attr.isIntAttribute()) {
      auto val = std::to_string(attr.getValueAsInt());
      attributes.set(keyAttr, StringAttr::get(&context, val));
      continue;
    }
    if (attr.isEnumAttribute()) {
      attributes.set(keyAttr, UnitAttr::get(&context));
      continue;
    }

    llvm_unreachable("unexpected attribute kind");
  }
}

spirv::FuncOp appendOrGetGenXDeclaration(OpBuilder &builder,
                                         llvm::GenXIntrinsic::ID id,
                                         ArrayRef<mlir::Type *> mlirTys) {
  auto mlirContext = builder.getContext();

  auto genXName = getGenXName(id, mlirTys);
  FunctionType funcTy = getGenXType(*mlirContext, id, mlirTys);

  NamedAttrList attributes;

  attributes.set(spirv::SPIRVDialect::getAttributeName(
                     spirv::Decoration::LinkageAttributes),
                 spirv::LinkageAttributesAttr::get(
                     mlirContext, genXName,
                     spirv::LinkageTypeAttr::get(builder.getContext(),
                                                 spirv::LinkageType::Import)));
  attributes.set(spirv::SPIRVDialect::getAttributeName(
                     spirv::Decoration::VectorComputeFunctionINTEL),
                 UnitAttr::get(builder.getContext()));

  llvm::LLVMContext llvmContext;
  auto llvmAttributes = llvm::GenXIntrinsic::getAttributes(llvmContext, id);

  for (auto &attr : llvmAttributes) {
    processPassthroughAttrs(*mlirContext, attributes, attr);
  }

  auto funcOp = spirv::appendOrGetFuncOp(
      mlir::UnknownLoc::get(mlirContext), builder, genXName, funcTy,
      spirv::FunctionControl::Inline, attributes);

  return funcOp;
}

} // namespace intel
} // namespace triton
} // namespace mlir
