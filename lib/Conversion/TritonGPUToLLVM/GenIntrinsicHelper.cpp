//
// Created by chengjun on 9/6/23.
//

#include "Utility.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "GenIntrinsicHelper.h"

namespace mlir {
namespace triton {
namespace intel {

// The code convert the function attribute from the original here:
// https://github.com/llvm/llvm-project/blob/e575b7cb7a64297583d6382c16ce264d9fe45d08/mlir/lib/Target/LLVMIR/ModuleImport.cpp#L1547
void processPassthroughAttrs(mlir::MLIRContext &context,
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

std::string getGenISAName(llvm::GenISAIntrinsic::ID id,
                          ArrayRef<mlir::Type *> mlirTys) {
  SmallVector<llvm::Type *, 4> llvmTys;
  llvm::LLVMContext llvmContext;
  mlir::LLVM::TypeToLLVMIRTranslator typeTranslator(llvmContext);
  for (mlir::Type *ty : mlirTys) {
    auto llvmTy = typeTranslator.translateType(*ty);
    llvmTys.push_back(llvmTy);
  }

  return llvm::GenISAIntrinsic::getName(id, llvmTys);
}

mlir::LLVM::LLVMFunctionType getGenISAType(mlir::MLIRContext &context,
                                 llvm::GenISAIntrinsic::ID id,
                                 ArrayRef<mlir::Type *> mlirTys) {
  SmallVector<llvm::Type *, 4> llvmTys;
  llvm::LLVMContext llvmContext;
  mlir::LLVM::TypeToLLVMIRTranslator llvmToMLIR(llvmContext);
  for (mlir::Type *ty : mlirTys) {
    llvmTys.push_back(llvmToMLIR.translateType(*ty));
  }

  auto llvmFuncType = llvm::GenISAIntrinsic::getType(llvmContext, id, llvmTys);

  LLVM::TypeFromLLVMIRTranslator mlirToLLVM(context);
  auto mlirFuncTy = mlirToLLVM.translateType(llvmFuncType);
  return mlirFuncTy.cast<mlir::LLVM::LLVMFunctionType>();
}


mlir::LLVM::LLVMFuncOp appendOrGetGenISADeclaration(OpBuilder &builder,
                                           llvm::GenISAIntrinsic::ID id,
                                           ArrayRef<mlir::Type *> mlirTys) {
  auto mlirContext = builder.getContext();

  SmallVector<llvm::Type *, 4> llvmTys;
  llvm::LLVMContext llvmContext;
  llvm::Module mod("temp", llvmContext);
  mlir::LLVM::TypeToLLVMIRTranslator llvmToMLIR(llvmContext);
  for (mlir::Type *ty : mlirTys) {
    llvmTys.push_back(llvmToMLIR.translateType(*ty));
  }
  auto llvmFunc = llvm::GenISAIntrinsic::getDeclaration(&mod, id, llvmTys);

  auto genISAName = llvmFunc->getName();
  auto llvmFuncType = llvmFunc->getFunctionType();
  LLVM::TypeFromLLVMIRTranslator mlirFromLLVM(*mlirContext);
  auto mlirFuncTy = mlirFromLLVM.translateType(llvmFuncType);
  mlir::LLVM::LLVMFunctionType funcTy = mlirFuncTy.cast<mlir::LLVM::LLVMFunctionType>();

  NamedAttrList attributes;
  auto llvmAttributes = llvmFunc->getAttributes();

  for (auto &attr : llvmAttributes) {
    processPassthroughAttrs(*mlirContext, attributes, attr);
  }

//  attributes.set(spirv::SPIRVDialect::getAttributeName(
//                     spirv::Decoration::LinkageAttributes),
//                 spirv::LinkageAttributesAttr::get(
//                     mlirContext, genISAName,
//                     spirv::LinkageTypeAttr::get(builder.getContext(),
//                                                 spirv::LinkageType::Import)));

  auto funcAttr = StringAttr::get(mlirContext, genISAName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(builder.getBlock()->getParent()->getParentOfType<mlir::LLVM::LLVMFuncOp>(),
      funcAttr);

  if (funcOp)
    return cast<mlir::LLVM::LLVMFuncOp>(*funcOp);

  auto parent = builder.getBlock()->getParent()->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  mlir::OpBuilder b(parent);
  auto ret = b.create<LLVM::LLVMFuncOp>(
      mlir::UnknownLoc::get(mlirContext), genISAName, funcTy, LLVM::Linkage::External,
      /*dsoLocal*/ false, LLVM::CConv::C, /*comdat=*/SymbolRefAttr{},
      attributes);
  return ret;

}

} // namespace intel
} // namespace triton
} // namespace mlir
