//
// Created by chengjun on 9/6/23.
//

#include "Utility.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "triton/Conversion/TritonGPUToSPIRV/VCIntrinsicHelper.h"

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

static llvm::GenISAIntrinsic::ID fixGenISAID(llvm::GenISAIntrinsic::ID id) {
  //  auto idBase = llvm::GenISAIntrinsic::getGenIntrinsicIDBase();
  //  return (llvm::GenISAIntrinsic::ID)(id - llvm::Intrinsic::num_intrinsics +
  //                                     idBase);
  return id;
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

/// Mapping between SPIR-V storage classes to Triton memory spaces.
///
#define STORAGE_SPACE_MAP_LIST(MAP_FN)                                         \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 1)                               \
  MAP_FN(spirv::StorageClass::Workgroup, 3)

#if 0
MAP_FN(spirv::StorageClass::StorageBuffer, 0)                                \
  MAP_FN(spirv::StorageClass::Uniform, 4)                                      \
  MAP_FN(spirv::StorageClass::Private, 5)                                      \
  MAP_FN(spirv::StorageClass::Function, 6)                                     \
  MAP_FN(spirv::StorageClass::PushConstant, 7)                                 \
  MAP_FN(spirv::StorageClass::UniformConstant, 8)                              \
  MAP_FN(spirv::StorageClass::Input, 9)                                        \
  MAP_FN(spirv::StorageClass::Output, 10)                                      \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 11)                              \
  MAP_FN(spirv::StorageClass::AtomicCounter, 12)                               \
  MAP_FN(spirv::StorageClass::Image, 13)                                       \
  MAP_FN(spirv::StorageClass::CallableDataKHR, 14)                             \
  MAP_FN(spirv::StorageClass::IncomingCallableDataKHR, 15)                     \
  MAP_FN(spirv::StorageClass::RayPayloadKHR, 16)                               \
  MAP_FN(spirv::StorageClass::HitAttributeKHR, 17)                             \
  MAP_FN(spirv::StorageClass::IncomingRayPayloadKHR, 18)                       \
  MAP_FN(spirv::StorageClass::ShaderRecordBufferKHR, 19)                       \
  MAP_FN(spirv::StorageClass::PhysicalStorageBuffer, 20)                       \
  MAP_FN(spirv::StorageClass::CodeSectionINTEL, 21)                            \
  MAP_FN(spirv::StorageClass::DeviceOnlyINTEL, 22)                             \
  MAP_FN(spirv::StorageClass::HostOnlyINTEL, 23)
#endif

std::optional<spirv::StorageClass> static getStorageClassForMemorySpace(
    unsigned space) {
#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case space:                                                                  \
    return storage;

  switch (space) {
    STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN)
  default:
    return std::nullopt;
  }
#undef STORAGE_SPACE_MAP_FN
}

mlir::FunctionType getGenISAType(mlir::MLIRContext &context,
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
  auto mlirType = mlirToLLVM.translateType(llvmFuncType);
  auto mlirLLVMFuncType = mlirType.cast<mlir::LLVM::LLVMFunctionType>();
  SmallVector<mlir::Type> spirvTys;
  auto inputTys = mlirLLVMFuncType.getParams();
  for (auto &ty : inputTys) {
    if (ty.isa<LLVM::LLVMPointerType>()) {
      auto ptrTy = ty.cast<LLVM::LLVMPointerType>();
      std::optional<spirv::StorageClass> storageClass =
          getStorageClassForMemorySpace(ptrTy.getAddressSpace());
      spirv::PointerType spirvBasePtrType = spirv::PointerType::get(
          mlir::IntegerType::get(&context, 8), *storageClass);
      spirvTys.push_back(spirvBasePtrType);
    } else {
      spirvTys.push_back(ty);
    }
  }

  SmallVector<mlir::Type> spirvRetTys;
  auto retTys = mlirLLVMFuncType.getReturnTypes();

  for (auto &ty : retTys) {
    if (ty.isa<LLVM::LLVMVoidType>()) {
      continue;
    } else {
      spirvRetTys.push_back(ty);
    }
  }

  auto funcTy = mlir::FunctionType::get(&context, spirvTys, spirvRetTys);
  return funcTy;
}

spirv::FuncOp appendOrGetGenISADeclaration(OpBuilder &builder,
                                           llvm::GenISAIntrinsic::ID id,
                                           ArrayRef<mlir::Type *> mlirTys) {
  auto mlirContext = builder.getContext();

  auto genISAName = getGenISAName(id, mlirTys);
  FunctionType funcTy = getGenISAType(*mlirContext, id, mlirTys);

  NamedAttrList attributes;

  attributes.set(spirv::SPIRVDialect::getAttributeName(
                     spirv::Decoration::LinkageAttributes),
                 spirv::LinkageAttributesAttr::get(
                     mlirContext, genISAName,
                     spirv::LinkageTypeAttr::get(builder.getContext(),
                                                 spirv::LinkageType::Import)));

  llvm::LLVMContext llvmContext;
  auto llvmAttributes = llvm::GenISAIntrinsic::getGenIntrinsicAttributes(
      llvmContext, fixGenISAID(id));

  for (auto &attr : llvmAttributes) {
    processPassthroughAttrs(*mlirContext, attributes, attr);
  }

  auto funcOp = spirv::appendOrGetFuncOp(
      mlir::UnknownLoc::get(mlirContext), builder, genISAName, funcTy,
      spirv::FunctionControl::Inline, attributes);

  return funcOp;
}

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
