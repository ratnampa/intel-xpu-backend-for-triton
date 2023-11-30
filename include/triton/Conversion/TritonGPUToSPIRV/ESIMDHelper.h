//
// Created by chengjun on 9/6/23.
//

#ifndef TRITON_ESIMDHELPER_H
#define TRITON_ESIMDHELPER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>

namespace mlir {
namespace triton {
namespace intel {

mlir::spirv::FuncOp appendOrGetESIMDFunc(OpBuilder &builder,
                                         std::string simdFuncName,
                                         FunctionType simdFuncTy, Location loc,
                                         const NamedAttrList &extraAttrs = {});

struct UniformArgType {
  UniformArgType(mlir::Type type) : mlirTy(type) {}

public:
  mlir::Type mlirTy;
};

template <typename> constexpr inline bool checkVectorizeArgType() {
  return true;
}
template <> constexpr inline bool checkVectorizeArgType<UniformArgType>() {
  return false;
}

class ESIMDToSIMTAdaptor {

  void vectorizeArgTypes(SmallVector<mlir::Type> &mlirSIMDTys,
                         SmallVector<mlir::Type> &mlirSIMTTys,
                         SmallVector<mlir::Type> &vectorizedTys) {}

  template <class ARG_TYPE, typename... ARGS_TYPES>
  void vectorizeArgTypes(SmallVector<mlir::Type> &mlirSIMDTys,
                         SmallVector<mlir::Type> &mlirSIMTTys,
                         SmallVector<mlir::Type> &vectorizedTys, ARG_TYPE argTy,
                         ARGS_TYPES... argsTys) {
    if constexpr (checkVectorizeArgType<ARG_TYPE>()) {
      if (auto vecTy = argTy.template dyn_cast<mlir::VectorType>()) {
        auto vecShape = vecTy.getShape();
        assert(vecShape.size() == 1 && "VCIntrinsic only suppor 1 dim now");
        mlir::VectorType simdTy = mlir::VectorType::get(
            vecShape[0] * threadsPerWarp, vecTy.getElementType());
        mlirSIMTTys.push_back(vecTy);
        mlirSIMDTys.push_back(simdTy);
        vectorizedTys.push_back(simdTy);
      } else {
        mlir::VectorType simdTy = mlir::VectorType::get(threadsPerWarp, argTy);
        mlirSIMTTys.push_back(argTy);
        mlirSIMDTys.push_back(simdTy);
        vectorizedTys.push_back(simdTy);
      }
    } else {
      mlirSIMTTys.push_back(argTy.mlirTy);
      mlirSIMDTys.push_back(argTy.mlirTy);
    }
    vectorizeArgTypes(mlirSIMDTys, mlirSIMTTys, vectorizedTys, argsTys...);
  }

public:
  ESIMDToSIMTAdaptor() {}

  template <class... TYPES>
  explicit ESIMDToSIMTAdaptor(OpBuilder &builder, std::string funcName,
                              TYPES... tys) {

    auto mod =
        builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
    threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    auto mlirContext = builder.getContext();

    SmallVector<mlir::Type> mlirSIMTTys;
    SmallVector<mlir::Type> mlirSIMDTys;
    SmallVector<mlir::Type> vectorizedTys;
    vectorizeArgTypes<TYPES...>(mlirSIMDTys, mlirSIMTTys, vectorizedTys,
                                tys...);

    SmallVector<mlir::Type *> vectorizedTysPtr;
    for (mlir::Type &ty : vectorizedTys) {
      vectorizedTysPtr.push_back(&ty);
    }

    // get the ESIMD fucntion.
    SmallVector<mlir::Type, 4> esimdArgTys(mlirSIMDTys.begin() + 1,
                                           mlirSIMDTys.end());
    esimdFunTy =
        mlir::FunctionType::get(mlirContext, esimdArgTys, {mlirSIMDTys[0]});
    esimdWrapper = mlir::triton::intel::appendOrGetESIMDFunc(
        builder, funcName, esimdFunTy, mlir::UnknownLoc::get(mlirContext));

    // Declear SIMT -> SIMD calling interface
    simtIntfName = "_Z33__regcall3____builtin_invoke_simd" + funcName;

    auto esimdFunPtrTy = spirv::FunctionPointerINTELType::get(
        esimdFunTy, spirv::StorageClass::Function);
    SmallVector<mlir::Type, 4> simtArgTys;
    simtArgTys.push_back(esimdFunPtrTy);
    simtArgTys.append(mlirSIMTTys.begin() + 1, mlirSIMTTys.end());
    simtIntfTy =
        mlir::FunctionType::get(mlirContext, {simtArgTys}, {mlirSIMTTys[0]});

    NamedAttrList attributes;
    attributes.set(spirv::SPIRVDialect::getAttributeName(
                       spirv::Decoration::LinkageAttributes),
                   spirv::LinkageAttributesAttr::get(
                       mlirContext, simtIntfName,
                       spirv::LinkageTypeAttr::get(
                           mlirContext, spirv::LinkageType::Import)));

    auto simtWrapper = spirv::appendOrGetFuncOp(
        mlir::UnknownLoc::get(mlirContext), builder, simtIntfName, simtIntfTy,
        spirv::FunctionControl::Inline, attributes);
  }

public:
  spirv::FuncOp esimdWrapper;
  mlir::FunctionType esimdFunTy;
  mlir::FunctionType simtIntfTy;
  std::string simtIntfName;
  int threadsPerWarp;
};

} // namespace intel
} // namespace triton
} // namespace mlir

#endif // TRITON_ESIMDHELPER_H
