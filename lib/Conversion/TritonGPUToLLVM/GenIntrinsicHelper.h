//
// Created by chengjun on 9/5/23.
//

#ifndef TRITON_VCINTRINSICHELPER_H
#define TRITON_VCINTRINSICHELPER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Value.h"
//#include "triton/Conversion/TritonGPUToSPIRV/ESIMDHelper.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
//#include "llvm/GenXIntrinsics/GenXIntrinsics.h"
// GenISA intrinsic
#include "GenIntrinsics.h"
#include <memory>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <string>
#include "Utility.h"

namespace mlir {
namespace triton {
namespace intel {

/**
VC Intrinsic helper to get the prototype of the vc intrinsic.
SIMT interface: add the SIMT callable interface for SIMT paradigm.
VC Intrinsic project repo:
https://github.com/intel/vc-intrinsics
*/
//std::string getGenXName(llvm::GenXIntrinsic::ID id,
//                        ArrayRef<mlir::Type *> mlirTys);
//
//mlir::FunctionType getGenXType(mlir::MLIRContext &context,
//                               llvm::GenXIntrinsic::ID id,
//                               ArrayRef<mlir::Type *> mlirTys);
//
//spirv::FuncOp appendOrGetGenXDeclaration(OpBuilder &builder,
//                                         llvm::GenXIntrinsic::ID id,
//                                         ArrayRef<mlir::Type *> mlirTys);

mlir::LLVM::LLVMFuncOp appendOrGetGenISADeclaration(OpBuilder &builder,
                                                    llvm::GenISAIntrinsic::ID id,
                                                    ArrayRef<mlir::Type *> mlirTys);

class GenISA_Prefetch {

public:
  explicit GenISA_Prefetch(OpBuilder &builder) : builder(builder) {
    // get GenISA intrinsic declaration.
    intrinsicDecl = appendOrGetGenISADeclaration(builder,
                                                 llvm::GenISAIntrinsic::ID::GenISA_LSC2DBlockPrefetch,
        {});
  }

  template <typename... Args>
  void operator()(OpBuilder &rewriter, Location loc, Args... args) {
    auto funName = intrinsicDecl.getName();
    auto retType = intrinsicDecl.getResultTypes();
    auto funCall = rewriter.create<LLVM::CallOp>(
        loc, retType, funName,
        ValueRange{args...});
    return;
  }

private:
  OpBuilder &builder;
  LLVM::LLVMFuncOp intrinsicDecl;
};

} // namespace intel
} // namespace triton
} // namespace mlir

#endif // TRITON_VCINTRINSICHELPER_H
