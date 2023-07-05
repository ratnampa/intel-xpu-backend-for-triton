//
// Created by chengjun on 9/6/23.
//

#ifndef TRITON_ESIMDHELPER_H
#define TRITON_ESIMDHELPER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

namespace mlir {
namespace triton {
namespace intel {

mlir::spirv::FuncOp appendOrGetESIMDFunc(OpBuilder &builder,
                                         std::string simdFuncName,
                                         FunctionType simdFuncTy, Location loc,
                                         const NamedAttrList &extraAttrs = {});

}
} // namespace triton
} // namespace mlir

#endif // TRITON_ESIMDHELPER_H
