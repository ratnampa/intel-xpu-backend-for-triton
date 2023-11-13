//
// Created by chengjun on 10/31/23.
//
#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

static Type getI1SameShapeFromTensorOrTensorPtr(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(tensorType.getShape(), i1Type,
                                 tensorType.getEncoding());
  } else if (auto ptrType = type.dyn_cast<triton::PointerType>()) {
    Type pointeeType = ptrType.getPointeeType();
    if (auto tensorType = pointeeType.dyn_cast<RankedTensorType>()) {
      return RankedTensorType::get(tensorType.getShape(), i1Type,
                                   tensorType.getEncoding());
    }
  }
  return Type();
}

template <class OpT>
ParseResult parseInsertSliceOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> allOperands;
  Type srcType, dstType;
  SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(srcType) || parser.parseArrow() ||
      parser.parseCustomTypeWithFallback(dstType))
    return failure();
  result.addTypes(dstType);

  SmallVector<Type> operandTypes;
  operandTypes.push_back(srcType); // src
  operandTypes.push_back(dstType); // dst
  operandTypes.push_back(
      IntegerType::get(parser.getBuilder().getContext(), 32)); // index

  int hasMask = 0, hasOther = 0;
  if (allOperands.size() >= 4) {
    operandTypes.push_back(
        getI1SameShapeFromTensorOrTensorPtr(srcType)); // mask
    hasMask = 1;
  }
  if (allOperands.size() >= 5) {
    operandTypes.push_back(triton::getPointeeType(srcType)); // other
    hasOther = 1;
  }

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();

  // Deduce operandSegmentSizes from the number of the operands.
  auto operandSegmentSizesAttrName =
      OpT::getOperandSegmentSizesAttrName(result.name);
  result.addAttribute(
      operandSegmentSizesAttrName,
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, hasMask, hasOther}));
  return success();
}

template <class OpT>
void printInsertSliceOp(OpAsmPrinter &printer, OpT insertSliceOp) {
  printer << " ";
  printer << insertSliceOp.getOperation()->getOperands();
  // "operandSegmentSizes" can be deduced, so we don't print it.
  printer.printOptionalAttrDict(
      insertSliceOp->getAttrs(),
      {insertSliceOp.getOperandSegmentSizesAttrName()});
  printer << " : ";
  printer.printStrippedAttrOrType(insertSliceOp.getSrc().getType());
  printer << " -> ";
  printer.printStrippedAttrOrType(insertSliceOp.getDst().getType());
}

void PrefetchCacheOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getPtr(),
                       SideEffects::DefaultResource::get());
  //  if (getIsVolatile())
  //    effects.emplace_back(MemoryEffects::Write::get(),
  //                         SideEffects::DefaultResource::get());
}

ParseResult PrefetchCacheOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> allOperands;
  Type srcType;
  SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) || parser.parseColon() ||
      parser.parseCustomTypeWithFallback(srcType))
    return failure();

  SmallVector<Type> operandTypes;
  operandTypes.push_back(srcType); // src

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();

  return success();
}

void PrefetchCacheOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer << getOperation()->getOperands();
  printer << " : ";
  printer.printStrippedAttrOrType(getPtr().getType());
  //  printInsertSliceOp<PrefetchCacheOp>(printer, *this);
}

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
