#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

void Load2DOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
}

static Type getLoadOpResultType(OpBuilder &builder, Type ptrType) {
  auto ptrTensorType = ptrType.dyn_cast<RankedTensorType>();
  if (!ptrTensorType)
    return ptrType.cast<PointerType>().getPointeeType();
  auto shape = ptrTensorType.getShape();
  Type elementType =
      ptrTensorType.getElementType().cast<PointerType>().getPointeeType();
  return RankedTensorType::get(shape, elementType);
}

void Load2DOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                     CacheModifier cache, EvictionPolicy evict, bool isVolatile) {
  return Load2DOp::build(builder, state, ptr, {}, cache, evict, isVolatile);
}

void Load2DOp::build(OpBuilder &builder, OperationState &state, Value ptr,
                     std::optional<PaddingOption> padding,
                     CacheModifier cache, EvictionPolicy evict, bool isVolatile) {
  // Operands
  state.addOperands(ptr);
//  if (mask) {
//    state.addOperands(mask);
//    if (other) {
//      state.addOperands(other);
//    }
//  }

  // Attributes
//  state.addAttribute(
//      getOperandSegmentSizesAttrName(state.name),
//      builder.getDenseI32ArrayAttr({1, (mask ? 1 : 0), (other ? 1 : 0)}));
//  if (boundaryCheck.has_value()) {
//    state.addAttribute(getBoundaryCheckAttrName(state.name),
//                       builder.getDenseI32ArrayAttr(boundaryCheck.value()));
//  }
  if (padding.has_value()) {
    state.addAttribute(
        getPaddingAttrName(state.name),
        PaddingOptionAttr::get(builder.getContext(), padding.value()));
  }
  state.addAttribute(getCacheAttrName(state.name),
                     CacheModifierAttr::get(builder.getContext(), cache));
  state.addAttribute(getEvictAttrName(state.name),
                     EvictionPolicyAttr::get(builder.getContext(), evict));
  state.addAttribute(getIsVolatileAttrName(state.name),
                     builder.getBoolAttr(isVolatile));

  // Result type
  Type resultType = getLoadOpResultType(builder, ptr.getType());
  state.addTypes({resultType});
}

void Load2DOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getPtr(),
                       SideEffects::DefaultResource::get());
  if (getIsVolatile())
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
}

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
