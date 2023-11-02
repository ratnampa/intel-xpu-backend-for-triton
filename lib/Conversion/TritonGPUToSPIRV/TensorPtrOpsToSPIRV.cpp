
#include "TensorPtrOpsToSPIRV.h"
using namespace mlir;
using namespace mlir::triton;

struct MakeTensorPtrOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::MakeTensorPtrOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::MakeTensorPtrOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto offsets = adaptor.getOffsets();
    auto shapes = adaptor.getShape();
    auto strides = adaptor.getStrides();
    auto base = adaptor.getBase();
    auto result = op.getResult();

    SmallVector<Value> elems;
    for (auto offset : offsets)
      elems.push_back(offset);
    for (auto shape : shapes)
      elems.push_back(shape);
    for (auto stride : strides)
      elems.push_back(stride);

    elems.push_back(base);

    auto newValue = getTypeConverter()->packLLElements(
        op.getLoc(), elems, rewriter, result.getType());
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

struct AdvanceOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::AdvanceOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::AdvanceOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto loc = op.getLoc();
    auto ptrType = op.getPtr().getType();
    auto tensorPtr = adaptor.getPtr();

    auto offsets = adaptor.getOffsets();
    auto elems =
        getTypeConverter()->unpackLLElements(loc, tensorPtr, rewriter, ptrType);

    SmallVector<Value, 2> newOffsets;

    for (auto [offset, oldOffset] : llvm::zip_first(offsets, elems)) {
      newOffsets.push_back((add(offset, oldOffset)));
    }

    for (size_t i = 0; i < newOffsets.size(); ++i) {
      elems[i] = newOffsets[i];
    }

    auto newValue = getTypeConverter()->packLLElements(op.getLoc(), elems,
                                                       rewriter, ptrType);
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

void populateTensorPtrOpsToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<AdvanceOpSPIRVConversion>(typeConverter, context, benefit);
  return;
}
