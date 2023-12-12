#include "DotOpToSPIRV.h"
#include "Utility.h"
#include "triton/Analysis/IntelUtility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToSPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter, Value tid);

LogicalResult convertXMXDot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToSPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter, Value tid);

struct DotOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::DotOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::DotOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShape = A.getType().cast<RankedTensorType>().getShape();
    size_t reduceAxis = 1;
    unsigned K = AShape[reduceAxis];
    bool isOuter = K == 1;

    auto mmaLayout = D.getType()
                         .cast<RankedTensorType>()
                         .getEncoding()
                         .dyn_cast<triton::gpu::intel::IntelMmaEncodingAttr>();
    auto loc = op.getLoc();
    if (!isOuter && mmaLayout) {
      return convertXMXDot(op, adaptor, getTypeConverter(), rewriter,
                           tid_val());
    }

    if (D.getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<BlockedEncodingAttr>())
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter,
                           tid_val());

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to SPIRV.");
  }
};

void populateDotOpToSPIRVPatterns(TritonGPUToSPIRVTypeConverter &typeConverter,
                                  mlir::MLIRContext *context,
                                  RewritePatternSet &patterns,
                                  ModuleAllocation &allocation,
                                  PatternBenefit benefit) {
  patterns.add<DotOpSPIRVConversion>(typeConverter, context, allocation,
                                     benefit);
}
