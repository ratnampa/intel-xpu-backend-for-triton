#if 0
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#endif
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/IntelUtility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include <memory>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

using namespace mlir;

class TritonIntelGPUDecomposeConversionsPass
    : public TritonIntelGPUDecomposeConversionsBase<
          TritonIntelGPUDecomposeConversionsPass> {
public:
  TritonIntelGPUDecomposeConversionsPass() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      // only used for JointMatrix. mma -> blocked. to mma -> shared -> blocked.
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcEncoding = srcType.getEncoding();
      auto dstEncoding = dstType.getEncoding();
      if (auto srcMmaEncoding =
              srcEncoding
                  .dyn_cast<triton::gpu::intel::IntelMmaEncodingAttr>()) {
        if (auto dstBlockedEncoding =
                dstEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>()) {
          llvm::outs() << "johnlu here1111\n";
          auto tmpType = RankedTensorType::get(
              dstType.getShape(), dstType.getElementType(),
              triton::gpu::SharedEncodingAttr::get(
                  mod.getContext(), 1, 1, 1, triton::gpu::getOrder(srcEncoding),
                  triton::gpu::getCTALayout(srcEncoding), false));
          auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
              cvtOp.getLoc(), tmpType, cvtOp.getOperand());
          auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
              cvtOp.getLoc(), dstType, tmp);
          cvtOp.replaceAllUsesWith(newConvert.getResult());
          cvtOp.erase();
        }
      }
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonIntelGPUDecomposeConversionsPass(
    std::map<std::string, int> computeCapability) {
  return std::make_unique<TritonIntelGPUDecomposeConversionsPass>();
}
