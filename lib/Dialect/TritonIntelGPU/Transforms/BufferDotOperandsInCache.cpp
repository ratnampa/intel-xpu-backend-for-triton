//
// Created by chengjun on 11/9/23.
//
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

class TritonIntelGPUBufferDotOperandsInCacheConversionsPass
    : public TritonIntelGPUBufferDotOperandsInCacheConversionsBase<
          TritonIntelGPUBufferDotOperandsInCacheConversionsPass> {
public:
  TritonIntelGPUBufferDotOperandsInCacheConversionsPass() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::InsertSliceAsyncOp isrtOp) -> void {
      llvm::outs() << "johnlu get the insrt op:" << isrtOp << "\n";
      llvm::outs().flush();
      //      OpBuilder builder(isrtOp);
      //      auto prefOp = builder.create<triton::gpu::intel::PrefetchCacheOp>(
      //          isrtOp.getLoc(), isrtOp.getType(), isrtOp.getSrc(),
      //          isrtOp.getResult(), isrtOp.getIndex(), isrtOp.getMask(),
      //          isrtOp.getOther(), isrtOp.getCache(),
      //          isrtOp.getEvict(), isrtOp.getIsVolatile(),
      //          isrtOp.getAxis());
      //      llvm::outs() << "johnlu get the prefOp op:" << prefOp << "\n";
      //      llvm::outs().flush();
      //      isrtOp.replaceAllUsesWith(prefOp.getResult());
      //      isrtOp.erase();
      //      auto srcType =
      //      cvtOp.getOperand().getType().cast<RankedTensorType>(); auto
      //      dstType = cvtOp.getType().cast<RankedTensorType>(); auto
      //      srcEncoding = srcType.getEncoding(); auto dstEncoding =
      //      dstType.getEncoding(); if (auto srcMmaEncoding =
      //              srcEncoding
      //                  .dyn_cast<triton::gpu::intel::IntelMmaEncodingAttr>())
      //                  {
      //        if (auto dstBlockedEncoding =
      //                dstEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>())
      //                {
      //          auto tmpType = RankedTensorType::get(
      //              dstType.getShape(), dstType.getElementType(),
      //              triton::gpu::SharedEncodingAttr::get(
      //                  mod.getContext(), 1, 1, 1,
      //                  triton::gpu::getOrder(srcEncoding),
      //                  triton::gpu::getCTALayout(srcEncoding), false));
      //          auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
      //              cvtOp.getLoc(), tmpType, cvtOp.getOperand());
      //          auto newConvert =
      //          builder.create<triton::gpu::ConvertLayoutOp>(
      //              cvtOp.getLoc(), dstType, tmp);
      //          cvtOp.replaceAllUsesWith(newConvert.getResult());
      //          cvtOp.erase();
      //        }
      //      }
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonIntelGPUBufferDotOperandsInCachePass(
    std::map<std::string, int> computeCapability) {
  return std::make_unique<
      TritonIntelGPUBufferDotOperandsInCacheConversionsPass>();
}
