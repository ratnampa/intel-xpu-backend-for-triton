#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/IntelUtility.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include <memory>

using namespace mlir;

namespace {

SmallVector<unsigned, 2> warpsPerTileV2(triton::DotOp dotOp,
                                        const ArrayRef<int64_t> shape,
                                        int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  for (Operation *op : slices)
    if (isa<triton::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, 8};
  bool changed = false;
  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

class BlockedToMMA : public mlir::RewritePattern {
  std::map<std::string, int> computeCapability;
  //  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding

public:
  BlockedToMMA(mlir::MLIRContext *context,
               std::map<std::string, int> computeCapability)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<triton::DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        oldRetType.getEncoding()
            .isa<triton::gpu::intel::IntelMmaEncodingAttr>())
      return failure();

    // for FMA, should retain the blocked layout.
    DeviceArch arch = computeCapabilityToXMXArch(computeCapability);
    if (!supportXMX(dotOp, arch))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();

    triton::gpu::intel::IntelMmaEncodingAttr mmaEnc;

    auto warpsPerTile = warpsPerTileV2(dotOp, retShape, numWarps);
    // Hard code for ATSM mnk 8*8*16, packed B on fp16.
    int threadsPerWarp = 32;
    if (computeCapability.find("threads_per_warp") != computeCapability.end()) {
      auto iter = computeCapability.find("threads_per_warp");
      threadsPerWarp = iter->second;
    }
    //    mmaEnc = triton::gpu::intel::IntelMmaEncodingAttr::get(
    //        oldRetType.getContext(), {8, 16}, {16, 8}, {8, 8}, {}, {2, 1}, {},
    //        warpsPerTile, threadsPerWarp);
    mmaEnc = triton::gpu::intel::IntelMmaEncodingAttr::get(
        oldRetType.getContext(), 8, 8, 8, 2, warpsPerTile, threadsPerWarp);

    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);

    unsigned bitwidth = oldAType.getElementType().getIntOrFloatBitWidth();
    unsigned XMXkWidth = 32 / bitwidth;
    auto newAEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(), XMXkWidth);
    auto newBEncoding = triton::gpu::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(), XMXkWidth);

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<triton::DotOp>(
        dotOp.getLoc(), newRetType, a, b, newAcc, dotOp.getAllowTF32(),
        dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    return success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

class TritonIntelGPUAccelerateMatmulPass
    : public TritonIntelGPUAccelerateMatmulBase<
          TritonIntelGPUAccelerateMatmulPass> {
public:
  TritonIntelGPUAccelerateMatmulPass() = default;
  TritonIntelGPUAccelerateMatmulPass(
      std::map<std::string, int> computeCapability) {
    this->computeCapability = std::move(computeCapability);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<::BlockedToMMA>(context, computeCapability);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonIntelGPUAccelerateMatmulPass(
    std::map<std::string, int> computeCapability) {
  return std::make_unique<TritonIntelGPUAccelerateMatmulPass>(
      computeCapability);
}
