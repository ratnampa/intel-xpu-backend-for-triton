//#include "../TritonGPUToSPIRV/Utility.h"
#include "triton/Conversion/IntelGPUToSPIRV/Passes.h"

#include "../TritonGPUToSPIRV/TritonGPUToSPIRVBase.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToSPIRV/VCIntrinsicHelper.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "triton/Conversion/IntelGPUToSPIRV/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::intel;

class ConvertTritonIntelGPUToSPIRV
    : public ConvertTritonIntelGPUToSPIRVBase<ConvertTritonIntelGPUToSPIRV> {

public:
  explicit ConvertTritonIntelGPUToSPIRV(
      std::map<std::string, int> computeCapability) {
    //    this->computeCapability = std::move(computeCapability);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);
#if 0
    spirv::Capability caps_opencl[] = {
        spirv::Capability::Addresses,
        spirv::Capability::Float16Buffer,
        spirv::Capability::Int64,
        spirv::Capability::Int16,
        spirv::Capability::Int8,
        spirv::Capability::Kernel,
        spirv::Capability::Linkage,
        spirv::Capability::Vector16,
        spirv::Capability::GenericPointer,
        spirv::Capability::Groups,
        spirv::Capability::Float16,
        spirv::Capability::Float64,
        spirv::Capability::AtomicFloat32AddEXT,
        spirv::Capability::ExpectAssumeKHR,
    };
    spirv::Extension exts_opencl[] = {
        spirv::Extension::SPV_EXT_shader_atomic_float_add,
        spirv::Extension::SPV_KHR_expect_assume};
    auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_4, caps_opencl,
                                            exts_opencl, context);
    auto targetAttr = spirv::TargetEnvAttr::get(
        triple, spirv::getDefaultResourceLimits(context),
        spirv::ClientAPI::OpenCL, spirv::Vendor::Unknown,
        spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);

    mod->setAttr(spirv::getTargetEnvAttrName(), targetAttr);

    SPIRVConversionOptions options;
    // TODO: need confirm
    options.use64bitIndex = true;
    TritonGPUToSPIRVTypeConverter spirvTypeConverter(targetAttr, options);

    patterns.add<PrefetchOpSPIRVConversion>(spirvTypeConverter, context,
                                                    10);
#endif
#if 0
#define POPULATE_NVGPU_OP(SRC_OP, ASM)                                         \
  patterns.add<NVGPUOpGenericPattern<SRC_OP>>(context, ASM, Constraints(),     \
                                              Constraints());
    POPULATE_NVGPU_OP(ttn::RegAllocOp, Reg_Alloc_Op)
    POPULATE_NVGPU_OP(ttn::WGMMAFenceOp, Wgmma_Fence_Op)
    POPULATE_NVGPU_OP(ttn::CGABarrierSyncOp, Cga_Barrier_Sync_op)
    POPULATE_NVGPU_OP(ttn::WGMMACommitGroupOp, Wgmma_Commit_Group_Op)
    POPULATE_NVGPU_OP(ttn::ClusterWaitOp, Cluster_Wait_Op)
    POPULATE_NVGPU_OP(ttn::FenceMBarrierInitOp, Fence_Mbarrier_Init_Op)
    POPULATE_NVGPU_OP(ttn::CGABarrierArriveOp, Cga_Barrier_Arrive_Op)
    POPULATE_NVGPU_OP(ttn::CGABarrierWaitOp, Cga_Barrier_Wait_Op)
    POPULATE_NVGPU_OP(ttn::RegDeallocOp, Reg_Dealloc_Op)
#undef POPULATE_NVGPU_OP
    patterns.add<NVGPUOpGenericPattern<ttn::MBarrierInitOp>>(
        context, Mbarrier_Init_Op, Constraints(), Constraints({"r", "b"}));
    patterns.add<NVGPUOpGenericPattern<ttn::MBarrierWaitOp>>(
        context, Mbarrier_Wait_Op, Constraints(), Constraints({"r", "r"}));
    patterns.add<NVGPUOpGenericPattern<ttn::NamedBarrierArriveOp>>(
        context, Named_Barrier_Arrive_Op, Constraints(),
        Constraints({"r", "r"}));
    patterns.add<NVGPUOpGenericPattern<ttn::NamedBarrierWaitOp>>(
        context, Named_Barrier_Wait_Op, Constraints(), Constraints({"r", "r"}));
    patterns.add<NVGPUOpGenericPattern<ttn::Sts64Op>>(
        context, Sts64_Op, Constraints(), Constraints({"r", "r", "r"}));
    patterns.add<NVGPUOpGenericPattern<ttn::ClusterCTAIdOp>>(
        context, Cluster_Cta_Id_Op, Constraints({"=r"}), Constraints());
    patterns.add<NVGPUOpGenericPattern<ttn::WGMMADescCreateOp>>(
        context, Wgmma_Desc_Create_op, Constraints({"=l"}),
        Constraints({"l", "l"}));

    patterns.add<FenceAsyncSharedOpPattern, StoreMatrixOpPattern,
                 OffsetOfStmatrixV4OpPattern, MBarrierArriveOpPattern,
                 ClusterArriveOpPattern, TMALoadTiledOpPattern,
                 TMAStoreTiledOpPattern, LoadDSmemOpPattern, WGMMAOpPattern,
                 WGMMAWaitGroupOpPattern, StoreDSmemOpPattern,
                 OffsetOfSts64OpPattern>(context);

#endif
    //    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
    //      signalPassFailure();
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonIntelGPUToSPIRVPass(
    std::map<std::string, int> computeCapability) {
  return std::make_unique<::ConvertTritonIntelGPUToSPIRV>(computeCapability);
}

} // namespace triton
} // namespace mlir
