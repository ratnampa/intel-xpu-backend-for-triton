#include "../DotOpToSPIRV.h"
#include "../Utility.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MmaEncodingAttr;

using ValueTableFMA = std::map<std::pair<int, int>, Value>;

static ValueTableFMA getValueTableFromStructFMA(
    Value val, int K, int n0, int shapePerCTATile, int sizePerThread,
    ConversionPatternRewriter &rewriter, Location loc,
    TritonGPUToSPIRVTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = typeConverter->unpackLLElements(loc, val, rewriter, type);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTATile)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToSPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter, Value tid) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = A.getType().cast<RankedTensorType>();
  auto bTensorTy = B.getType().cast<RankedTensorType>();
  auto dTensorTy = D.getType().cast<RankedTensorType>();

  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  auto bShapePerCTA = getShapePerCTA(bTensorTy);

  BlockedEncodingAttr dLayout =
      dTensorTy.getEncoding().cast<BlockedEncodingAttr>();
  auto order = dLayout.getOrder();
  auto cc =
      typeConverter->unpackLLElements(loc, adaptor.getC(), rewriter, dTensorTy);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTATile = getShapePerCTATile(dLayout);

  int K = aShapePerCTA[1];
  int M = aShapePerCTA[0];
  int N = bShapePerCTA[1];

  int mShapePerCTATile =
      order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nShapePerCTATile =
      order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];

  auto has =
      getValueTableFromStructFMA(llA, K, M, mShapePerCTATile, mSizePerThread,
                                 rewriter, loc, typeConverter, aTensorTy);
  auto hbs =
      getValueTableFromStructFMA(llB, K, N, nShapePerCTATile, nSizePerThread,
                                 rewriter, loc, typeConverter, bTensorTy);

  SmallVector<Value> ret = cc;
  bool isCRow = order[0] == 1;

#if 0
  std::string printFunName;
  printFunName = "print_mm";
  auto printFuncTy = mlir::FunctionType::get(
      rewriter.getContext(), {i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, f32_ty, f32_ty, f32_ty}, TypeRange());

  NamedAttrList attributes;
  attributes.set("libname", StringAttr::get(rewriter.getContext(), "libdevice"));
  attributes.set("libpath", StringAttr::get(rewriter.getContext(), ""));
  auto linkageTypeAttr =
      rewriter.getAttr<::mlir::spirv::LinkageTypeAttr>(spirv::LinkageType::Import);
  auto linkageAttr = rewriter.getAttr<::mlir::spirv::LinkageAttributesAttr>(
      printFunName, linkageTypeAttr);
  attributes.set("linkage_attributes", linkageAttr);
  spirv::appendOrGetFuncOp(loc, rewriter, printFunName, printFuncTy,
                           spirv::FunctionControl::Inline, attributes);

  auto mod = op->getParentOfType<ModuleOp>();
  unsigned threadsPerWarp =
      triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warp = udiv(tid, i32_val(threadsPerWarp));
  Value lane = urem(tid, i32_val(threadsPerWarp));

  llvm::outs() << "johnlu op:" << op << "\n";

  llvm::outs() << "johnlu sizePerThread:" << "\n";
  for (auto &i : sizePerThread)
    llvm::outs() << " " << i;
  llvm::outs() << "\n";

  llvm::outs() << "johnlu shapePerCTATile:" << "\n";
  for (auto &i : shapePerCTATile)
    llvm::outs() << " " << i;
  llvm::outs() << "\n";
  llvm::outs() << "johnlu operands a coordi:" << "\n";
  for (auto &i : has)
    llvm::outs() << " m:" << i.first.first << " n:" <<i.first.second << "\n";
  llvm::outs() << "\n";

  llvm::outs() << "johnlu M:" << M << "\n";
  llvm::outs() << "johnlu N:" << N << "\n";
  llvm::outs() << "johnlu K:" << K << "\n";
  llvm::outs() << "johnlu mShapePerCTATile:" << mShapePerCTATile << "\n";
  llvm::outs() << "johnlu nShapePerCTATile:" << nShapePerCTATile << "\n";
  llvm::outs() << "johnlu mSizePerThread:" << mSizePerThread << "\n";
  llvm::outs() << "johnlu nSizePerThread:" << nSizePerThread << "\n";
  llvm::outs().flush();
#endif

#if 0

  for (auto &i : has) {
    rewriter.create<spirv::FunctionCallOp>(
        loc, TypeRange(), printFunName,
        ValueRange{warp, lane, i32_val(i.first.first), i32_val(i.first.second), i32_val(0),
                   i.second,
                   i.second, i.second});
  }
#endif

  for (unsigned k = 0; k < K; k++) {
    for (unsigned m = 0; m < M; m += mShapePerCTATile)
      for (unsigned n = 0; n < N; n += nShapePerCTATile)
        for (unsigned mm = 0; mm < mSizePerThread; ++mm)
          for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
            int mIdx = m / mShapePerCTATile * mSizePerThread + mm;
            int nIdx = n / nShapePerCTATile * nSizePerThread + nn;

            int z = isCRow
                        ? mIdx * N / nShapePerCTATile * mSizePerThread + nIdx
                        : nIdx * M / mShapePerCTATile * nSizePerThread + mIdx;
            ret[z] = rewriter.create<spirv::CLFmaOp>(loc, has[{m + mm, k}],
                                                     hbs[{n + nn, k}], ret[z]);
#if 0

#if 0
            // Create block structure for the masked memory copy.
            auto *preheader = rewriter.getInsertionBlock();
            auto opPosition = rewriter.getInsertionPoint();
            auto *tailblock = rewriter.splitBlock(preheader, opPosition);
            auto *condblock = rewriter.createBlock(tailblock);

            // Test the mask
            rewriter.setInsertionPoint(preheader, preheader->end());
            rewriter.create<mlir::cf::CondBranchOp>(
                loc, icmp_eq(warp, i32_val(0)), condblock, tailblock);

            rewriter.setInsertionPoint(condblock, condblock->end());
#endif
            // Do the print
            rewriter.create<spirv::FunctionCallOp>(
                loc, TypeRange(), printFunName,
                ValueRange{warp, lane, i32_val(m + mm), i32_val(n + nn), i32_val(k),
                           has[{m + mm, k}],
                           hbs[{n + nn, k}], ret[z]});
#if 0
            rewriter.create<mlir::cf::BranchOp>(loc, tailblock);
            // The memory copy insert position
            rewriter.setInsertionPoint(tailblock, tailblock->begin());
#endif

#endif
          }
  }

#if 0
  for (unsigned m = 0; m < M; m += mShapePerCTATile)
    for (unsigned n = 0; n < N; n += nShapePerCTATile)
      for (unsigned mm = 0; mm < mSizePerThread; ++mm)
        for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
          int mIdx = m / mShapePerCTATile * mSizePerThread + mm;
          int nIdx = n / nShapePerCTATile * nSizePerThread + nn;

          int z = isCRow
                      ? mIdx * N / nShapePerCTATile * mSizePerThread + nIdx
                      : nIdx * M / mShapePerCTATile * nSizePerThread + mIdx;

          rewriter.create<spirv::FunctionCallOp>(
              loc, TypeRange(), "print_mm",
              ValueRange{warp, lane, i32_val(m + mm), i32_val(n + nn), i32_val(z), ret[z]});
        }
#endif

  auto res = typeConverter->packLLElements(loc, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
