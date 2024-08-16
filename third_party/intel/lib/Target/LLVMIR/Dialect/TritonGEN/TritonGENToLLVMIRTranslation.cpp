//===-TritonGENToLLVMIRTranslation.cpp - TritonGEN Dialect to LLVM IR -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the TritonGEN dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "Target/LLVMIR/Dialect/TritonGEN/TritonGENToLLVMIRTranslation.h"

#include "Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"

namespace {
using namespace mlir;
class TritonGENDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    StringRef attrName = attribute.getName().getValue();
    llvm::dbgs() << "\n\nhello from amend: " << attrName << "\n\n";
    if (attrName ==
        triton::TritonGEN::TritonGENDialect::getCacheControlsAttrName()) {
      auto decorationAttr =
          dyn_cast<triton::TritonGEN::DecorationCacheControlAttr>(
              attribute.getValue());
      if (!decorationAttr)
        return op->emitOpError(
            "Expecting triton_gen.decoration_cache_control attribute");
      if (instructions.size() != 1)
        return op->emitOpError("Expecting a single instruction");
      return handleDecorationCacheControl(op, instructions.front(),
                                          decorationAttr);
    }
    if (isKernel(op)) {
      auto spatialExtents = mlir::gpu::lookupSpatialExtents(op);
      llvm::dbgs() << "setting metadata: " << (spatialExtents ? "yes" : "no")
                   << '\n';
      if (spatialExtents) {
        llvm::Function *llvmFunc = moduleTranslation.lookupFunction(
            cast<LLVM::LLVMFuncOp>(op).getName());
        llvm::LLVMContext &llvmContext = moduleTranslation.getLLVMContext();
        auto addMetaData = [llvmFunc, &llvmContext](StringRef name,
                                                    Attribute attribute) {
          SmallVector<llvm::Metadata *, 3> metadata;
          llvm::Type *i64 = llvm::IntegerType::get(llvmContext, 64);
          for (int64_t i : extractFromIntegerArrayAttr<int64_t>(attribute)) {
            llvm::Constant *constant = llvm::ConstantInt::get(i64, i);
            metadata.push_back(llvm::ConstantAsMetadata::get(constant));
          }
          llvm::MDNode *node = llvm::MDNode::get(llvmContext, metadata);
          llvm::dbgs() << "setting md: " << name << '\n';
          llvmFunc->setMetadata(name, node);
        };
        if (spatialExtents.getReqdSubgroupSize()) {
          addMetaData("intel_reqd_sub_group_size",
                      spatialExtents.getReqdSubgroupSize());
        }
        if (spatialExtents.getReqdWorkgroupSize()) {
          addMetaData("reqd_work_group_size",
                      spatialExtents.getReqdWorkgroupSize());
        }
        if (spatialExtents.getMaxWorkgroupSize()) {
          addMetaData("max_work_group_size",
                      spatialExtents.getMaxWorkgroupSize());
        }
      }
    } else {
      auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
      if (fn) {
        llvm::dbgs() << "function not kernel: ";
        fn.dump();
      }
    }
    return success();
  }

private:
  template <typename IntTy>
  static llvm::Metadata *getConstantIntMD(llvm::Type *type, IntTy val) {
    return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(type, val));
  }

  static LogicalResult handleDecorationCacheControl(
      Operation *op, llvm::Instruction *inst,
      triton::TritonGEN::DecorationCacheControlAttr attribute) {
    ArrayRef<Attribute> attrs = attribute.getDecorations();
    SmallVector<llvm::Metadata *> decorations;
    llvm::LLVMContext &ctx = inst->getContext();
    llvm::Type *i32Ty = llvm::IntegerType::getInt32Ty(ctx);
    llvm::transform(
        attrs, std::back_inserter(decorations),
        [&ctx, i32Ty](Attribute attr) -> llvm::Metadata * {
          return TypeSwitch<Attribute, llvm::Metadata *>(attr)
              .Case<triton::TritonGEN::LoadCacheControlDecorationAttr,
                    triton::TritonGEN::StoreCacheControlDecorationAttr>(
                  [&ctx, i32Ty](auto attr) {
                    constexpr size_t decorationCacheControlArity = 4;
                    constexpr uint32_t loadCacheControlKey = 6442;
                    constexpr uint32_t storeCacheControlKey = 6443;
                    constexpr uint32_t decorationKey =
                        std::is_same_v<
                            decltype(attr),
                            triton::TritonGEN::LoadCacheControlDecorationAttr>
                            ? loadCacheControlKey
                            : storeCacheControlKey;
                    std::array<uint32_t, decorationCacheControlArity> values{
                        decorationKey, attr.getCacheLevel(),
                        static_cast<uint32_t>(attr.getCacheControl()),
                        attr.getOperandNumber()};
                    std::array<llvm::Metadata *, decorationCacheControlArity>
                        metadata;
                    llvm::transform(values, metadata.begin(),
                                    [i32Ty](uint32_t value) {
                                      return getConstantIntMD(i32Ty, value);
                                    });
                    return llvm::MDNode::get(ctx, metadata);
                  });
        });
    constexpr llvm::StringLiteral decorationCacheControlMDName =
        "spirv.DecorationCacheControlINTEL";
    inst->setMetadata(decorationCacheControlMDName,
                      llvm::MDNode::get(ctx, decorations));
    return success();
  }

  // Checks if the given operation is a kernel function.
  bool isKernel(Operation *op) const {
    auto fn = dyn_cast<LLVM::LLVMFuncOp>(op);
    return fn && fn.getCConv() == LLVM::CConv::SPIR_KERNEL;
  }
};
} // namespace

namespace mlir {
void registerTritonGENDialectTranslation(DialectRegistry &registry) {
  registry.insert<triton::TritonGEN::TritonGENDialect>();
  registry.addExtension(
      +[](MLIRContext *ctx, triton::TritonGEN::TritonGENDialect *dialect) {
        dialect->addInterfaces<TritonGENDialectLLVMIRTranslationInterface>();
      });
}

void registerTritonGENDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerTritonGENDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
} // namespace mlir
