//
// Created by chengjun on 9/5/23.
//

#ifndef TRITON_VCINTRINSICHELPER_H
#define TRITON_VCINTRINSICHELPER_H

//#include "Utility.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToSPIRV/ESIMDHelper.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/GenXIntrinsics/GenXIntrinsics.h"
// GenISA intrinsic
#include "GenIntrinsics.h"
#include <memory>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <string>

namespace mlir {
namespace triton {
namespace intel {

/**
VC Intrinsic helper to get the prototype of the vc intrinsic.
SIMT interface: add the SIMT callable interface for SIMT paradigm.
VC Intrinsic project repo:
https://github.com/intel/vc-intrinsics
*/
std::string getGenXName(llvm::GenXIntrinsic::ID id,
                        ArrayRef<mlir::Type *> mlirTys);

mlir::FunctionType getGenXType(mlir::MLIRContext &context,
                               llvm::GenXIntrinsic::ID id,
                               ArrayRef<mlir::Type *> mlirTys);

spirv::FuncOp appendOrGetGenXDeclaration(OpBuilder &builder,
                                         llvm::GenXIntrinsic::ID id,
                                         ArrayRef<mlir::Type *> mlirTys);

std::string getGenISAName(llvm::GenISAIntrinsic::ID id,
                          ArrayRef<mlir::Type *> mlirTys);

spirv::FuncOp appendOrGetGenISADeclaration(OpBuilder &builder,
                                           llvm::GenISAIntrinsic::ID id,
                                           ArrayRef<mlir::Type *> mlirTys);

mlir::FunctionType getGenISAType(mlir::MLIRContext &context,
                                 llvm::GenISAIntrinsic::ID id,
                                 ArrayRef<mlir::Type *> mlirTys);

void processPassthroughAttrs(mlir::MLIRContext &context,
                             NamedAttrList &attributes,
                             const llvm::AttributeSet &funcAttrs);

class IntrinsicCommon;
class VCIntrinsic;

class IntrinsicBuilder {
#if 0
  struct Operand {
    std::string constraint;
    Value value;
    int idx{-1};
    llvm::SmallVector<Operand *> list;
    std::function<std::string(int idx)> repr;

    // for list
    Operand() = default;
    Operand(const Operation &) = delete;
    Operand(Value value, StringRef constraint)
        : value(value), constraint(constraint) {}

    bool isList() const { return !value && constraint.empty(); }

    Operand *listAppend(Operand *arg) {
      list.push_back(arg);
      return this;
    }

    Operand *listGet(size_t nth) const {
      assert(nth < list.size());
      return list[nth];
    }

    std::string dump() const;
  };

  struct Modifier {
    Value value;
    std::string modifier;
    std::string arg;
    llvm::SmallVector<Modifier *> list;

    Modifier() = default;
    Modifier(const Operation &) = delete;
    Modifier(Value value, StringRef arg) : value(value), arg(arg) {}

    bool isList() const { return !value && modifier.empty(); }

    Modifier *listAppend(Modifier *arg) {
      list.push_back(arg);
      return this;
    }

    Modifier *listGet(size_t index) const {
      assert(index < list.size());
      return list[index];
    }

    std::string to_str() const {
      std::string str = modifier;
      if (!arg.empty()) {
        str += ":" + arg;
      }
      return str;
    }

    std::string dump() const;
  };
#endif
public:
  explicit IntrinsicBuilder(int threadsPerWarp, OpBuilder &builder)
      : threadsPerWarp(threadsPerWarp), builder(builder){};

  template <typename INSTR = VCIntrinsic, typename... Args>
  INSTR *create(Args &&...args) {
    instrs.emplace_back(std::make_unique<INSTR>(this, args...));
    return static_cast<INSTR *>(instrs.back().get());
  }
#if 0
  template <typename INSTR = GCNInstr, typename... Args>
  INSTR *create(Args &&...args) {
    instrs.emplace_back(std::make_unique<INSTR>(this, args...));
    return static_cast<INSTR *>(instrs.back().get());
  }

  // Create a list of operands.
  Operand *newListOperand() { return newOperand(); }

  Operand *newListOperand(ArrayRef<std::pair<mlir::Value, std::string>> items) {
    auto *list = newOperand();
    for (auto &item : items) {
      list->listAppend(newOperand(item.first, item.second));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, mlir::Value val,
                          const std::string &constraint) {
    auto *list = newOperand();
    for (int i = 0; i < count; ++i) {
      list->listAppend(newOperand(val, constraint));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, const std::string &constraint) {
    auto *list = newOperand();
    for (int i = 0; i < count; ++i) {
      list->listAppend(newOperand(constraint));
    }
    return list;
  }

  // Create a new operand. It will not add to operand list.
  // @value: the MLIR value bind to this operand.
  // @constraint: ASM operand constraint, .e.g. "=r"
  // @formatter: extra format to represent this operand in ASM code, default is
  //             "%{0}".format(operand.idx).
  Operand *newOperand(mlir::Value value, StringRef constraint,
                      std::function<std::string(int idx)> formatter = nullptr);

  // Create a new operand which is written to, that is, the constraint starts
  // with "=", e.g. "=r".
  Operand *newOperand(StringRef constraint);

  // Create a constant integer operand.
  Operand *newConstantOperand(int v);
  // Create a constant operand with explicit code specified.
  Operand *newConstantOperand(const std::string &v);

  Operand *newAddrOperand(mlir::Value addr, StringRef constraint);

  Modifier *newModifier(StringRef modifier, StringRef arg);

  llvm::SmallVector<Operand *, 4> getAllArgs() const;

  llvm::SmallVector<Value, 4> getAllMLIRArgs() const;

  std::string getConstraints() const;

  std::string dump() const;

  mlir::Value launch(ConversionPatternRewriter &rewriter, Location loc,
                     Type resTy, bool hasSideEffect = true,
                     bool isAlignStack = false,
                     ArrayRef<Attribute> attrs = {}) const;

private:
  Operand *newOperand() {
    argArchive.emplace_back(std::make_unique<Operand>());
    return argArchive.back().get();
  }

  Modifier *newModifier() {
    modArchive.emplace_back(std::make_unique<Modifier>());
    return modArchive.back().get();
  }

#endif
  //  friend class GCNInstr;
  //  friend class GCNInstrCommon;
public:
  llvm::SmallVector<std::unique_ptr<IntrinsicCommon>, 2> instrs;
  //  int oprCounter{};
  OpBuilder &builder;
  int threadsPerWarp;
};
class IntrinsicBuilder;

class IntrinsicCommon {

public:
  IntrinsicCommon(IntrinsicBuilder *builder) {}

private:
};

template <class ConcreteT>
struct VCIntrinsicSIMTAdaptor : public IntrinsicCommon {

  void vectorizeArgTypes(SmallVector<mlir::Type> &mlirSIMDTys,
                         SmallVector<mlir::Type> &mlirSIMTTys,
                         SmallVector<mlir::Type> &vectorizedTys) {}

  template <class ARG_TYPE, typename... ARGS_TYPES>
  void vectorizeArgTypes(SmallVector<mlir::Type> &mlirSIMDTys,
                         SmallVector<mlir::Type> &mlirSIMTTys,
                         SmallVector<mlir::Type> &vectorizedTys, ARG_TYPE argTy,
                         ARGS_TYPES... argsTys) {
    if constexpr (checkVectorizeArgType<ARG_TYPE>()) {
      if (auto vecTy = argTy.template dyn_cast<mlir::VectorType>()) {
        auto vecShape = vecTy.getShape();
        assert(vecShape.size() == 1 && "VCIntrinsic only suppor 1 dim now");
        mlir::VectorType simdTy = mlir::VectorType::get(
            vecShape[0] * threadsPerWarp, vecTy.getElementType());
        mlirSIMTTys.push_back(vecTy);
        mlirSIMDTys.push_back(simdTy);
        vectorizedTys.push_back(simdTy);
      } else {
        llvm_unreachable("un-supported type of SIMT wrapper");
      }
    } else {
      mlirSIMTTys.push_back(argTy.mlirTy);
      mlirSIMDTys.push_back(argTy.mlirTy);
    }
    vectorizeArgTypes(mlirSIMDTys, mlirSIMTTys, vectorizedTys, argsTys...);
  }

  template <class... TYPES>
  explicit VCIntrinsicSIMTAdaptor(IntrinsicBuilder *builder,
                                  llvm::GenXIntrinsic::ID intrinsic,
                                  std::string suffix, TYPES... tys)
      : IntrinsicCommon(builder), threadsPerWarp(builder->threadsPerWarp) {
    auto mlirContext = builder->builder.getContext();

    SmallVector<mlir::Type> mlirSIMTTys;
    SmallVector<mlir::Type> mlirSIMDTys;
    SmallVector<mlir::Type> vectorizedTys;
    vectorizeArgTypes<TYPES...>(mlirSIMDTys, mlirSIMTTys, vectorizedTys,
                                tys...);

    SmallVector<mlir::Type *> vectorizedTysPtr;
    for (mlir::Type &ty : vectorizedTys) {
      vectorizedTysPtr.push_back(&ty);
    }

    // get GenX intrinsic declaration.
    vcIntrinsicDecl = appendOrGetGenXDeclaration(builder->builder, intrinsic,
                                                 vectorizedTysPtr);

    // get the ESIMD wrapper.
    auto genXName =
        mlir::triton::intel::getGenXName(intrinsic, vectorizedTysPtr);
    std::string esimdWrapperName =
        "VCIntrinsicWrapper_" + genXName + "_" + suffix;

    esimdFunc = ESIMDToSIMTAdaptor(builder->builder, esimdWrapperName, tys...);
    spirv::FuncOp &esimdWrapper = esimdFunc.esimdWrapper;
  }

protected:
  ESIMDToSIMTAdaptor esimdFunc;
  spirv::FuncOp vcIntrinsicDecl;
  int threadsPerWarp;
};

template <class ConcreteT> struct GenISAIntrinsic : public IntrinsicCommon {

  template <class... TYPES>
  explicit GenISAIntrinsic(IntrinsicBuilder *builder,
                           llvm::GenISAIntrinsic::ID intrinsic, TYPES... tys)
      : IntrinsicCommon(builder) {
    SmallVector<mlir::Type *> mlirTys{&tys...};
    // get GenISA intrinsic declaration.
    intrinsicDecl =
        appendOrGetGenISADeclaration(builder->builder, intrinsic, mlirTys);
  }

protected:
  spirv::FuncOp intrinsicDecl;
};

// data type for D_C_A_B.
enum class DPASEngineType : uint8_t {
  // floating-point XMX engine instr
  FP32_FP32_FP16_FP16 = 0, // default
  FP32_FP32_BF16_BF16,
  FP32_FP32_TF32_TF32,
  FP16_FP16_FP16_FP16,
  BF16_BF16_BF16_BF16,
  // integer XMX engine instr
  // TODO: add integer support
  //
  NOT_APPLICABLE,
};

// refer to IGC/visa/Common_ISA_util.cpp#87
static inline uint8_t encodePrecision(DPASEngineType type) {
  if (type == DPASEngineType::FP32_FP32_BF16_BF16 ||
      type == DPASEngineType::BF16_BF16_BF16_BF16)
    return 9;
  else if (type == DPASEngineType::FP32_FP32_FP16_FP16 ||
           type == DPASEngineType::FP16_FP16_FP16_FP16)
    return 10;
  else if (type == DPASEngineType::FP32_FP32_TF32_TF32)
    return 12;
  else {
    assert(0 && "add more support");
    return 0;
  }
};

class GenISA_DPAS : public GenISAIntrinsic<GenISA_DPAS> {

public:
  explicit GenISA_DPAS(IntrinsicBuilder *builder, DPASEngineType dpasTy,
                       uint32_t systolicDepth, uint32_t repeatCount,
                       mlir::Type dTy, mlir::Type cTy, mlir::Type aTy,
                       mlir::Type bTy)
      : GenISAIntrinsic<GenISA_DPAS>(
            builder, llvm::GenISAIntrinsic::ID::GenISA_sub_group_dpas, dTy, cTy,
            aTy, bTy),
        dpasTy(dpasTy), systolicDepth(systolicDepth), repeatCount(repeatCount) {
  }

  Value operator()(ConversionPatternRewriter &rewriter, Location loc, Value C,
                   Value A, Value B) {

    auto srcPrec = encodePrecision(dpasTy);

    auto funName = intrinsicDecl.getName();
    auto retType = intrinsicDecl.getResultTypes();
    auto funCall = rewriter.create<spirv::FunctionCallOp>(
        loc, retType, funName,
        ValueRange{C, A, B, i32_val(srcPrec) /*src1's precision*/,
                   i32_val(srcPrec) /*src1's precision*/,
                   i32_val(systolicDepth) /*systolic depth*/,
                   i32_val(repeatCount) /*repeate count*/,
                   int_val(1, 0) /*is double = false*/});
    auto ret = funCall.getReturnValue();
    return ret;
  }

private:
  DPASEngineType dpasTy;
  uint32_t systolicDepth;
  uint32_t repeatCount;
};

class GenISA_Prefetch {

public:
  explicit GenISA_Prefetch(IntrinsicBuilder *builder, mlir::Type ptrTy,
                           int dataSize)
      : dataSize(dataSize) {
    // get GenISA intrinsic declaration.
    SmallVector<mlir::Type *> mlirTys{&ptrTy};
    auto genISAName =
        getGenISAName(llvm::GenISAIntrinsic::ID::GenISA_LSCPrefetch, {});
    auto mlirContext = builder->builder.getContext();
    FunctionType funcTy = getGenISAType(
        *mlirContext, llvm::GenISAIntrinsic::ID::GenISA_LSCPrefetch, mlirTys);

    NamedAttrList attributes;

    attributes.set(spirv::SPIRVDialect::getAttributeName(
                       spirv::Decoration::LinkageAttributes),
                   spirv::LinkageAttributesAttr::get(
                       mlirContext, genISAName,
                       spirv::LinkageTypeAttr::get(
                           mlirContext, spirv::LinkageType::Import)));

    llvm::LLVMContext llvmContext;
    auto llvmAttributes = llvm::GenISAIntrinsic::getGenIntrinsicAttributes(
        llvmContext, llvm::GenISAIntrinsic::ID::GenISA_LSCPrefetch);

    for (auto &attr : llvmAttributes) {
      processPassthroughAttrs(*mlirContext, attributes, attr);
    }

    intrinsicDecl = spirv::appendOrGetFuncOp(
        mlir::UnknownLoc::get(mlirContext), builder->builder, genISAName,
        funcTy, spirv::FunctionControl::Inline, attributes);
  }

  Value operator()(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                   int vecSize) {
    auto funName = intrinsicDecl.getName();
    auto retType = intrinsicDecl.getResultTypes();
    auto funCall = rewriter.create<spirv::FunctionCallOp>(
        loc, retType, funName,
        ValueRange{ptr, i32_val(0) /*offset to the ptr*/,
                   i32_val(dataSize) /*data size in bytes*/,
                   i32_val(vecSize) /*element number*/,
                   i32_val(0) /*cache control*/});
    return Value();
  }

private:
  spirv::FuncOp intrinsicDecl;
  int dataSize;
};

class GenXDPAS2 : public VCIntrinsicSIMTAdaptor<GenXDPAS2> {

public:
  explicit GenXDPAS2(IntrinsicBuilder *builder, DPASEngineType dpasTy,
                     uint8_t repeatCount, mlir::Type dTy, mlir::Type cTy,
                     mlir::Type bTy, mlir::Type aTy)
      : VCIntrinsicSIMTAdaptor<GenXDPAS2>(
            builder, llvm::GenXIntrinsic::ID::genx_dpas2,
            [=]() -> std::string {
              // encoding the suffix for the unique name of SIMT interface.
              auto srcPrec = encodePrecision(dpasTy);
              std::string suffix = "srcPrec_" + std::to_string(srcPrec) +
                                   "_systolicDepth_" + std::to_string(8) +
                                   "_repeatCount_" +
                                   std::to_string(repeatCount);
              return suffix;
            }(),
            dTy, cTy, bTy, aTy) {
    auto &esimdWrapper = esimdFunc.esimdWrapper;
    if (esimdWrapper.empty()) {
      auto entryBlock = esimdWrapper.addEntryBlock();

      OpBuilder rewriter(entryBlock, entryBlock->begin());
      mlir::Location loc = mlir::UnknownLoc::get(rewriter.getContext());

      auto srcPrecision = encodePrecision(dpasTy);
      auto args = entryBlock->getArguments();
      auto funCall = rewriter.create<spirv::FunctionCallOp>(
          mlir::UnknownLoc::get(rewriter.getContext()),
          TypeRange{vcIntrinsicDecl.getResultTypes()},
          vcIntrinsicDecl.getName(),
          ValueRange{
              args[0] /*c - src0*/, args[1] /*b - src1*/, args[2] /*a - src2*/,
              i32_val(srcPrecision) /*int information of src1 PresisionType*/,
              i32_val(srcPrecision) /*int information of src2 PresisionType*/,
              i32_val(8) /*int SystolicDepth*/,
              i32_val(repeatCount) /*int RepeatCount*/,
              i32_val(1) /*int sign dst( 0 - unsigned, 1 sign)*/,
              i32_val(1) /*int sign dst( 0 - unsigned, 1 sign)*/});
      auto ret = funCall.getReturnValue();

      rewriter.create<spirv::ReturnValueOp>(
          mlir::UnknownLoc::get(rewriter.getContext()), TypeRange(),
          ValueRange{ret});
    }
  }

  Value operator()(ConversionPatternRewriter &rewriter, Location loc, Value C,
                   Value B, Value A) {
    // get the interface for the SIMT.
    auto funPtrTy = spirv::FunctionPointerINTELType::get(
        esimdFunc.esimdFunTy, spirv::StorageClass::Function);
    spirv::INTELConstantFunctionPointerOp funValue =
        rewriter.create<spirv::INTELConstantFunctionPointerOp>(
            loc, funPtrTy, esimdFunc.esimdWrapper.getName());

    auto funCall = rewriter.create<spirv::FunctionCallOp>(
        loc, TypeRange{esimdFunc.simtIntfTy.getResults()},
        esimdFunc.simtIntfName, ValueRange{funValue, C, B, A});
    auto ret = funCall.getReturnValue();
    return ret;
  }
};

#if 0
// GCN instruction common interface.
// Put the generic logic for all the instructions here.
struct IntrinsicCommon {
  explicit IntrinsicCommon(GCNBuilder *builder) : builder(builder) {}

  using Operand = GCNBuilder::Operand;
  using Modifier = GCNBuilder::Modifier;

  // clang-format off
  GCNInstrExecution& operator()() { return call({}, {}); }
  GCNInstrExecution& operator()(Operand* a) { return call({a}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b) { return call({a, b}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c) { return call({a, b, c}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d) { return call({a, b, c, d}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e) { return call({a, b, c, d, e}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f) { return call({a, b, c, d, e, f}, {}); }
  GCNInstrExecution& operator()(Operand* a, Operand* b, Operand* c, Operand* d, Operand * e, Operand* f, Operand* g) { return call({a, b, c, d, e, f, g}, {}); }
  // clang-format on

  // Set operands of this instruction.
  GCNInstrExecution &operator()(llvm::ArrayRef<Operand *> oprs,
                                llvm::ArrayRef<Modifier *> mods);

protected:
  GCNInstrExecution &call(llvm::ArrayRef<Operand *> oprs,
                          ArrayRef<Modifier *> mods);

  GCNBuilder *builder{};
  llvm::SmallVector<std::string, 4> instrParts;

  friend class GCNInstrExecution;
};

template <class ConcreteT> struct GCNInstrBase : public GCNInstrCommon {
  using Operand = GCNBuilder::Operand;
  using Modifier = GCNBuilder::Modifier;

  explicit GCNInstrBase(GCNBuilder *builder, const std::string &name)
      : GCNInstrCommon(builder) {
    o(name);
  }

  ConcreteT &o(const std::string &suffix, bool predicate = true) {
    if (predicate)
      instrParts.push_back(suffix);
    return *static_cast<ConcreteT *>(this);
  }
};

enum VectorWidth { Byte = 8, Short = 16, Dword = 32, Qword = 64 };

struct GCNInstr : public GCNInstrBase<GCNInstr> {
  using GCNInstrBase<GCNInstr>::GCNInstrBase;

  GCNInstr &float_op_type(int width) {
    switch (width) {
    case Byte:
      assert(Byte != width);
      break;
    case Short:
      o("f16");
      break;
    case Dword:
      o("f32");
      break;
    case Qword:
      o("f64");
      break;
    default:
      break;
    }
    return *this;
  }
};

struct GCNInstrExecution {
  using Operand = GCNBuilder::Operand;
  using Modifier = GCNBuilder::Modifier;

  llvm::SmallVector<Operand *> argsInOrder;
  llvm::SmallVector<Modifier *> mods;

  GCNInstrExecution() = default;
  explicit GCNInstrExecution(GCNInstrCommon *instr,
                             llvm::ArrayRef<Operand *> oprs,
                             llvm::ArrayRef<Modifier *> modifiers)
      : instr(instr), argsInOrder(oprs.begin(), oprs.end()),
        mods(modifiers.begin(), modifiers.end()) {}

  std::string dump() const;

  SmallVector<Operand *> getArgList() const;

  GCNInstrCommon *instr{};
};

struct GCNMemInstr : public GCNInstrBase<GCNMemInstr> {
  using GCNInstrBase<GCNMemInstr>::GCNInstrBase;
  // Add specific type suffix to instruction

  GCNMemInstr &load_type(int width) {
    switch (width) {
    case Byte:
      o("ubyte");
      break;
    case Short:
      o("ushort");
      break;
    case Dword:
      o("dword");
      break;
    case Qword:
      o("dwordx2");
      break;
    default:
      break;
    }
    return *this;
  }

  GCNMemInstr &store_type(int width) {
    switch (width) {
    case Byte:
      o("byte");
      break;
    case Short:
      o("short");
      break;
    case Dword:
      o("dword");
      break;
    case Qword:
      o("dwordx2");
      break;
    default:
      break;
    }
    return *this;
  }
};
#endif

} // namespace intel
} // namespace triton
} // namespace mlir

#endif // TRITON_VCINTRINSICHELPER_H
