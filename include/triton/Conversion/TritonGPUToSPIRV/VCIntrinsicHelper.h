//
// Created by chengjun on 9/5/23.
//

#ifndef TRITON_VCINTRINSICHELPER_H
#define TRITON_VCINTRINSICHELPER_H

#include "Utility.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToSPIRV/ESIMDHelper.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/GenXIntrinsics/GenXIntrinsics.h"
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

class VCIntrinsicCommon;
class VCIntrinsic;

class VCIBuilder {
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
  explicit VCIBuilder(int threadsPerWarp, OpBuilder &builder)
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
  llvm::SmallVector<std::unique_ptr<VCIntrinsicCommon>, 2> instrs;
  //  int oprCounter{};
  OpBuilder &builder;
  int threadsPerWarp;
};
class VCIBuilder;

class VCIntrinsicCommon {

public:
  VCIntrinsicCommon(VCIBuilder *builder) {}

private:
};

template <class ConcreteT>
struct VCIntrinsicSIMTAdaptor : public VCIntrinsicCommon {

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
  explicit VCIntrinsicSIMTAdaptor(VCIBuilder *builder,
                                  llvm::GenXIntrinsic::ID intrinsic,
                                  TYPES... tys)
      : VCIntrinsicCommon(builder), threadsPerWarp(builder->threadsPerWarp) {
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
    auto vcIntrinsicDecl = appendOrGetGenXDeclaration(
        builder->builder, intrinsic, vectorizedTysPtr);

    // get the ESIMD wrapper.
    auto genXName =
        mlir::triton::intel::getGenXName(intrinsic, vectorizedTysPtr);
    std::string esimdWrapperName = "VCIntrinsicWrapper_" + genXName;

    esimdFunc = ESIMDToSIMTAdaptor(builder->builder, esimdWrapperName, tys...);
    spirv::FuncOp &esimdWrapper = esimdFunc.esimdWrapper;

    if (esimdWrapper.empty()) {
      auto block = esimdWrapper.addEntryBlock();
      auto entryBlock = &esimdWrapper.getFunctionBody().front();

      OpBuilder builder(entryBlock, entryBlock->begin());

      auto funCall = builder.create<spirv::FunctionCallOp>(
          mlir::UnknownLoc::get(builder.getContext()),
          TypeRange{vcIntrinsicDecl.getResultTypes()},
          vcIntrinsicDecl.getName(), ValueRange{entryBlock->getArguments()});
      auto ret = funCall.getReturnValue();

      builder.create<spirv::ReturnValueOp>(
          mlir::UnknownLoc::get(builder.getContext()), TypeRange(),
          ValueRange{ret});
    }
  }

protected:
  ESIMDToSIMTAdaptor esimdFunc;
  int threadsPerWarp;
};

class GenXDPAS2 : public VCIntrinsicSIMTAdaptor<GenXDPAS2> {
public:
  explicit GenXDPAS2(VCIBuilder *builder, mlir::VectorType dTy,
                     mlir::VectorType cTy, mlir::VectorType bTy,
                     mlir::VectorType aTy)
      : VCIntrinsicSIMTAdaptor<GenXDPAS2>(
            builder, llvm::GenXIntrinsic::ID::genx_dpas2, dTy, cTy, bTy, aTy,
            UniformArgType(mlir::IntegerType::get(
                builder->builder.getContext(),
                32)), // int information of src1 PresisionType
            UniformArgType(mlir::IntegerType::get(
                builder->builder.getContext(),
                32)), // int information of src2 PresisionType
            UniformArgType(mlir::IntegerType::get(builder->builder.getContext(),
                                                  32)), // int SystolicDepth
            UniformArgType(mlir::IntegerType::get(builder->builder.getContext(),
                                                  32)), // int RepeatCount
            UniformArgType(mlir::IntegerType::get(
                builder->builder.getContext(),
                32)), // int sign dst( 0 - unsigned, 1 sign)
            UniformArgType(mlir::IntegerType::get(
                builder->builder.getContext(),
                32)) // int sign dst( 0 - unsigned, 1 sign)
        ) {}

  Value operator()(ConversionPatternRewriter &rewriter, Location loc) {
#if 0
      // refer to IGC/visa/Common_ISA_util.cpp#87
      auto encodePrecision = [&](Type type) -> uint8_t {
        if (type == rewriter.getBF16Type())
          return 9;
        else if (type == rewriter.getF16Type())
          return 10;
        else if (type == rewriter.getTF32Type())
          return 12;
        else {
          assert(0 && "add more support");
          return 0;
        }
      };
#endif
    // get the interface for the SIMT.
    auto funPtrTy = spirv::FunctionPointerINTELType::get(
        esimdFunc.esimdFunTy, spirv::StorageClass::Function);
    spirv::INTELConstantFunctionPointerOp funValue =
        rewriter.create<spirv::INTELConstantFunctionPointerOp>(
            loc, funPtrTy, esimdFunc.esimdWrapper.getName());

    auto funCall = rewriter.create<spirv::FunctionCallOp>(
        loc, TypeRange{esimdFunc.simtIntfTy.getResults()},
        esimdFunc.simtIntfName, ValueRange{funValue, i32_val(0)});
    auto ret = funCall.getReturnValue();
    return ret;
  }
};

#if 1

#endif
#if 0
// GCN instruction common interface.
// Put the generic logic for all the instructions here.
struct VCIntrinsicCommon {
  explicit VCIntrinsicCommon(GCNBuilder *builder) : builder(builder) {}

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
