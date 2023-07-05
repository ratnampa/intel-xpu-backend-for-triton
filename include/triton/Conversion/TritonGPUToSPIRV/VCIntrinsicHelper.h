//
// Created by chengjun on 9/5/23.
//

#ifndef TRITON_VCINTRINSICHELPER_H
#define TRITON_VCINTRINSICHELPER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToSPIRV/ESIMDHelper.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/GenXIntrinsics/GenXIntrinsics.h"
#include <memory>
#include <string>

namespace mlir {
class ConversionPatternRewriter;
class Location;

namespace triton {
namespace intel {
using llvm::StringRef;

// class GCNInstr;
// class GCNInstrCommon;
// class GCNInstrExecution;
/**
VC Intrinsic helper to get the prototype of the vc intrinsic.
VC Intrinsic project repo:
https://github.com/intel/vc-intrinsics

*/

enum class lsc_subopcode : uint8_t {
  load = 0x00,
  load_strided = 0x01,
  load_quad = 0x02,
  load_block2d = 0x03,
  store = 0x04,
  store_strided = 0x05,
  store_quad = 0x06,
  store_block2d = 0x07,
  //
  atomic_iinc = 0x08,
  atomic_idec = 0x09,
  atomic_load = 0x0a,
  atomic_store = 0x0b,
  atomic_iadd = 0x0c,
  atomic_isub = 0x0d,
  atomic_smin = 0x0e,
  atomic_smax = 0x0f,
  atomic_umin = 0x10,
  atomic_umax = 0x11,
  atomic_icas = 0x12,
  atomic_fadd = 0x13,
  atomic_fsub = 0x14,
  atomic_fmin = 0x15,
  atomic_fmax = 0x16,
  atomic_fcas = 0x17,
  atomic_and = 0x18,
  atomic_or = 0x19,
  atomic_xor = 0x1a,
  //
  load_status = 0x1b,
  store_uncompressed = 0x1c,
  ccs_update = 0x1d,
  read_state_info = 0x1e,
  fence = 0x1f,
};
// The regexp for ESIMD intrinsics:
// /^_Z(\d+)__esimd_\w+/
// sycl/ext/oneapi/experimental/invoke_simd.hpp::__builtin_invoke_simd
// overloads instantiations:
static constexpr char INVOKE_SIMD_PREF[] =
    "_Z33__regcall3____builtin_invoke_simd";
static constexpr char ESIMD_INTRIN_PREF0[] = "_Z";
static constexpr char ESIMD_INTRIN_PREF1[] = "__esimd_";
static constexpr char ESIMD_INSERTED_VSTORE_FUNC_NAME[] = "_Z14__esimd_vstorev";
static constexpr char SPIRV_INTRIN_PREF[] = "__spirv_BuiltIn";
struct ESIMDIntrinDesc {
  // Denotes argument translation rule kind.
  enum GenXArgRuleKind {
    SRC_CALL_ARG, // is a call argument
    SRC_CALL_ALL, // this and subsequent args are just copied from the src call
    SRC_TMPL_ARG, // is an integer template argument
    UNDEF,        // is an undef value
    CONST_INT8,   // is an i8 constant
    CONST_INT16,  // is an i16 constant
    CONST_INT32,  // is an i32 constant
    CONST_INT64,  // is an i64 constant
  };

  enum class GenXArgConversion : int16_t {
    NONE,   // no conversion
    TO_I1,  // convert vector of N-bit integer to 1-bit
    TO_I8,  // convert vector of N-bit integer to 18-bit
    TO_I16, // convert vector of N-bit integer to 16-bit
    TO_I32, // convert vector of N-bit integer to 32-bit
  };

  // Denotes GenX intrinsic name suffix creation rule kind.
  enum GenXSuffixRuleKind {
    NO_RULE,
    BIN_OP,  // ".<binary operation>" - e.g. "*.add"
    NUM_KIND // "<numeric kind>" - e.g. "*i" for integer, "*f" for float
  };

  // Represents a rule how a GenX intrinsic argument is created from the source
  // call instruction.
  struct ArgRule {
    GenXArgRuleKind Kind;
    union Info {
      struct {
        int16_t CallArgNo;      // SRC_CALL_ARG: source call arg num
                                // SRC_TMPL_ARG: source template arg num
                                // UNDEF: source call arg num to get type from
                                // -1 denotes return value
        GenXArgConversion Conv; // GenXArgConversion
      } Arg;
      int NRemArgs;          // SRC_CALL_ALL: number of remaining args
      unsigned int ArgConst; // CONST_I16 OR CONST_I32: constant value
    } I;
  };

  // Represents a rule how a GenX intrinsic name suffix is created from the
  // source call instruction.
  struct NameRule {
    GenXSuffixRuleKind Kind;
    union Info {
      int CallArgNo; // DATA_TYPE: source call arg num to get type from
      int TmplArgNo; // BINOP: source template arg num denoting the binary op
    } I;
  };

  std::string GenXSpelling;
  SmallVector<ArgRule, 16> ArgRules;
  NameRule SuffixRule = {NO_RULE, {0}};

  int getNumGenXArgs() const {
    auto NRules = ArgRules.size();

    if (NRules == 0)
      return 0;

    // SRC_CALL_ALL is a "shortcut" to save typing, must be the last rule
    if (ArgRules[NRules - 1].Kind == GenXArgRuleKind::SRC_CALL_ALL)
      return ArgRules[NRules - 1].I.NRemArgs + (NRules - 1);
    return NRules;
  }

  bool isValid() const { return !GenXSpelling.empty(); }
};

using IntrinTable = std::unordered_map<std::string, ESIMDIntrinDesc>;

class ESIMDIntrinDescTable {
private:
  IntrinTable Table;

#define DEF_ARG_RULE(Nm, Kind)                                                 \
  static constexpr ESIMDIntrinDesc::ArgRule Nm(int16_t N) {                    \
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::Kind, {{N, {}}}};         \
  }
  DEF_ARG_RULE(l, SRC_CALL_ALL)
  DEF_ARG_RULE(u, UNDEF)

  static constexpr ESIMDIntrinDesc::ArgRule t(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::NONE}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t1(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I1}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t8(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I8}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t16(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I16}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule t32(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_TMPL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I32}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule a(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_CALL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::NONE}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule ai1(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_CALL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::TO_I1}}};
  }

  // Just an alias for a(int16_t N) to mark surface index arguments.
  static constexpr ESIMDIntrinDesc::ArgRule aSI(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{
        ESIMDIntrinDesc::SRC_CALL_ARG,
        {{N, ESIMDIntrinDesc::GenXArgConversion::NONE}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule c8(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::CONST_INT8, {{N, {}}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule c8(lsc_subopcode OpCode) {
    return c8(static_cast<uint8_t>(OpCode));
  }

  static constexpr ESIMDIntrinDesc::ArgRule c16(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::CONST_INT16, {{N, {}}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule c32(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::CONST_INT32, {{N, {}}}};
  }

  static constexpr ESIMDIntrinDesc::ArgRule c64(int16_t N) {
    return ESIMDIntrinDesc::ArgRule{ESIMDIntrinDesc::CONST_INT64, {{N, {}}}};
  }

  static constexpr ESIMDIntrinDesc::NameRule bo(int16_t N) {
    return ESIMDIntrinDesc::NameRule{ESIMDIntrinDesc::BIN_OP, {N}};
  }

  static constexpr ESIMDIntrinDesc::NameRule nk(int16_t N) {
    return ESIMDIntrinDesc::NameRule{ESIMDIntrinDesc::NUM_KIND, {N}};
  }

public:
  // The table which describes rules how to generate @llvm.genx.* intrinsics
  // from templated __esimd* intrinsics. The general rule is that the order and
  // the semantics of intrinsic arguments is the same in both intrinsic forms.
  // But for some arguments, where @llvm.genx.* mandates that the argument must
  // be 'constant' (see Intrinsic_definitions.py from the vcintrinsics repo),
  // it is passed as template argument to the corrsponding __esimd* intrinsic,
  // hence leading to some "gaps" in __esimd* form's arguments compared to the
  // @llvm.genx.* form.
  // TODO - fix all __esimd* intrinsics and table entries according to the rule
  // above.
  ESIMDIntrinDescTable() {
    Table = {
        // An element of the table is std::pair of <key, value>; key is the
        // source
        // spelling of and intrinsic (what follows the "__esimd_" prefix), and
        // the
        // value is an instance of the ESIMDIntrinDesc class.
        // Example for the "rdregion" intrinsic encoding:
        // "rdregion" - the GenX spelling of the intrinsic ("llvm.genx." prefix
        //      and type suffixes maybe added to get full GenX name)
        // {a(0), t(3),...}
        //      defines a map from the resulting genx.* intrinsic call arguments
        //      to the source call's template or function call arguments, e.g.
        //      0th genx arg - maps to 0th source call arg
        //      1st genx arg - maps to 3rd template argument of the source call
        // nk(N) or bo(N)
        //      a rule applied to the base intrinsic name in order to
        //      construct a full name ("llvm.genx." prefix s also added); e.g.
        //      - nk(-1) denotes adding the return type name-based suffix - "i"
        //          for integer, "f" - for floating point
        {"rdregion",
         {"rdregion", {a(0), t(3), t(4), t(5), a(1), t(6)}, nk(-1)}},
        {"rdindirect",
         {"rdregion", {a(0), c32(0), c32(1), c32(0), a(1), t(3)}, nk(-1)}},
        {{"wrregion"},
         {{"wrregion"},
          {a(0), a(1), t(3), t(4), t(5), a(2), t(6), ai1(3)},
          nk(-1)}},
        {{"wrindirect"},
         {{"wrregion"},
          {a(0), a(1), c32(0), c32(1), c32(0), a(2), t(3), ai1(3)},
          nk(-1)}},
        {"vload", {"vload", {l(0)}}},
        {"vstore", {"vstore", {a(1), a(0)}}},

        {"svm_block_ld_unaligned", {"svm.block.ld.unaligned", {l(0)}}},
        {"svm_block_ld", {"svm.block.ld", {l(0)}}},
        {"svm_block_st", {"svm.block.st", {l(1)}}},
        {"svm_gather", {"svm.gather", {ai1(1), t(3), a(0), u(-1)}}},
        {"svm_gather4_scaled",
         {"svm.gather4.scaled", {ai1(1), t(2), c16(0), c64(0), a(0), u(-1)}}},
        {"svm_scatter", {"svm.scatter", {ai1(2), t(3), a(0), a(1)}}},
        {"svm_scatter4_scaled",
         {"svm.scatter4.scaled", {ai1(2), t(2), c16(0), c64(0), a(0), a(1)}}},

        // intrinsics to query thread's coordinates:
        {"group_id_x", {"group.id.x", {}}},
        {"group_id_y", {"group.id.y", {}}},
        {"group_id_z", {"group.id.z", {}}},
        {"local_id", {"local.id", {}}},
        {"local_size", {"local.size", {}}},
        {"svm_atomic0", {"svm.atomic", {ai1(1), a(0), u(-1)}, bo(0)}},
        {"svm_atomic1", {"svm.atomic", {ai1(2), a(0), a(1), u(-1)}, bo(0)}},
        {"svm_atomic2",
         {"svm.atomic", {ai1(3), a(0), a(1), a(2), u(-1)}, bo(0)}},
        {"dp4", {"dp4", {a(0), a(1)}}},

        {"fence", {"fence", {a(0)}}},
        {"barrier", {"barrier", {}}},
        {"sbarrier", {"sbarrier", {a(0)}}},

        // arg0: i32 modifiers, constant
        // arg1: i32 surface index
        // arg2: i32 plane, constant
        // arg3: i32 block width in bytes, constant
        // (block height inferred from return type size and block width)
        // arg4: i32 x byte offset
        // arg5: i32 y byte offset
        {"media_ld", {"media.ld", {t(3), aSI(0), t(5), t(6), a(1), a(2)}}},

        // arg0: i32 modifiers, constant
        // arg1: i32 surface index
        // arg2: i32 plane, constant
        // arg3: i32 block width in bytes, constant
        // (block height inferred from data type size and block width)
        // arg4: i32 x byte offset
        // arg5: i32 y byte offset
        // arg6: data to write (overloaded)
        {"media_st",
         {"media.st", {t(3), aSI(0), t(5), t(6), a(1), a(2), a(3)}}},

        // arg0 : i32 is_modified, CONSTANT
        // arg1 : i32 surface index
        // arg2 : i32 offset(in owords for.ld / in bytes for.ld.unaligned)
        {"oword_ld_unaligned", {"oword.ld.unaligned", {t(3), aSI(0), a(1)}}},
        {"oword_ld", {"oword.ld", {t(3), aSI(0), a(1)}}},

        // arg0: i32 surface index
        // arg1: i32 offset (in owords)
        // arg2: data to write (overloaded)
        {"oword_st", {"oword.st", {aSI(0), a(1), a(2)}}},

        // surface index-based gather/scatter:
        // arg0: i32 log2 num blocks, CONSTANT (0/1/2 for num blocks 1/2/4)
        // arg1: i16 scale, CONSTANT
        // arg2: i32 surface index
        // arg3: i32 global offset in bytes
        // arg4: vXi32 element offset in bytes (overloaded)
        {"gather_scaled2",
         {"gather.scaled2", {t(3), t(4), aSI(0), a(1), a(2)}}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 log2 num blocks, CONSTANT (0/1/2 for num blocks 1/2/4)
        // arg2: i16 scale, CONSTANT
        // arg3: i32 surface index
        // arg4: i32 global offset in bytes
        // arg5: vXi32 element offset in bytes (overloaded)
        // arg6: old value of the data read
        {"gather_scaled",
         {"gather.scaled", {ai1(0), t(3), t(4), aSI(1), a(2), a(3), u(-1)}}},

        // arg0: i32 log2 num blocks, CONSTANT (0/1/2 for num blocks 1/2/4)
        // arg1: i16 scale, CONSTANT
        // arg2: i32 surface index
        // arg3: i32 global offset in bytes
        // arg4: vXi32 element offset in bytes (overloaded)
        // arg5: vXi1 predicate (overloaded)
        {"gather_masked_scaled2",
         {"gather.masked.scaled2", {t(3), t(4), aSI(0), a(1), a(2), ai1(3)}}},

        // arg0: i32 channel mask, CONSTANT
        // arg1: i16 scale, CONSTANT
        // arg2: i32 surface index
        // arg3: i32 global offset in bytes
        // arg4: vXi32 element offset in bytes
        // arg5: vXi1 predicate (overloaded)
        {"gather4_masked_scaled2",
         {"gather4.masked.scaled2", {t(2), t(4), aSI(0), a(1), a(2), ai1(3)}}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 log2 num blocks, CONSTANT (0/1/2 for num blocks 1/2/4)
        // arg2: i16 scale, CONSTANT
        // arg3: i32 surface index
        // arg4: i32 global offset in bytes
        // arg5: vXi32 element offset (overloaded)
        // arg6: data to write (overloaded)
        {"scatter_scaled",
         {"scatter.scaled", {ai1(0), t(3), t(4), aSI(1), a(2), a(3), a(4)}}},

        // arg0: vXi1 predicate (overloaded) (overloaded)
        // arg1: i32 channel mask, CONSTANT
        // arg2: i16 scale, CONSTANT
        // arg3: i32 surface index
        // arg4: i32 global offset in bytes
        // arg5: vXi32 element offset in bytes (overloaded)
        // arg6: old value of the data read
        {"gather4_scaled",
         {"gather4.scaled", {ai1(0), t(3), t(4), aSI(1), a(2), a(3), u(-1)}}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 channel mask, constant
        // arg2: i16 scale, constant
        // arg3: i32 surface index
        // arg4: i32 global offset in bytes
        // arg5: vXi32 element offset in bytes (overloaded)
        // arg6: data to write (overloaded)
        {"scatter4_scaled",
         {"scatter4.scaled", {ai1(0), t(3), t(4), aSI(1), a(2), a(3), a(4)}}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 surface index
        // arg2: vXi32 element offset in bytes
        // arg3: vXi32 original value of the register that the data is read into
        {"dword_atomic0",
         {"dword.atomic", {ai1(0), aSI(1), a(2), u(-1)}, bo(0)}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 surface index
        // arg2: vXi32 element offset in bytes (overloaded)
        // arg3: vXi32/vXfloat src
        // arg4: vXi32/vXfloat original value of the register that the data is
        // read into
        {"dword_atomic1",
         {"dword.atomic", {ai1(0), aSI(1), a(2), a(3), u(-1)}, bo(0)}},

        // arg0: vXi1 predicate (overloaded)
        // arg1: i32 surface index
        // arg2: vXi32 element offset in bytes
        // arg3: vXi32 src0
        // arg4: vXi32 src1
        // arg5: vXi32 original value of the register that the data is read into
        {"dword_atomic2",
         {"dword.atomic", {ai1(0), aSI(1), a(2), a(3), a(4), u(-1)}, bo(0)}},

        {"raw_sends2",
         {"raw.sends2",
          {a(0), a(1), ai1(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9), a(10),
           a(11)}}},
        {"raw_send2",
         {"raw.send2",
          {a(0), a(1), ai1(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9)}}},
        {"raw_sends2_noresult",
         {"raw.sends2.noresult",
          {a(0), a(1), ai1(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9)}}},
        {"raw_send2_noresult",
         {"raw.send2.noresult",
          {a(0), a(1), ai1(2), a(3), a(4), a(5), a(6), a(7)}}},
        {"wait", {"dummy.mov", {a(0)}}},
        {"dpas2",
         {"dpas2", {a(0), a(1), a(2), t(0), t(1), t(2), t(3), t(11), t(12)}}},
        {"dpas_nosrc0", {"dpas.nosrc0", {a(0), a(1), t(0)}}},
        {"dpasw", {"dpasw", {a(0), a(1), a(2), t(0)}}},
        {"dpasw_nosrc0", {"dpasw.nosrc0", {a(0), a(1), t(0)}}},
        {"nbarrier", {"nbarrier", {a(0), a(1), a(2)}}},
        {"raw_send_nbarrier_signal",
         {"raw.send.noresult", {a(0), ai1(4), a(1), a(2), a(3)}}},
        {"lsc_load_slm",
         {"lsc.load.slm",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0)}}},
        {"lsc_load_merge_slm",
         {"lsc.load.merge.slm",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0), a(2)}}},
        {"lsc_load_bti",
         {"lsc.load.bti",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), aSI(2)}}},
        {"lsc_load_merge_bti",
         {"lsc.load.merge.bti",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), aSI(2), a(2)}}},
        {"lsc_load_stateless",
         {"lsc.load.stateless",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0)}}},
        {"lsc_load_merge_stateless",
         {"lsc.load.merge.stateless",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0), a(2)}}},
        {"lsc_prefetch_bti",
         {"lsc.prefetch.bti",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), aSI(2)}}},
        {"lsc_prefetch_stateless",
         {"lsc.prefetch.stateless",
          {ai1(0), c8(lsc_subopcode::load), t8(1), t8(2), t16(3), t32(4), t8(5),
           t8(6), t8(7), c8(0), a(1), c32(0)}}},
        {"lsc_store_slm",
         {"lsc.store.slm",
          {ai1(0), c8(lsc_subopcode::store), t8(1), t8(2), t16(3), t32(4),
           t8(5), t8(6), t8(7), c8(0), a(1), a(2), c32(0)}}},
        {"lsc_store_bti",
         {"lsc.store.bti",
          {ai1(0), c8(lsc_subopcode::store), t8(1), t8(2), t16(3), t32(4),
           t8(5), t8(6), t8(7), c8(0), a(1), a(2), aSI(3)}}},
        {"lsc_store_stateless",
         {"lsc.store.stateless",
          {ai1(0), c8(lsc_subopcode::store), t8(1), t8(2), t16(3), t32(4),
           t8(5), t8(6), t8(7), c8(0), a(1), a(2), c32(0)}}},
        {"lsc_load2d_stateless",
         {"lsc.load2d.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t8(4), t8(5), t16(6), t16(7), t8(8),
           a(1), a(2), a(3), a(4), a(5), a(6)}}},
        {"lsc_prefetch2d_stateless",
         {"lsc.prefetch2d.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t8(4), t8(5), t16(6), t16(7), t8(8),
           a(1), a(2), a(3), a(4), a(5), a(6)}}},
        {"lsc_store2d_stateless",
         {"lsc.store2d.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t8(4), t8(5), t16(6), t16(7), t8(8),
           a(1), a(2), a(3), a(4), a(5), a(6), a(7)}}},
        {"lsc_xatomic_slm_0",
         {"lsc.xatomic.slm",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), u(-1), u(-1), c32(0), u(-1)}}},
        {"lsc_xatomic_slm_1",
         {"lsc.xatomic.slm",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), u(-1), c32(0), u(-1)}}},
        {"lsc_xatomic_slm_2",
         {"lsc.xatomic.slm",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), a(3), c32(0), u(-1)}}},
        {"lsc_xatomic_bti_0",
         {"lsc.xatomic.bti",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), u(-1), u(-1), aSI(2), u(-1)}}},
        {"lsc_xatomic_bti_1",
         {"lsc.xatomic.bti",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), u(-1), aSI(3), u(-1)}}},
        {"lsc_xatomic_bti_2",
         {"lsc.xatomic.bti",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), a(3), aSI(4), u(-1)}}},
        {"lsc_xatomic_stateless_0",
         {"lsc.xatomic.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), u(-1), u(-1), c32(0), u(-1)}}},
        {"lsc_xatomic_stateless_1",
         {"lsc.xatomic.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), u(-1), c32(0), u(-1)}}},
        {"lsc_xatomic_stateless_2",
         {"lsc.xatomic.stateless",
          {ai1(0), t8(1), t8(2), t8(3), t16(4), t32(5), t8(6), t8(7), t8(8),
           c8(0), a(1), a(2), a(3), c32(0), u(-1)}}},
        {"lsc_fence", {"lsc.fence", {ai1(0), t8(0), t8(1), t8(2)}}},
        {"sat", {"sat", {a(0)}}},
        {"fptoui_sat", {"fptoui.sat", {a(0)}}},
        {"fptosi_sat", {"fptosi.sat", {a(0)}}},
        {"uutrunc_sat", {"uutrunc.sat", {a(0)}}},
        {"ustrunc_sat", {"ustrunc.sat", {a(0)}}},
        {"sutrunc_sat", {"sutrunc.sat", {a(0)}}},
        {"sstrunc_sat", {"sstrunc.sat", {a(0)}}},
        {"abs", {"abs", {a(0)}, nk(-1)}},
        {"ssshl", {"ssshl", {a(0), a(1)}}},
        {"sushl", {"sushl", {a(0), a(1)}}},
        {"usshl", {"usshl", {a(0), a(1)}}},
        {"uushl", {"uushl", {a(0), a(1)}}},
        {"ssshl_sat", {"ssshl.sat", {a(0), a(1)}}},
        {"sushl_sat", {"sushl.sat", {a(0), a(1)}}},
        {"usshl_sat", {"usshl.sat", {a(0), a(1)}}},
        {"uushl_sat", {"uushl.sat", {a(0), a(1)}}},
        {"rol", {"rol", {a(0), a(1)}}},
        {"ror", {"ror", {a(0), a(1)}}},
        {"rndd", {"rndd", {a(0)}}},
        {"rnde", {"rnde", {a(0)}}},
        {"rndu", {"rndu", {a(0)}}},
        {"rndz", {"rndz", {a(0)}}},
        {"umulh", {"umulh", {a(0), a(1)}}},
        {"smulh", {"smulh", {a(0), a(1)}}},
        {"frc", {"frc", {a(0)}}},
        {"fmax", {"fmax", {a(0), a(1)}}},
        {"umax", {"umax", {a(0), a(1)}}},
        {"smax", {"smax", {a(0), a(1)}}},
        {"lzd", {"lzd", {a(0)}}},
        {"fmin", {"fmin", {a(0), a(1)}}},
        {"umin", {"umin", {a(0), a(1)}}},
        {"smin", {"smin", {a(0), a(1)}}},
        {"bfrev", {"bfrev", {a(0)}}},
        {"cbit", {"cbit", {a(0)}}},
        {"bfi", {"bfi", {a(0), a(1), a(2), a(3)}}},
        {"sbfe", {"sbfe", {a(0), a(1), a(2)}}},
        {"fbl", {"fbl", {a(0)}}},
        {"sfbh", {"sfbh", {a(0)}}},
        {"ufbh", {"ufbh", {a(0)}}},
        {"inv", {"inv", {a(0)}}},
        {"log", {"log", {a(0)}}},
        {"exp", {"exp", {a(0)}}},
        {"sqrt", {"sqrt", {a(0)}}},
        {"ieee_sqrt", {"ieee.sqrt", {a(0)}}},
        {"rsqrt", {"rsqrt", {a(0)}}},
        {"sin", {"sin", {a(0)}}},
        {"cos", {"cos", {a(0)}}},
        {"pow", {"pow", {a(0), a(1)}}},
        {"ieee_div", {"ieee.div", {a(0), a(1)}}},
        {"uudp4a", {"uudp4a", {a(0), a(1), a(2)}}},
        {"usdp4a", {"usdp4a", {a(0), a(1), a(2)}}},
        {"sudp4a", {"sudp4a", {a(0), a(1), a(2)}}},
        {"ssdp4a", {"ssdp4a", {a(0), a(1), a(2)}}},
        {"uudp4a_sat", {"uudp4a.sat", {a(0), a(1), a(2)}}},
        {"usdp4a_sat", {"usdp4a.sat", {a(0), a(1), a(2)}}},
        {"sudp4a_sat", {"sudp4a.sat", {a(0), a(1), a(2)}}},
        {"ssdp4a_sat", {"ssdp4a.sat", {a(0), a(1), a(2)}}},
        {"any", {"any", {ai1(0)}}},
        {"all", {"all", {ai1(0)}}},
        {"lane_id", {"lane.id", {}}},
        {"test_src_tmpl_arg",
         {"test.src.tmpl.arg", {t(0), t1(1), t8(2), t16(3), t32(4), c8(17)}}},
        {"slm_init", {"slm.init", {a(0)}}},
        {"bf_cvt", {"bf.cvt", {a(0)}}},
        {"tf32_cvt", {"tf32.cvt", {a(0)}}},
        {"__devicelib_ConvertFToBF16INTEL",
         {"__spirv_ConvertFToBF16INTEL", {a(0)}}},
        {"__devicelib_ConvertBF16ToFINTEL",
         {"__spirv_ConvertBF16ToFINTEL", {a(0)}}},
        {"addc", {"addc", {l(0)}}},
        {"subb", {"subb", {l(0)}}},
        {"bfn", {"bfn", {a(0), a(1), a(2), t(0)}}}};
  }

  const IntrinTable &getTable() { return Table; }
};

bool isStructureReturningFunction(StringRef FunctionName);

const IntrinTable &getIntrinTable();

const ESIMDIntrinDesc &getIntrinDesc(StringRef SrcSpelling);

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

#if 1
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
  //  llvm::SmallVector<std::unique_ptr<Operand>, 6> argArchive;
  //  llvm::SmallVector<std::unique_ptr<Modifier>, 2> modArchive;
  llvm::SmallVector<std::unique_ptr<VCIntrinsicCommon>, 2> instrs;
  //  llvm::SmallVector<std::unique_ptr<GCNInstrExecution>, 4> executions;
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

  template <class... TYPES>
  explicit VCIntrinsicSIMTAdaptor(VCIBuilder *builder,
                                  llvm::GenXIntrinsic::ID intrinsic,
                                  TYPES... tys)
      : VCIntrinsicCommon(builder) {
    auto mlirContext = builder->builder.getContext();
    int threadsPerWarp = builder->threadsPerWarp;

    SmallVector<mlir::Type *> mlirSIMTTys{&tys...};
    SmallVector<mlir::Type> mlirSIMDTys;
    for (mlir::Type *ty : mlirSIMTTys) {
      if (auto vecTy = ty->dyn_cast<mlir::VectorType>()) {
        auto vecShape = vecTy.getShape();
        assert(vecShape.size() == 1 && "VCIntrinsic only suppor 1 dim now");
        mlir::VectorType simdTy = mlir::VectorType::get(
            vecShape[0] * threadsPerWarp, vecTy.getElementType());
        mlirSIMDTys.push_back(simdTy);
        continue;
      }
    }

    SmallVector<mlir::Type *> mlirSIMDTysPtr;
    for (mlir::Type &ty : mlirSIMDTys) {
      mlirSIMDTysPtr.push_back(&ty);
    }

    // get GenX intrinsic declaration.
    vcIntrinsicDecl =
        appendOrGetGenXDeclaration(builder->builder, intrinsic, mlirSIMDTysPtr);

    //    auto types = mlir::triton::intel::getGenXType(*mlirContext, intrinsic,
    //    mlirSIMDTysPtr);

    // get the ESIMD wrapper.
    auto genXName = mlir::triton::intel::getGenXName(intrinsic, mlirSIMDTysPtr);
    std::string esimdWrapperName = "VCIntrinsicWrapper_" + genXName;
    SmallVector<mlir::Type, 8> ArgTys(mlirSIMDTys.begin() + 1,
                                      mlirSIMDTys.end());
    auto esimdFunTy =
        mlir::FunctionType::get(mlirContext, ArgTys, {mlirSIMDTys[0]});
    NamedAttrList attributes;
    esimdWrapper = mlir::triton::intel::appendOrGetESIMDFunc(
        builder->builder, esimdWrapperName, esimdFunTy,
        mlir::UnknownLoc::get(mlirContext));
  }

  //  Value operator()() {

  // get the interface for the SIMT.
  //    auto funPtrTy =
  //        spirv::FunctionPointerINTELType::get(esimdFunTy,
  //        spirv::StorageClass::Function);
  //    spirv::INTELConstantFunctionPointerOp funValue =
  //        builder->builder.create<spirv::INTELConstantFunctionPointerOp>(mlir::UnknownLoc::get(mlirContext),
  //        funPtrTy, esimdWrapperName);
  //  }

protected:
  mlir::spirv::FuncOp vcIntrinsicDecl;
  mlir::spirv::FuncOp esimdWrapper;
  mlir::spirv::FuncOp simtIntf;
};

enum VectorWidth { Byte = 8, Short = 16, Dword = 32, Qword = 64 };

class GenXDPAS2 : public VCIntrinsicSIMTAdaptor<GenXDPAS2> {
public:
  explicit GenXDPAS2(VCIBuilder *builder, mlir::VectorType dTy,
                     mlir::VectorType cTy, mlir::VectorType bTy,
                     mlir::VectorType aTy)
      : VCIntrinsicSIMTAdaptor<GenXDPAS2>(
            builder, llvm::GenXIntrinsic::ID::genx_dpas2, dTy, cTy, bTy, aTy) {

    if (esimdWrapper.empty()) {
      auto block = esimdWrapper.addEntryBlock();
      auto entryBlock = &esimdWrapper.getFunctionBody().front();

      OpBuilder rewriter(entryBlock, entryBlock->begin());

      Value retVal = rewriter.create<spirv::UndefOp>(
          mlir::UnknownLoc::get(rewriter.getContext()), dTy);

      //      for (int i = 0; i < simdLenght; i++)
      //        retVal = insert_val(dTy, i32_val(i + 10), retVal,
      //                            rewriter.getI32ArrayAttr(i));

      rewriter.create<spirv::ReturnValueOp>(
          mlir::UnknownLoc::get(rewriter.getContext()), TypeRange(),
          ValueRange{retVal});
    }
  }
};

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

#endif

} // namespace intel
} // namespace triton
} // namespace mlir

#endif // TRITON_VCINTRINSICHELPER_H
