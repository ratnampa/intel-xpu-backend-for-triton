#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "triton/Dialect/TritonIntelGPU/IR/Dialect.cpp.inc"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::gpu::intel;

// Utility
namespace mlir {
namespace triton {
namespace gpu {
namespace intel {}
} // namespace gpu
} // namespace triton
} // namespace mlir

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = attr.dyn_cast<IntegerAttr>();
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned, 2> &res,
                                       StringRef desc) {
  auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

static LogicalResult parseUInt(AsmParser &parser, const NamedAttribute &attr,
                               unsigned &value, StringRef desc) {
  return parseIntAttrValue(parser, attr.getValue(), value, desc);
};

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// MMA encoding
//===----------------------------------------------------------------------===//

bool IntelMmaEncodingAttr::isVolta() const { return false; }

bool IntelMmaEncodingAttr::isAmpere() const { return false; }

SmallVector<unsigned> IntelMmaEncodingAttr::getSizePerThread() const {
  unsigned threadsPerWarp = getThreadsPerWarp();
  auto shapeC = getShapeC();
  unsigned elemsNum = product<unsigned>(shapeC);
  unsigned elemsPerThread = elemsNum / threadsPerWarp;
  if (elemsPerThread == 4) {
    return {2, 2};
  } else if (elemsPerThread == 2) {
    return {2, 1};
  } else {
    llvm_unreachable("Unexpected mma version");
  }
};

SmallVector<unsigned>
IntelMmaEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  auto shapeC = getShapeC();
  return {shapeC[0] * getWarpsPerCTA()[0], shapeC[1] * getWarpsPerCTA()[1]};
}
SmallVector<unsigned>
IntelMmaEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                        Type eltTy) const {
  size_t rank = shape.size();
  assert(rank == 2 && "Unexpected rank of mma layout");

  SmallVector<unsigned> elemsPerThread(rank);
  auto shapePerCTA = getShapePerCTA({1, 1}, shape);
  unsigned tilesRow = ceil<unsigned>(shape[0], shapePerCTA[0]);
  unsigned tilesCol = ceil<unsigned>(shape[1], shapePerCTA[1]);
  auto sizePerThread = getSizePerThread();
  elemsPerThread[0] = sizePerThread[0] * tilesRow;
  elemsPerThread[1] = sizePerThread[1] * tilesCol;

  return elemsPerThread;
}
unsigned IntelMmaEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                      Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}
SmallVector<unsigned> IntelMmaEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res{1, 1};
  return res;
}
SmallVector<unsigned> IntelMmaEncodingAttr::getCTAOrder() const {
  SmallVector<unsigned> res{0, 1};
  return res;
}
SmallVector<unsigned> IntelMmaEncodingAttr::getCTAsPerCGA() const {
  SmallVector<unsigned> res{1, 1};
  return res;
}
unsigned IntelMmaEncodingAttr::getNumCTAs() const { return 1; }
SmallVector<int64_t> IntelMmaEncodingAttr::getXMXRep(ArrayRef<int64_t> shape,
                                                     int bitwidth,
                                                     int opIdx) const {
  SmallVector<int> shapePerWarp = {8, 8, 4 * 64 / bitwidth};
  auto warpsPerCTA = getWarpsPerCTA();
  if (opIdx == 0)
    return {std::max<int64_t>(1, shape[0] / (shapePerWarp[0] * warpsPerCTA[0])),
            std::max<int64_t>(1, shape[1] / shapePerWarp[2])};
  else {
    assert(opIdx == 1);
    return {
        std::max<int64_t>(1, shape[0] / shapePerWarp[2]),
        std::max<int64_t>(1, shape[1] / (shapePerWarp[1] * warpsPerCTA[1]))};
  }
}
unsigned IntelMmaEncodingAttr::getTotalElemsPerThreadForOperands(
    ArrayRef<int64_t> shape, mlir::Type eltTy, int opIdx) const {
  auto shapePerCTA = getShapePerCTA(*this, shape);
  int warpsPerCTAM = getWarpsPerCTA()[0];
  int warpsPerCTAN = getWarpsPerCTA()[1];
  auto rep = getXMXRep(shapePerCTA, eltTy.getIntOrFloatBitWidth(), opIdx);
  if (opIdx == 0)
    return rep[0] * rep[1];
  else // if (opIdx == 1)
    return rep[0] * rep[1];
}
Attribute IntelMmaEncodingAttr::getCTALayout() const {
  return CTALayoutAttr::get(getContext(), getCTAsPerCGA(), getCTASplitNum(),
                            getCTAOrder());
}

Attribute IntelMmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned, 2> warpsPerCTA;
  SmallVector<unsigned, 2> shapeA;
  SmallVector<unsigned, 2> shapeB;
  SmallVector<unsigned, 2> shapeC;
  SmallVector<unsigned, 2> packedA;
  SmallVector<unsigned, 2> packedB;
  SmallVector<unsigned, 2> packedC;
  unsigned threadsPerWarp;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "A") {
      if (parseIntArrayAttr(parser, attr, shapeA, "shapeA").failed())
        return {};
    }
    if (attr.getName() == "B") {
      if (parseIntArrayAttr(parser, attr, shapeB, "shapeB").failed())
        return {};
    }
    if (attr.getName() == "C") {
      if (parseIntArrayAttr(parser, attr, shapeC, "shapeC").failed())
        return {};
    }
    if (attr.getName() == "packedA") {
      if (parseIntArrayAttr(parser, attr, packedA, "packedA").failed())
        return {};
    }
    if (attr.getName() == "packedB") {
      if (parseIntArrayAttr(parser, attr, packedB, "packedB").failed())
        return {};
    }
    if (attr.getName() == "packedC") {
      if (parseIntArrayAttr(parser, attr, packedC, "packedC").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "threadsPerWarp") {
      if (parseUInt(parser, attr, threadsPerWarp, "threadsPerWarp").failed())
        return {};
    }
  }

  return parser.getChecked<IntelMmaEncodingAttr>(
      parser.getContext(), shapeA, shapeB, shapeC, packedA, packedB, packedC,
      warpsPerCTA, threadsPerWarp);
}

void IntelMmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "A = [" << getShapeA() << "], "
          << "B = [" << getShapeB() << "], "
          << "C = [" << getShapeC() << "], ";
  auto packedA = getPackedA();
  if (!packedA.empty())
    printer << "packedA = [" << packedA << "], ";
  auto packedB = getPackedB();
  if (!packedB.empty())
    printer << "packedB = [" << packedB << "], ";
  auto packedC = getPackedC();
  if (!packedC.empty())
    printer << "packedC = [" << packedC << "], ";
  printer << "threadsPerWarp = " << getThreadsPerWarp() << ", "
          << "warpsPerCTA = [" << getWarpsPerCTA() << "]"
          << "}>";
}

//===----------------------------------------------------------------------===//

void TritonIntelGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc"
      >();
}

// verify TritonIntelGPU ops
LogicalResult
TritonIntelGPUDialect::verifyOperationAttribute(Operation *op,
                                                NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
