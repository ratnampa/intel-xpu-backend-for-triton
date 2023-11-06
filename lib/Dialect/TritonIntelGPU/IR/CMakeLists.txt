add_mlir_dialect_library(TritonIntelGPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonIntelGPUTableGen
  TritonIntelGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
)
