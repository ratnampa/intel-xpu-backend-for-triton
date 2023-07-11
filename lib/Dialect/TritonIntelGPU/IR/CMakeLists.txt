add_mlir_dialect_library(TritonIntelGPUIR
  Dialect.cpp

  DEPENDS
  TritonIntelGPUTableGen
  TritonIntelGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUOps
)
