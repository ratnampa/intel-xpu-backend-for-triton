add_triton_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Utility.cpp

  DEPENDS
  TritonIntelGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonGPUTransforms
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonIntelGPUIR
)
