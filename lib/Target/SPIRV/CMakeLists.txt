# SPIRV Tools is required to disassemble the spirv module
find_package(SPIRV_Tools)

# SPIRV-LLVM Translator is required.
find_package(SPIRVToLLVMTranslator)

add_mlir_translation_library(TritonSPIRV
        SPIRVTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRLLVMIRTransforms
        MLIRSupport
        MLIRTargetLLVMIRExport
        # spirv tools
        SPIRV-Tools
        SPIRV-Tools-link
        LLVMSPIRVLib
        )

#add_custom_command(TARGET LLVMSPIRVLib PRE_BUILD COMMAND git apply somepatch.patch)
#MESSAGE(STATUS  "johnlu SPIRVToLLVMTranslator_SOURCE_DIR ${SPIRVToLLVMTranslator_SOURCE_DIR}")
execute_process(
        COMMAND ${GIT_EXECUTABLE} checkout .
        WORKING_DIRECTORY ${SPIRVToLLVMTranslator_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY)

execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --whitespace=fix /home/chengjun/workspace/CLionProject/triton-torch-xpu/third_party/intel_xpu_backend/lib/Target/SPIRV/freeze_insn_workaround.patch
        WORKING_DIRECTORY ${SPIRVToLLVMTranslator_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY)

# Add SPIRVLLVMTranslator include dir
target_include_directories(TritonSPIRV PRIVATE ${SPIRVToLLVMTranslator_INCLUDE_DIR})
