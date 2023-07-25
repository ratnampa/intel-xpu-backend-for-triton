# - Try to find SPIRV
#
include(FetchContent)

if (NOT SPIRV_TOOLS_FOUND)

    # Download spirv.hpp from the official SPIRV-Headers repository.
    # One can skip this step by manually setting
    # EXTERNAL_SPIRV_HEADERS_SOURCE_DIR path.
    if(NOT DEFINED EXTERNAL_SPIRV_HEADERS_SOURCE_DIR)
        set(EXTERNAL_SPIRV_HEADERS_SOURCE_DIR
                "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Headers")
        message(STATUS "SPIR-V Headers location is not specified. Will try to download
      spirv.hpp from https://github.com/KhronosGroup/SPIRV-Headers into
      ${EXTERNAL_SPIRV_HEADERS_SOURCE_DIR}")
        set(SPIRV_HEADERS_SKIP_INSTALL ON)
        set(SPIRV_HEADERS_SKIP_EXAMPLES ON)
        file(READ spirv-headers-tag.conf SPIRV_HEADERS_TAG)
        # Strip the potential trailing newline from tag
        string(STRIP "${SPIRV_HEADERS_TAG}" SPIRV_HEADERS_TAG)
        FetchContent_Declare(spirv-headers
                GIT_REPOSITORY    https://github.com/KhronosGroup/SPIRV-Headers.git
                GIT_TAG           ${SPIRV_HEADERS_TAG}
                SOURCE_DIR ${EXTERNAL_SPIRV_HEADERS_SOURCE_DIR}
                )
        FetchContent_MakeAvailable(spirv-headers)
    endif()

    SET(SPIRV-Headers_SOURCE_DIR ${EXTERNAL_SPIRV_HEADERS_SOURCE_DIR})
    SET(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR  "${EXTERNAL_SPIRV_HEADERS_SOURCE_DIR}" CACHE INTERNAL "LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR")

    if(NOT DEFINED EXTERNAL_SPIRV_TOOLS_SOURCE_DIR)
        set(EXTERNAL_SPIRV_TOOLS_SOURCE_DIR
                "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools")
        message(STATUS "SPIR-V Tools location is not specified. Will try to download
      spirv tools from https://github.com/KhronosGroup/SPIRV-Tools into
      ${EXTERNAL_SPIRV_TOOLS_SOURCE_DIR}")
        set(SPIRV_SKIP_TESTS ON)
        file(READ spirv-tools-tag.conf SPIRV_TOOLS_TAG)
        # Strip the potential trailing newline from tag
        string(STRIP "${SPIRV_TOOLS_TAG}" SPIRV_TOOLS_TAG)
        FetchContent_Declare(spirv-tools
                GIT_REPOSITORY    https://github.com/KhronosGroup/SPIRV-Tools.git
                GIT_TAG           ${SPIRV_TOOLS_TAG}
                SOURCE_DIR ${EXTERNAL_SPIRV_TOOLS_SOURCE_DIR}
                )
        FetchContent_MakeAvailable(spirv-tools)
    endif()

    find_package_handle_standard_args(
            SPIRV_Tools
            FOUND_VAR SPIRV_TOOLS_FOUND
            REQUIRED_VARS
                EXTERNAL_SPIRV_TOOLS_SOURCE_DIR)


endif (NOT SPIRV_TOOLS_FOUND)
