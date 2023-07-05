# - Try to find SPIRV-LLVM-Translator
#
include(FetchContent)

if (NOT VCIntrinsics_FOUND)

    set(VCIntrinsics_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/VCIntrinsics")
    message(STATUS "VC Intrinsics location is not specified. Will try to download
                  VCIntrinsics from https://github.com/intel/vc-intrinsics.git into
                  ${VCIntrinsics_SOURCE_DIR}")
    #    set(SPIRV_HEADERS_SKIP_INSTALL ON)
    #    set(SPIRV_HEADERS_SKIP_EXAMPLES ON)
    file(READ vc-intrinsics-tag.conf VCIntrinsics_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${VCIntrinsics_TAG}" VCIntrinsics_TAG)
    FetchContent_Declare(vc-intrinsics
            GIT_REPOSITORY    https://github.com/intel/vc-intrinsics.git
            GIT_TAG           ${VCIntrinsics_TAG}
            SOURCE_DIR ${VCIntrinsics_SOURCE_DIR}
            )

    FetchContent_MakeAvailable(vc-intrinsics)

    set(VCIntrinsics_INCLUDE_DIR "${VCIntrinsics_SOURCE_DIR}/include"
            CACHE INTERNAL "VCIntrinsics_INCLUDE_DIR")

    find_package_handle_standard_args(
            VCIntrinsics
            FOUND_VAR VCIntrinsics_FOUND
            REQUIRED_VARS
            VCIntrinsics_SOURCE_DIR)

endif (NOT VCIntrinsics_FOUND)
