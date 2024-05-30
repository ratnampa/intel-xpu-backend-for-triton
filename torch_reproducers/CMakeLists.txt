cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(reproducer)

set(CMAKE_CXX_COMPILER icpx)
set(BUILD_SHARED_LIBS OFF)

list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb/")

find_package(Torch REQUIRED)

set(COMPILE_FLAGS "-fsycl -Wall")
set(LINK_FLAGS "-fsycl -lze_loader")

set(TARGET_NAME torch_reproducer)
add_executable(${TARGET_NAME} torch_reproducer.cpp)
target_include_directories(${TARGET_NAME} PRIVATE "/opt/intel/oneapi/compiler/latest/include")
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

target_link_libraries(torch_reproducer "${TORCH_LIBRARIES}")
set_property(TARGET torch_reproducer PROPERTY CXX_STANDARD 17)