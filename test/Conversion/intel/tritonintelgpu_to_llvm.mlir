// RUN: triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: check that the gpu spatial_extents is inserted
// CHECK: module attributes {{{.*}}gpu.spatial_extents<reqdSubgroupSize = 16 : i32>
module attributes { "triton_gpu.threads-per-warp" = 16 : i32, "triton_gpu.num-warps" = 4 : i32 } { }
