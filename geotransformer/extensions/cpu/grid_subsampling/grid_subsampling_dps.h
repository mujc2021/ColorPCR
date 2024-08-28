#pragma once

#include <vector>
#include "../../common/torch_helper.h"
// step into 1.2.1.1.1 : 下采样
std::vector<at::Tensor> grid_subsampling_dps(
  at::Tensor points,
  at::Tensor dps,
  at::Tensor lengths,
  float voxel_size
);
