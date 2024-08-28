#pragma once

#include <vector>
#include <unordered_map>
#include "../../extra/cloud/cloud.h"

class SampledData {
public:
  int count;
  PointXYZ point;

  SampledData() {
    count = 0;
    point = PointXYZ();
  }

  void update(const PointXYZ& p) {
    count += 1;
    point += p;
  }
};

void single_grid_subsampling_cpu(
  std::vector<PointXYZ>& o_points,
  std::vector<PointXYZ>& s_points,
  std::vector<PointXYZ>& o_dps,
  std::vector<PointXYZ>& s_dps,
  float voxel_size
);
// step into 1.2.1.1.1.1 : 下采样
void grid_subsampling_cpu(
  std::vector<PointXYZ>& points,
  std::vector<PointXYZ>& s_points,
  std::vector<PointXYZ>& dps,
  std::vector<PointXYZ>& s_dps,
  std::vector<long>& o_lengths,
  std::vector<long>& s_lengths,
  float voxel_size
);

