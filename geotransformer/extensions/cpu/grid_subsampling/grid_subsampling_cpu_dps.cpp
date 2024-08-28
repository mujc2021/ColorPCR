#include "grid_subsampling_cpu_dps.h"
#include <cstdio>

// step into 1.2.1.1.1.1.1 : 下采样
void single_grid_subsampling_cpu(
  std::vector<PointXYZ>& points,  // 要下采样的点云
  std::vector<PointXYZ>& s_points,  // 要装的vector
  std::vector<PointXYZ>& dps,
  std::vector<PointXYZ>& s_dps,
//  std::vector<long>& s_indexes,
  float voxel_size
) {

  PointXYZ minCorner = min_point(points);  // 三个坐标极小值，只用作转化为origincorner
  PointXYZ maxCorner = max_point(points);
  PointXYZ originCorner = floor(minCorner * (1. / voxel_size)) * voxel_size;

  std::size_t sampleNX = static_cast<std::size_t>(  // x方向采集了最多多少voxel
    floor((maxCorner.x - originCorner.x) / voxel_size) + 1
  );
  std::size_t sampleNY = static_cast<std::size_t>(
    floor((maxCorner.y - originCorner.y) / voxel_size) + 1
  );

  std::size_t iX = 0;
  std::size_t iY = 0;
  std::size_t iZ = 0;
  std::size_t mapIdx = 0;
  std::unordered_map<std::size_t, SampledData> data;
  std::unordered_map<std::size_t, SampledData> dp_data;

  int NN = points.size();
  int NB = dps.size();

//  for (auto& p : points) {
    for (int i = 0; i < NN; ++i){
        PointXYZ& p = points[i];
        PointXYZ& dp = dps[i];

        iX = static_cast<std::size_t>(floor((p.x - originCorner.x) / voxel_size));  // 换成点坐标(除以体素大小)
        iY = static_cast<std::size_t>(floor((p.y - originCorner.y) / voxel_size));
        iZ = static_cast<std::size_t>(floor((p.z - originCorner.z) / voxel_size));

        // 按照层来进行归类，属于相同位置的点就会被归到一个下采样点上，为什么对这样采集到的体素进行匹配是正确的？？？
        mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

        if (!data.count(mapIdx)) {  // 如果还没有这个key
          data.emplace(mapIdx, SampledData());  // 构造键值对
          dp_data.emplace(mapIdx, SampledData());
        }

        data[mapIdx].update(p);

        dp_data[mapIdx].update(dp);
        // 可以看出下采样后的data中包含的是属于对应key的那些点的坐标的和
    }

  s_dps.reserve(dp_data.size());  // 至少分配n个元素的空间,节省了一次次扩容的时间
  for (auto& v : dp_data) {
    s_dps.push_back(v.second.point * (1.0 / v.second.count));  // 范围内点的dps平均值，为什么一定要写 *1.0/value
  }

  s_points.reserve(data.size());
  for (auto& v : data) {
    s_points.push_back(v.second.point * (1.0 / v.second.count));  // 范围内点的坐标平均值，为什么一定要写 *1.0/value？？？
  }
}

// step into 1.2.1.1.1.1 : 下采样
void grid_subsampling_cpu(
  std::vector<PointXYZ>& points,
  std::vector<PointXYZ>& s_points,
  std::vector<PointXYZ>& dps,
  std::vector<PointXYZ>& s_dps,
  std::vector<long>& lengths,
  std::vector<long>& s_lengths,
  float voxel_size
) {
  std::size_t start_index = 0;
  std::size_t batch_size = lengths.size();  //2

//  std::printf("%d\n", batch_size);

  for (std::size_t b = 0; b < batch_size; b++) {  // 循环两次,分别对src和tar下采样
    std::vector<PointXYZ> cur_points = std::vector<PointXYZ>(
      points.begin() + start_index,
      points.begin() + start_index + lengths[b]
    );
    std::vector<PointXYZ> cur_dps = std::vector<PointXYZ>(
      dps.begin() + start_index,
      dps.begin() + start_index + lengths[b]
    );
    std::vector<PointXYZ> cur_s_points;
    std::vector<PointXYZ> cur_s_dps;
//    std::vector<long> cur_s_indexes;

//    single_grid_subsampling_cpu(cur_points, cur_s_points, cur_s_indexes, voxel_size);

    // step into 1.2.1.1.1.1.1 : 下采样
    single_grid_subsampling_cpu(cur_points, cur_s_points, cur_dps, cur_s_dps, voxel_size);

    s_points.insert(s_points.end(), cur_s_points.begin(), cur_s_points.end());
    s_dps.insert(s_dps.end(), cur_s_dps.begin(), cur_s_dps.end());

    s_lengths.push_back(cur_s_points.size());

    start_index += lengths[b];
  }

  return;
}
