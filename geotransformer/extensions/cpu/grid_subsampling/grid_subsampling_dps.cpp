#include <cstring>
#include "grid_subsampling_dps.h"
#include "grid_subsampling_cpu_dps.h"

std::vector<at::Tensor> grid_subsampling_dps(
  at::Tensor points,
  at::Tensor dps,
  at::Tensor lengths,
  float voxel_size
) {
  CHECK_CPU(points);
  CHECK_CPU(lengths);
  CHECK_CPU(dps);
  CHECK_IS_FLOAT(points);
  CHECK_IS_FLOAT(dps);
  CHECK_IS_LONG(lengths);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(lengths);
  CHECK_CONTIGUOUS(dps);

  std::size_t batch_size = lengths.size(0);
  std::size_t total_points = points.size(0);

  std::vector<PointXYZ> vec_points = std::vector<PointXYZ>(  // 区间构造，从头到尾，传了两个头尾指针，利用到了内存空间连续性
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()),  // 返回tensor对应的首元素内存地址
    reinterpret_cast<PointXYZ*>(points.data_ptr<float>()) + total_points
  );
  std::vector<PointXYZ> vec_s_points;

  std::vector<PointXYZ> vec_dps = std::vector<PointXYZ>(
    reinterpret_cast<PointXYZ*>(dps.data_ptr<float>()),
    reinterpret_cast<PointXYZ*>(dps.data_ptr<float>()) + total_points
  );
  std::vector<PointXYZ> vec_s_dps;

  std::vector<long> vec_lengths = std::vector<long>(
    lengths.data_ptr<long>(),
    lengths.data_ptr<long>() + batch_size
  );
  std::vector<long> vec_s_lengths;
  // step into 1.2.1.1.1.1 : 下采样
  grid_subsampling_cpu(
    vec_points,
    vec_s_points,
    vec_dps,
    vec_s_dps,
    vec_lengths,
    vec_s_lengths,
    voxel_size
  );
  // 至此已经将下采样得到的点全部放进了vec_s_points和vec_s_dps中，方法是将每个点云中的点按照到“最小点”的坐标进行采样，取超点坐标平均值

  std::size_t total_s_points = vec_s_points.size();
  std::size_t total_s_dps = vec_s_dps.size();

  at::Tensor s_points = torch::zeros(
    {total_s_points, 3},
    at::device(points.device()).dtype(at::ScalarType::Float)
  );
  at::Tensor s_lengths = torch::zeros(
    {batch_size},
    at::device(lengths.device()).dtype(at::ScalarType::Long)
  );

  at::Tensor s_dps = torch::zeros(
    {total_s_dps, 3},
    at::device(dps.device()).dtype(at::ScalarType::Float)
  );


  std::memcpy(  // 将参数2指针指向的内存复制参数3个内存到参数1指针
    s_points.data_ptr<float>(),
    reinterpret_cast<float*>(vec_s_points.data()),
    sizeof(float) * total_s_points * 3
  );
  std::memcpy(
    s_dps.data_ptr<float>(),
    reinterpret_cast<float*>(vec_s_dps.data()),
    sizeof(float) * total_s_dps * 3
  );
  std::memcpy(
    s_lengths.data_ptr<long>(),
    vec_s_lengths.data(),
    sizeof(long) * batch_size
  );


  return {s_points, s_dps, s_lengths};
}
