from functools import partial

import numpy as np
import torch

from geotransformer.modules.ops import grid_subsample, radius_search, grid_subsample_dps
from geotransformer.utils.torch import build_dataloader
from geotransformer.modules.ops import index_select

# Stack mode utilities

def add_entropy_from_up(data_dict):  # 从上采样点计算每个下采样点的熵
    neighbors_list = data_dict['neighbors']
    subsampling_list = data_dict['subsampling']
    hsv_list = data_dict['hsv']
    entropy_list = []
    num_stages = len(neighbors_list)
    for i in range(1, num_stages):
        num_points = len(neighbors_list[i])
        num_up_points = len(neighbors_list[i - 1])
        h_up = hsv_list[i - 1][:, 0].reshape(-1, 1)
        sub2up = subsampling_list[i - 1]
        padded_sub2up = torch.cat([sub2up, num_up_points * torch.ones_like(sub2up[:1])], dim=0)
        # padded_sub2up(Nsub+1, Ksub), neighbors_list[i](Nsub, Ksub)
        patch_neighbor_indices = index_select(padded_sub2up, neighbors_list[i], dim=0).reshape(num_points, -1)  # (Nsub, Ksub, Ksub)->(Nsub, K方)
        ordered_indices, _ = torch.sort(patch_neighbor_indices)
        standard = ordered_indices.clone()
        standard[:, 1:] = standard[:, :-1]
        standard[:, 0] = -1 * torch.ones_like(standard[:, 0])
        mask = standard == ordered_indices
        filtered_indices = ordered_indices.masked_fill(mask, num_up_points)  # (Nsub, K方)
        # padded_hs = torch.cat([h_up, num_points * torch.tensor([-1]).reshape(1, 1).cuda()], dim=0)  # (Nup+1, 1)最后一行填充成-num_points
        padded_hs = torch.cat([h_up, num_points * torch.tensor([-1]).reshape(1, 1)], dim=0)  # (Nup+1, 1)最后一行填充成-num_points
        patch_local_hs = index_select(padded_hs, filtered_indices, dim=0).squeeze(-1)  # (Nsub, K方)
        add_help_tensor = torch.tensor(range(num_points)).reshape(-1, 1)
        # patch_local_hs = patch_local_hs + add_help_tensor.cuda()
        patch_local_hs = patch_local_hs + add_help_tensor
        num_row = torch.sum((filtered_indices != num_up_points), dim=1).reshape(-1, 1)
        # histogram = torch.histc(patch_local_hs, bins=6 * num_points, min=0, max=num_points).reshape(num_points, -1)
        histogram = torch.from_numpy(np.histogram(patch_local_hs.numpy(), bins=6*num_points, range=(0, num_points))[0]).reshape(num_points, -1)
        probs = histogram / num_row
        # mask = torch.isnan(probs)
        # probs[mask] = 0
        entropy_list.append(torch.distributions.Categorical(probs=probs).entropy())
        # if torch.isnan(entropy_list[-1]).sum() > 0:
        #     print('find it!')
        #     pass
    data_dict['entropy_list'] = entropy_list

def add_entropy_from_neiborhood(data_dict):
    neighbors = data_dict['neighbors'][0]
    hsv = data_dict['hsv'][0]

    num_points = len(neighbors)
    h = hsv[:, 0].reshape(-1, 1)
    padded_hs = torch.cat([h, num_points * torch.tensor([-1]).reshape(1, 1)],
                          dim=0)  # (N+1, 1)最后一行填充成-num_points

    patch_local_hs = index_select(padded_hs, neighbors, dim=0).squeeze(-1)
    add_help_tensor = torch.tensor(range(num_points)).reshape(-1, 1)
    patch_local_hs = patch_local_hs + add_help_tensor
    num_row = torch.sum((neighbors != num_points), dim=1).reshape(-1, 1)
    histogram = torch.from_numpy(
        np.histogram(patch_local_hs.numpy(), bins=6 * num_points, range=(0, num_points))[0]).reshape(num_points, -1)
    probs = histogram / num_row
    # mask = torch.isnan(probs)
    # probs[mask] = 0
    entropy = torch.distributions.Categorical(probs=probs).entropy()
    # mask = torch.isnan(entropy)
    # entropy[mask] = 0
    data_dict['entropy'] = entropy

def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits, hsv=None):
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []
    hsv_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            if hsv is not None:
                points, hsv, lengths = grid_subsample_dps(points, hsv, lengths, voxel_size=voxel_size)
            else:
                points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        if hsv is not None:
            hsv_list.append(hsv)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2
    if hsv is not None:
        ret = {
            'points': points_list,
            'hsv': hsv_list,
            'lengths': lengths_list,
            'neighbors': neighbors_list,
            'subsampling': subsampling_list,
            'upsampling': upsampling_list,
        }
        # add_entropy_from_up(ret)
        return ret
    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }


def single_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for single point cloud in stack mode.

    Points are organized in the following order: [P_1, ..., P_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool=True)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: feats, points, normals
    if 'normals' in collated_dict:
        normals = torch.cat(collated_dict.pop('normals'), dim=0)
    else:
        normals = None
    feats = torch.cat(collated_dict.pop('feats'), dim=0)
    points_list = collated_dict.pop('points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    if normals is not None:
        collated_dict['normals'] = normals
    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    hsv = None
    if 'ref_hsv' in collated_dict:
        hsv_list = collated_dict.pop('ref_hsv') + collated_dict.pop('src_hsv')
        hsv = torch.cat(hsv_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages,
                                                voxel_size, search_radius, neighbor_limits, hsv=hsv)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    search_radius,
    neighbor_limits,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader
