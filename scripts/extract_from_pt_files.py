#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 .pt 文件中提取并聚合 depth、points、poses 等数据
"""

import os
import torch
import numpy as np
from pathlib import Path
import argparse
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from tqdm import tqdm
except ImportError:
    print("警告: 未找到 tqdm，将不显示进度条")
    tqdm = lambda x, **kwargs: x
try:
    import imageio.v3 as iio
except ImportError:
    try:
        import imageio as iio
    except ImportError:
        print("警告: 未找到 imageio，将无法保存彩色图像")
        iio = None
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    print("警告: 未找到 matplotlib，将使用简单的颜色映射")
    HAS_MATPLOTLIB = False
    cm = None
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("警告: 未找到 open3d，将无法生成点云文件")
    HAS_OPEN3D = False
    o3d = None

try:
    import torch
except ImportError:
    print("错误: 未找到 torch 模块。")
    print("请确保已激活 conda 环境（例如: conda activate infinitevggt）")
    sys.exit(1)


def extract_data_from_pt_files(pt_path: str, output_path: str = None):
    """
    从 .pt 文件中提取并聚合所有数据
    
    Args:
        pt_path: .pt 文件路径或包含 .pt 文件的目录路径
        output_path: 输出文件路径（可选）
    
    Returns:
        包含所有聚合数据的字典
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise ValueError(f"路径不存在: {pt_path}")
    
    # 判断是文件还是目录
    if pt_path.is_file():
        # 如果是单个文件
        if pt_path.suffix != '.pt':
            raise ValueError(f"文件必须是 .pt 格式: {pt_path}")
        pt_files = [pt_path]
        print(f"处理单个文件: {pt_path.name}")
    elif pt_path.is_dir():
        # 如果是目录，获取所有 .pt 文件并按文件名排序
        pt_files = sorted(pt_path.glob("*.pt"))
        if not pt_files:
            raise ValueError(f"在 {pt_path} 中未找到 .pt 文件")
        print(f"找到 {len(pt_files)} 个 .pt 文件")
    else:
        raise ValueError(f"路径既不是文件也不是目录: {pt_path}")
    
    # 存储所有帧的数据
    all_depths = []
    all_depth_confs = []
    all_pts3d = []
    all_confs = []
    all_camera_poses = []
    all_extrinsics = []
    all_intrinsics = []
    all_images = []
    all_camera_poses_world = []
    
    frame_indices = []
    
    # 使用 tqdm 显示读取进度
    for pt_file in tqdm(pt_files, desc="读取 .pt 文件", unit="文件"):
        try:
            data = torch.load(pt_file, map_location='cpu', weights_only=False)
            
            # 检测文件格式：聚合文件还是单帧文件
            # 聚合文件直接包含 depth, world_points 等键，且第一个维度是批次维度
            # 单帧文件包含 pred/view/meta 结构
            is_aggregated = "pred" not in data and ("depth" in data or "world_points" in data)
            
            if is_aggregated:
                # 这是聚合文件，直接提取数据
                print(f"检测到聚合文件格式，包含多帧数据")
                
                # 首先确定帧数（从任何可用的数据中）
                num_frames = 0
                if "depth" in data and data["depth"].dim() >= 2:
                    num_frames = data["depth"].shape[0]
                elif "images" in data and data["images"].dim() >= 3:
                    num_frames = data["images"].shape[0]
                elif "world_points" in data and data["world_points"].dim() >= 2:
                    num_frames = data["world_points"].shape[0]
                elif "pose_enc" in data and data["pose_enc"].dim() >= 2:
                    num_frames = data["pose_enc"].shape[0]
                elif "extrinsic" in data and data["extrinsic"].dim() >= 2:
                    num_frames = data["extrinsic"].shape[0]
                
                if num_frames == 0:
                    print("警告: 无法确定帧数，假设为单帧")
                    num_frames = 1
                
                print(f"检测到 {num_frames} 帧数据")
                
                # 初始化 frame_indices
                frame_indices = list(range(num_frames))
                
                # 从聚合文件中提取数据
                if "depth" in data:
                    depth = data["depth"]
                    # 处理各种可能的深度图形状
                    if depth.dim() == 5:  # (1, N, H, W, 1) 或类似
                        depth = depth.squeeze(0).squeeze(-1)  # 去掉第一个和最后一个维度
                    elif depth.dim() == 4:  # (N, H, W, 1) 或 (1, N, H, W)
                        if depth.shape[0] == 1:
                            depth = depth.squeeze(0)
                        if depth.dim() == 4 and depth.shape[-1] == 1:
                            depth = depth.squeeze(-1)
                    # 如果是 (N, H, W)，按帧拆分
                    if depth.dim() == 3:
                        for i in range(depth.shape[0]):
                            all_depths.append(depth[i])
                    else:
                        all_depths.append(depth)
                
                if "depth_conf" in data:
                    depth_conf = data["depth_conf"]
                    if depth_conf.dim() == 3:
                        for i in range(depth_conf.shape[0]):
                            all_depth_confs.append(depth_conf[i])
                    else:
                        all_depth_confs.append(depth_conf)
                
                if "world_points" in data:
                    pts3d = data["world_points"]
                    if pts3d.dim() == 4:  # (N, H, W, 3)
                        for i in range(pts3d.shape[0]):
                            all_pts3d.append(pts3d[i])
                    elif pts3d.dim() == 3:  # (N, M, 3)
                        for i in range(pts3d.shape[0]):
                            all_pts3d.append(pts3d[i])
                    else:
                        all_pts3d.append(pts3d)
                
                if "world_points_conf" in data:
                    conf = data["world_points_conf"]
                    if conf.dim() == 3:
                        for i in range(conf.shape[0]):
                            all_confs.append(conf[i])
                    elif conf.dim() == 2:
                        for i in range(conf.shape[0]):
                            all_confs.append(conf[i])
                    else:
                        all_confs.append(conf)
                
                if "pose_enc" in data:
                    camera_pose = data["pose_enc"]
                    if camera_pose.dim() == 3:
                        for i in range(camera_pose.shape[0]):
                            all_camera_poses.append(camera_pose[i])
                    else:
                        all_camera_poses.append(camera_pose)
                
                if "extrinsic" in data:
                    extrinsic = data["extrinsic"]
                    if extrinsic.dim() == 3:
                        for i in range(extrinsic.shape[0]):
                            all_extrinsics.append(extrinsic[i])
                    else:
                        all_extrinsics.append(extrinsic)
                
                if "intrinsic" in data:
                    intrinsic = data["intrinsic"]
                    if intrinsic is not None:
                        if intrinsic.dim() == 3:
                            for i in range(intrinsic.shape[0]):
                                all_intrinsics.append(intrinsic[i])
                        elif intrinsic.dim() == 2:
                            # 单帧的内参，复制给所有帧
                            for i in range(num_frames):
                                all_intrinsics.append(intrinsic)
                        else:
                            all_intrinsics.append(intrinsic)
                
                if "images" in data:
                    images = data["images"]
                    if images.dim() == 4:  # (N, C, H, W) or (N, H, W, C)
                        for i in range(images.shape[0]):
                            all_images.append(images[i])
                    else:
                        all_images.append(images)
                
                # 验证数据一致性
                if all_depths and len(all_depths) != num_frames:
                    print(f"警告: 深度图数量 ({len(all_depths)}) 与帧数 ({num_frames}) 不匹配")
                if all_images and len(all_images) != num_frames:
                    print(f"警告: 图像数量 ({len(all_images)}) 与帧数 ({num_frames}) 不匹配")
                
                # 聚合文件处理完成，跳出循环（只处理第一个文件）
                break
            else:
                # 这是单帧文件，按原来的方式处理
                # 提取帧索引
                frame_idx = data.get("meta", {}).get("frame_idx", None)
                if frame_idx is None:
                    # 尝试从文件名提取
                    try:
                        frame_idx = int(pt_file.stem)
                    except ValueError:
                        # 如果文件名不是数字，使用索引
                        frame_idx = len(frame_indices)
                frame_indices.append(frame_idx)
                
                # 提取 pred 部分的数据
                pred = data.get("pred", {})
                
                # 提取深度图
                if "depth" in pred:
                    depth = pred["depth"]
                    # 如果是 (1, H, W) 或 (B, H, W)，去掉 batch 维度
                    if depth.dim() == 3 and depth.shape[0] == 1:
                        depth = depth.squeeze(0)
                    all_depths.append(depth)
                
                # 提取深度置信度
                if "depth_conf" in pred:
                    depth_conf = pred["depth_conf"]
                    if depth_conf.dim() == 3 and depth_conf.shape[0] == 1:
                        depth_conf = depth_conf.squeeze(0)
                    all_depth_confs.append(depth_conf)
                
                # 提取 3D 点
                if "pts3d_in_other_view" in pred:
                    pts3d = pred["pts3d_in_other_view"]
                    if pts3d.dim() == 4 and pts3d.shape[0] == 1:  # (1, H, W, 3)
                        pts3d = pts3d.squeeze(0)
                    elif pts3d.dim() == 3 and pts3d.shape[0] == 1:  # (1, N, 3)
                        pts3d = pts3d.squeeze(0)
                    all_pts3d.append(pts3d)
                
                # 提取置信度
                if "conf" in pred:
                    conf = pred["conf"]
                    if conf.dim() == 3 and conf.shape[0] == 1:
                        conf = conf.squeeze(0)
                    elif conf.dim() == 2 and conf.shape[0] == 1:
                        conf = conf.squeeze(0)
                    all_confs.append(conf)
                
                # 提取相机位姿编码
                if "camera_pose" in pred:
                    camera_pose = pred["camera_pose"]
                    if camera_pose.dim() == 3 and camera_pose.shape[0] == 1:
                        camera_pose = camera_pose.squeeze(0)
                    all_camera_poses.append(camera_pose)
                
                # 提取外参矩阵
                if "extrinsic_world_to_cam" in pred:
                    extrinsic = pred["extrinsic_world_to_cam"]
                    if extrinsic.dim() == 3 and extrinsic.shape[0] == 1:
                        extrinsic = extrinsic.squeeze(0)
                    all_extrinsics.append(extrinsic)
                
                # 提取内参矩阵
                if "intrinsic" in pred:
                    intrinsic = pred["intrinsic"]
                    if intrinsic.dim() == 3 and intrinsic.shape[0] == 1:
                        intrinsic = intrinsic.squeeze(0)
                    all_intrinsics.append(intrinsic)
                
                # 提取 view 部分的数据
                view = data.get("view", {})
                
                # 提取图像
                if "img" in view:
                    img = view["img"]
                    if img.dim() == 4 and img.shape[0] == 1:  # (1, C, H, W)
                        img = img.squeeze(0)
                    all_images.append(img)
                
                # 提取世界坐标系下的相机位姿
                if "camera_pose" in view:
                    cam_pose_world = view["camera_pose"]
                    if cam_pose_world.dim() == 3 and cam_pose_world.shape[0] == 1:
                        cam_pose_world = cam_pose_world.squeeze(0)
                    all_camera_poses_world.append(cam_pose_world)
            
        except Exception as e:
            print(f"警告: 读取 {pt_file} 时出错: {e}")
            continue
    
    print(f"成功读取 {len(frame_indices)} 个文件")
    
    # 聚合所有数据
    aggregated_data = {
        "frame_indices": np.array(frame_indices),
    }
    
    if all_depths:
        aggregated_data["depth"] = torch.stack(all_depths, dim=0)
        print(f"  深度图: {aggregated_data['depth'].shape}")
    
    if all_depth_confs:
        aggregated_data["depth_conf"] = torch.stack(all_depth_confs, dim=0)
        print(f"  深度置信度: {aggregated_data['depth_conf'].shape}")
    
    if all_pts3d:
        # 3D点可能是不同形状的，需要特殊处理
        try:
            aggregated_data["world_points"] = torch.stack(all_pts3d, dim=0)
            print(f"  3D点: {aggregated_data['world_points'].shape}")
        except RuntimeError as e:
            print(f"  警告: 无法堆叠3D点（形状不一致）: {e}")
            aggregated_data["world_points"] = all_pts3d  # 保存为列表
    
    if all_confs:
        try:
            aggregated_data["world_points_conf"] = torch.stack(all_confs, dim=0)
            print(f"  点置信度: {aggregated_data['world_points_conf'].shape}")
        except RuntimeError as e:
            print(f"  警告: 无法堆叠点置信度（形状不一致）: {e}")
            aggregated_data["world_points_conf"] = all_confs
    
    if all_camera_poses:
        aggregated_data["pose_enc"] = torch.stack(all_camera_poses, dim=0)
        print(f"  位姿编码: {aggregated_data['pose_enc'].shape}")
    
    if all_extrinsics:
        aggregated_data["extrinsic"] = torch.stack(all_extrinsics, dim=0)
        print(f"  外参矩阵: {aggregated_data['extrinsic'].shape}")
    
    if all_intrinsics:
        aggregated_data["intrinsic"] = torch.stack(all_intrinsics, dim=0)
        print(f"  内参矩阵: {aggregated_data['intrinsic'].shape}")
    
    if all_images:
        aggregated_data["images"] = torch.stack(all_images, dim=0)
        print(f"  图像: {aggregated_data['images'].shape}")
    
    if all_camera_poses_world:
        aggregated_data["camera_pose_world"] = torch.stack(all_camera_poses_world, dim=0)
        print(f"  世界坐标系相机位姿: {aggregated_data['camera_pose_world'].shape}")
    
    # 保存到文件
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 将所有张量移到 CPU（如果还没有）
        for key, value in aggregated_data.items():
            if isinstance(value, torch.Tensor):
                aggregated_data[key] = value.cpu()
        
        torch.save(aggregated_data, output_path)
        print(f"\n数据已保存到: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    return aggregated_data


def save_to_directory_format(data: dict, out_dir: str):
    """
    将数据保存为文件夹格式（depth, conf, color, camera）
    
    Args:
        data: 包含所有数据的字典
        out_dir: 输出目录路径
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    subdirs = {
        'depth': out_dir / 'depth',
        'conf': out_dir / 'conf',
        'color': out_dir / 'color',
        'camera': out_dir / 'camera'
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # 获取数据
    depth = data.get("depth")
    depth_conf = data.get("depth_conf")
    conf = data.get("world_points_conf")
    images = data.get("images")
    extrinsic = data.get("extrinsic")
    intrinsic = data.get("intrinsic")
    camera_pose_world = data.get("camera_pose_world")
    
    # 确定帧数
    if depth is not None:
        N = depth.shape[0]
    elif images is not None:
        N = images.shape[0]
    elif extrinsic is not None:
        N = extrinsic.shape[0]
    else:
        print("错误: 无法确定帧数")
        return
    
    print(f"保存 {N} 帧数据到 {out_dir}")
    
    # 处理相机位姿
    if extrinsic is not None:
        extrinsic_np = extrinsic.cpu().numpy() if isinstance(extrinsic, torch.Tensor) else extrinsic
        # 转换 w2c 到 c2w
        if extrinsic_np.shape[-2:] == (3, 4):
            # (N, 3, 4) -> (N, 4, 4)
            w2c = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
            w2c[:, :3, :] = extrinsic_np
        elif extrinsic_np.shape[-2:] == (4, 4):
            if extrinsic_np.ndim == 2:
                # 单帧 (4, 4)，扩展为 (N, 4, 4)
                w2c = extrinsic_np[None].repeat(N, 0)
            else:
                # 多帧 (N, 4, 4)
                w2c = extrinsic_np
        else:
            print(f"警告: 无法处理外参矩阵形状: {extrinsic_np.shape}")
            w2c = None
        if w2c is not None:
            c2w = np.linalg.inv(w2c)
        else:
            c2w = None
    elif camera_pose_world is not None:
        c2w = camera_pose_world.cpu().numpy() if isinstance(camera_pose_world, torch.Tensor) else camera_pose_world
        if c2w.shape[-2:] == (3, 4):
            # 转换为 4x4
            c2w_4x4 = np.eye(4, dtype=np.float32)[None].repeat(N, 0)
            c2w_4x4[:, :3, :] = c2w
            c2w = c2w_4x4
    else:
        print("警告: 未找到相机位姿数据")
        c2w = None
    
    # 处理内参
    if intrinsic is not None:
        intrinsic_np = intrinsic.cpu().numpy() if isinstance(intrinsic, torch.Tensor) else intrinsic
        if intrinsic_np.shape[-2:] == (3, 3):
            focal = intrinsic_np[:, 0, 0]
            pp = intrinsic_np[:, :2, 2]
        else:
            focal = None
            pp = None
    else:
        focal = None
        pp = None
    
    # 保存每一帧，使用 tqdm 显示保存进度
    for i in tqdm(range(N), desc="保存数据", unit="帧"):
        frame_idx = i
        
        # 保存深度图
        if depth is not None:
            depth_i = depth[i]
            if isinstance(depth_i, torch.Tensor):
                depth_i = depth_i.cpu().numpy()
            np.save(subdirs['depth'] / f"{frame_idx:06d}.npy", depth_i)
        
        # 保存深度置信度
        if depth_conf is not None:
            depth_conf_i = depth_conf[i]
            if isinstance(depth_conf_i, torch.Tensor):
                depth_conf_i = depth_conf_i.cpu().numpy()
            np.save(subdirs['conf'] / f"{frame_idx:06d}.npy", depth_conf_i)
        # 或者保存点云置信度
        elif conf is not None:
            conf_i = conf[i]
            if isinstance(conf_i, torch.Tensor):
                conf_i = conf_i.cpu().numpy()
            # 如果是点云置信度，可能需要 reshape
            if conf_i.ndim == 0:
                # 标量，跳过
                pass
            else:
                np.save(subdirs['conf'] / f"{frame_idx:06d}.npy", conf_i)
        
        # 保存彩色图像
        if images is not None and iio is not None:
            img_i = images[i]
            if isinstance(img_i, torch.Tensor):
                img_i = img_i.cpu().numpy()
            
            # 处理图像格式：确保是 (H, W, C) 格式
            if img_i.ndim == 3:
                if img_i.shape[0] == 3 and img_i.shape[2] != 3:  # (C, H, W)
                    img_i = img_i.transpose(1, 2, 0)  # (H, W, C)
                elif img_i.shape[-1] == 3:  # 已经是 (H, W, C)
                    pass
                elif img_i.shape[0] == 3:  # (C, H, W)
                    img_i = img_i.transpose(1, 2, 0)
            
            # 确保是 RGB 格式（不是 BGR）
            if img_i.shape[-1] == 3:
                # 如果通道顺序是 BGR，转换为 RGB
                # 这里假设已经是 RGB，如果需要可以添加转换
                pass
            
            # 归一化到 [0, 255]
            if img_i.max() <= 1.0:
                img_i = (img_i * 255).astype(np.uint8)
            else:
                img_i = np.clip(img_i, 0, 255).astype(np.uint8)
            
            iio.imwrite(subdirs['color'] / f"{frame_idx:06d}.png", img_i)
        
        # 保存相机参数
        if c2w is not None:
            pose = c2w[i]
            
            # 构建内参矩阵
            if focal is not None and pp is not None:
                intr = np.eye(3, dtype=np.float32)
                intr[0, 0] = intr[1, 1] = focal[i]
                intr[:2, 2] = pp[i]
            elif intrinsic_np is not None:
                if intrinsic_np.shape[-2:] == (3, 3):
                    intr = intrinsic_np[i]
                else:
                    intr = np.eye(3, dtype=np.float32)
            else:
                intr = np.eye(3, dtype=np.float32)
            
            np.savez(subdirs['camera'] / f"{frame_idx:06d}.npz", pose=pose, intrinsics=intr)
    
    print(f"数据已保存到: {out_dir}")
    print(f"  - depth: {subdirs['depth']}")
    print(f"  - conf: {subdirs['conf']}")
    print(f"  - color: {subdirs['color']}")
    print(f"  - camera: {subdirs['camera']}")
    
    # 自动生成深度图可视化（在 depth 同级目录创建 depth_vis）
    try:
        visualize_depth_folder(str(subdirs['depth']))
    except Exception as e:
        print(f"警告: 生成深度图可视化时出错: {e}")
    
    # 自动生成相机轨迹可视化
    try:
        visualize_camera_trajectory(str(subdirs['camera']))
    except Exception as e:
        print(f"警告: 生成相机轨迹可视化时出错: {e}")
    
    # 自动生成点云（融合所有帧，使用体素降采样）
    try:
        generate_point_clouds(
            str(subdirs['depth']),
            str(subdirs['camera']),
            str(subdirs['color']) if 'color' in subdirs else None,
            str(out_dir / "pointclouds"),
            voxel_size=0.005  # 2cm 体素大小，只对合并点云降采样，单帧保持原始密度
        )
    except Exception as e:
        print(f"警告: 生成点云时出错: {e}")


def visualize_depth_folder(depth_dir: str, output_dir: str = None):
    """
    将 depth 文件夹中的深度图可视化（保存为灰度图）
    
    Args:
        depth_dir: depth 文件夹路径
        output_dir: 输出文件夹路径（默认为 depth_dir 的父目录下的 depth_vis，即与 depth 同级）
    """
    depth_dir = Path(depth_dir)
    if not depth_dir.exists():
        raise ValueError(f"目录不存在: {depth_dir}")
    
    if not depth_dir.is_dir():
        raise ValueError(f"路径不是目录: {depth_dir}")
    
    # 确定输出目录：默认在 depth 文件夹的同级目录创建 depth_vis
    if output_dir is None:
        output_dir = depth_dir.parent / "depth_vis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 .npy 文件
    depth_files = sorted(depth_dir.glob("*.npy"))
    
    if not depth_files:
        raise ValueError(f"在 {depth_dir} 中未找到 .npy 文件")
    
    print(f"找到 {len(depth_files)} 个深度图文件")
    
    # 首先计算所有深度图的范围（用于统一颜色映射）
    all_depths = []
    for depth_file in depth_files:
        depth = np.load(depth_file)
        # 过滤无效值
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            all_depths.append(valid_depth)
    
    if len(all_depths) == 0:
        raise ValueError("所有深度图都无效")
    
    all_depths = np.concatenate(all_depths)
    vmin = np.percentile(all_depths, 1)
    vmax = np.percentile(all_depths, 99)
    
    print(f"深度范围: [{vmin:.3f}, {vmax:.3f}]")
    
    # 可视化函数 - 保存为灰度图
    def colorize_depth(depth, vmin, vmax):
        """将深度图转换为灰度图像"""
        # 处理不同形状的深度图，确保是 2D (H, W)
        depth = np.asarray(depth)
        if depth.ndim > 2:
            # 如果是 (1, H, W) 或 (H, W, 1) 等，压缩到 2D
            depth = depth.squeeze()
        if depth.ndim != 2:
            raise ValueError(f"深度图应该是 2D 数组，但得到形状: {depth.shape}")
        
        H, W = depth.shape
        
        # 创建掩码（有效深度值）
        mask = depth > 0
        
        # 归一化深度值到 [0, 1]
        # 近处深度值小 -> 0（黑色），远处深度值大 -> 1（白色）
        depth_normalized = np.clip(depth, vmin, vmax)
        depth_normalized = (depth_normalized - vmin) / (vmax - vmin + 1e-8)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        
        # 转换为灰度图 (H, W)
        # 近处（深度值小）-> 黑色（0），远处（深度值大）-> 白色（255）
        gray = depth_normalized
        
        # 将无效区域设为黑色
        gray[~mask] = 0
        
        # 转换为 uint8 格式 (0-255)
        return (gray * 255).astype(np.uint8)
    
    # 处理每个深度图的辅助函数（用于多进程）
    def process_single_depth_vis(args):
        depth_file, output_dir, vmin, vmax = args
        try:
            depth = np.load(depth_file)
            gray_depth = colorize_depth(depth, vmin, vmax)
            output_file = output_dir / f"{depth_file.stem}.jpg"
            if iio is not None:
                iio.imwrite(output_file, gray_depth)
            else:
                try:
                    import cv2
                    cv2.imwrite(str(output_file), gray_depth)
                except ImportError:
                    pass
            return True
        except Exception as e:
            print(f"警告: 处理 {depth_file} 时出错: {e}")
            return False
    
    # 并行处理深度图可视化
    num_workers_vis = max(1, min(cpu_count() // 2, len(depth_files)))
    tasks_vis = [(df, output_dir, vmin, vmax) for df in depth_files]
    
    with Pool(processes=num_workers_vis) as pool:
        list(tqdm(
            pool.imap(process_single_depth_vis, tasks_vis),
            total=len(depth_files),
            desc="可视化深度图",
            unit="文件"
        ))
    
    print(f"\n深度图可视化已保存到: {output_dir}")
    print(f"共保存 {len(depth_files)} 个可视化图像")


def visualize_camera_trajectory(camera_dir: str, output_file: str = None):
    """
    可视化相机轨迹
    
    Args:
        camera_dir: camera 文件夹路径
        output_file: 输出图像文件路径（默认为 camera_dir 的父目录下的 camera_trajectory.png）
    """
    if not HAS_MATPLOTLIB:
        print("警告: 需要 matplotlib 来可视化相机轨迹")
        return
    
    camera_dir = Path(camera_dir)
    if not camera_dir.exists():
        raise ValueError(f"目录不存在: {camera_dir}")
    
    if not camera_dir.is_dir():
        raise ValueError(f"路径不是目录: {camera_dir}")
    
    # 确定输出文件
    if output_file is None:
        output_file = camera_dir.parent / "camera_trajectory.png"
    else:
        output_file = Path(output_file)
    
    # 获取所有 .npz 文件
    camera_files = sorted(camera_dir.glob("*.npz"))
    
    if not camera_files:
        raise ValueError(f"在 {camera_dir} 中未找到 .npz 文件")
    
    print(f"找到 {len(camera_files)} 个相机文件")
    
    # 提取所有相机位置
    positions = []
    orientations = []
    
    for camera_file in tqdm(camera_files, desc="读取相机位姿", unit="文件"):
        try:
            data = np.load(camera_file)
            pose = data['pose']  # 应该是 4x4 的 c2w 矩阵
            
            # 提取位置（平移部分）
            position = pose[:3, 3]
            positions.append(position)
            
            # 提取朝向（旋转矩阵的前向向量，即 z 轴方向）
            forward = pose[:3, 2]  # 相机的前向方向（在 c2w 中）
            orientations.append(forward)
        except Exception as e:
            print(f"警告: 读取 {camera_file} 时出错: {e}")
            continue
    
    if len(positions) == 0:
        raise ValueError("未能读取任何相机位姿")
    
    positions = np.array(positions)  # (N, 3)
    orientations = np.array(orientations)  # (N, 3)
    
    print(f"成功读取 {len(positions)} 个相机位姿")
    print(f"位置范围: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
          f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
          f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    
    # 创建 3D 轨迹图
    fig = plt.figure(figsize=(12, 10))
    
    # 选择最佳视角（通常是 XZ 或 XY 平面）
    # 计算轨迹的主要方向
    pos_range = positions.max(axis=0) - positions.min(axis=0)
    main_plane_idx = np.argmax(pos_range[:2])  # 选择 X 或 Y 作为主要方向
    
    if main_plane_idx == 0:
        # XZ 平面视图
        ax = fig.add_subplot(111)
        ax.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Trajectory', alpha=0.7)
        ax.scatter(positions[:, 0], positions[:, 2], c=range(len(positions)), 
                  cmap='viridis', s=30, alpha=0.8, label='Camera positions')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title('Camera Trajectory (XZ view)', fontsize=14)
    else:
        # XY 平面视图
        ax = fig.add_subplot(111)
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory', alpha=0.7)
        ax.scatter(positions[:, 0], positions[:, 1], c=range(len(positions)), 
                  cmap='viridis', s=30, alpha=0.8, label='Camera positions')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Camera Trajectory (XY view)', fontsize=14)
    
    # 添加起点和终点标记
    ax.scatter(positions[0, 0], positions[0, 2] if main_plane_idx == 0 else positions[0, 1], 
              c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 2] if main_plane_idx == 0 else positions[-1, 1], 
              c='red', s=100, marker='s', label='End', zorder=5)
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n相机轨迹可视化已保存到: {output_file}")
    
    # 同时创建 3D 视图（如果可能）
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig_3d = plt.figure(figsize=(12, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # 绘制 3D 轨迹
        ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'b-', linewidth=2, alpha=0.7, label='Trajectory')
        ax_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                     c=range(len(positions)), cmap='viridis', s=30, alpha=0.8)
        
        # 标记起点和终点
        ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                     c='green', s=100, marker='o', label='Start')
        ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                     c='red', s=100, marker='s', label='End')
        
        ax_3d.set_xlabel('X (m)', fontsize=12)
        ax_3d.set_ylabel('Y (m)', fontsize=12)
        ax_3d.set_zlabel('Z (m)', fontsize=12)
        ax_3d.set_title('Camera Trajectory (3D view)', fontsize=14)
        ax_3d.legend()
        
        output_file_3d = output_file.parent / "camera_trajectory_3d.png"
        plt.tight_layout()
        plt.savefig(output_file_3d, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"3D 相机轨迹可视化已保存到: {output_file_3d}")
    except ImportError:
        print("注意: 无法创建 3D 视图（需要 mpl_toolkits.mplot3d）")


def process_single_frame_pointcloud(args):
    """处理单帧点云的辅助函数（用于多进程）"""
    i, depth_file, camera_file, color_dir, voxel_size = args
    
    try:
        # 读取深度图
        depth = np.load(depth_file)
        if depth.ndim > 2:
            depth = depth.squeeze()
        
        # 读取相机参数
        camera_data = np.load(camera_file)
        pose = camera_data['pose']  # c2w 矩阵 (4, 4)
        intrinsics = camera_data['intrinsics']  # (3, 3)
        
        # 反投影到相机坐标系
        points_cam = depth_to_camera_coords(depth, intrinsics)
        
        # 转换到世界坐标系
        H, W = depth.shape
        points_cam_flat = points_cam.reshape(-1, 3)
        depth_flat = depth.reshape(-1)
        
        # 过滤无效深度
        valid_mask = depth_flat > 0
        points_cam_valid = points_cam_flat[valid_mask]
        
        if len(points_cam_valid) == 0:
            return None, None, i
        
        # 转换为齐次坐标
        points_cam_homo = np.hstack([points_cam_valid, np.ones((len(points_cam_valid), 1))])
        
        # 转换到世界坐标系
        points_world_homo = (pose @ points_cam_homo.T).T
        points_world = points_world_homo[:, :3]
        
        # 获取颜色（如果有）
        if color_dir is not None:
            color_file = color_dir / f"{depth_file.stem}.png"
            if color_file.exists() and iio:
                color_img = iio.imread(color_file)
                if color_img.ndim == 3 and color_img.shape[2] == 3:
                    color_flat = color_img.reshape(-1, 3)[valid_mask]
                    colors = color_flat.astype(np.float32) / 255.0
                else:
                    colors = None
            else:
                colors = None
        else:
            colors = None
        
        # 如果没有颜色，使用深度值作为颜色（灰度）
        if colors is None:
            depth_valid = depth_flat[valid_mask]
            depth_normalized = (depth_valid - depth_valid.min()) / (depth_valid.max() - depth_valid.min() + 1e-8)
            colors = np.stack([depth_normalized, depth_normalized, depth_normalized], axis=1)
        
        return points_world, colors, i
    except Exception as e:
        print(f"警告: 处理第 {i} 帧时出错: {e}")
        return None, None, i


def depth_to_camera_coords(depth, intrinsic):
    """
    将深度图反投影到相机坐标系
    
    Args:
        depth: 深度图 (H, W)
        intrinsic: 相机内参矩阵 (3, 3)
    
    Returns:
        相机坐标系下的3D点 (H, W, 3)
    """
    H, W = depth.shape
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 反投影到相机坐标系
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    # 组合成3D点
    points_cam = np.stack([x, y, z], axis=-1)
    
    return points_cam


def generate_point_clouds(depth_dir: str, camera_dir: str, color_dir: str = None, output_dir: str = None, voxel_size: float = 0.04, num_workers: int = None):
    """
    从深度图、相机位姿和图像生成点云（PCD格式）
    
    Args:
        depth_dir: depth 文件夹路径
        camera_dir: camera 文件夹路径
        color_dir: color 文件夹路径（可选）
        output_dir: 输出目录路径（默认为 depth_dir 的父目录下的 pointclouds）
        voxel_size: 体素降采样大小（米），默认 0.04m (4cm)，设为 None 则不降采样
        num_workers: 并行处理的进程数，默认使用一半CPU核心
    """
    if not HAS_OPEN3D:
        print("警告: 需要 open3d 来生成点云文件")
        return
    
    depth_dir = Path(depth_dir)
    camera_dir = Path(camera_dir)
    color_dir = Path(color_dir) if color_dir else None
    
    if not depth_dir.exists():
        raise ValueError(f"目录不存在: {depth_dir}")
    if not camera_dir.exists():
        raise ValueError(f"目录不存在: {camera_dir}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = depth_dir.parent / "pointclouds"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有文件
    depth_files = sorted(depth_dir.glob("*.npy"))
    camera_files = sorted(camera_dir.glob("*.npz"))
    
    if len(depth_files) != len(camera_files):
        print(f"警告: 深度图数量 ({len(depth_files)}) 与相机文件数量 ({len(camera_files)}) 不匹配")
    
    num_files = min(len(depth_files), len(camera_files))
    print(f"找到 {num_files} 个深度图和相机文件对")
    
    # 确定工作进程数
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)  # 使用一半的CPU核心
    num_workers = min(num_workers, num_files)  # 不超过文件数量
    
    print(f"使用 {num_workers} 个进程并行处理...")
    
    # 准备任务参数
    tasks = [
        (i, depth_files[i], camera_files[i], color_dir, voxel_size)
        for i in range(num_files)
    ]
    
    # 并行处理所有帧
    all_points = [None] * num_files
    all_colors = [None] * num_files
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_frame_pointcloud, tasks),
            total=num_files,
            desc="生成点云",
            unit="帧"
        ))
    
    # 整理结果并先保存单帧点云
    valid_frames = []
    print("\n保存每帧的点云...")
    for points_world, colors, i in tqdm(results, desc="保存单帧点云", unit="帧"):
        if points_world is not None:
            all_points[i] = points_world
            all_colors[i] = colors
            valid_frames.append(i)
            
            # 立即保存单帧点云
            frame_pcd = o3d.geometry.PointCloud()
            frame_pcd.points = o3d.utility.Vector3dVector(points_world)
            frame_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 单帧点云不降采样，保持原始密度（只在合并时降采样）
            
            frame_pcd_file = output_dir / f"frame_{i:06d}.pcd"
            o3d.io.write_point_cloud(str(frame_pcd_file), frame_pcd)
    
    # 过滤掉 None 值
    all_points = [p for p in all_points if p is not None]
    all_colors = [c for c in all_colors if c is not None]
    
    if len(all_points) == 0:
        print("错误: 未能生成任何点云")
        return
    
    print(f"已保存 {len(valid_frames)} 帧的单帧点云")
    
    # 合并所有点云
    print("\n合并所有点云...")
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    print(f"合并前总点数: {len(merged_points)}")
    
    # 如果点数太多，先进行粗略的均匀采样以加速后续处理
    MAX_POINTS_FOR_O3D = 10_000_000  # 1000万点
    if len(merged_points) > MAX_POINTS_FOR_O3D:
        print(f"点数过多 ({len(merged_points)}), 先进行粗略采样...")
        step = len(merged_points) // MAX_POINTS_FOR_O3D
        indices = np.arange(0, len(merged_points), step)
        merged_points = merged_points[indices]
        merged_colors = merged_colors[indices]
        print(f"粗略采样后点数: {len(merged_points)}")
    
    # 降采样（如果指定了体素大小）
    if voxel_size is not None and voxel_size > 0:
        print(f"使用体素降采样 (voxel_size={voxel_size}m)...")
        # 先创建临时点云进行降采样
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(merged_points)
        temp_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        
        pcd_downsampled = temp_pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"降采样后点数: {len(pcd_downsampled.points)} (减少了 {len(merged_points) - len(pcd_downsampled.points)} 个点)")
        pcd = pcd_downsampled
    else:
        # 创建 Open3D 点云对象
        print("创建点云对象...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    # 保存合并的点云
    merged_pcd_file = output_dir / "merged_pointcloud.pcd"
    o3d.io.write_point_cloud(str(merged_pcd_file), pcd)
    print(f"\n合并的点云已保存到: {merged_pcd_file}")
    
    print(f"\n所有点云已保存到: {output_dir}")
    print(f"  - 合并点云: merged_pointcloud.pcd")
    print(f"  - 单帧点云: frame_XXXXXX.pcd (共 {len(valid_frames)} 帧)")


def main():
    parser = argparse.ArgumentParser(
        description="从 .pt 文件中提取并聚合数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pt_dir",
        type=str,
        required=True,
        help=".pt 文件路径或包含 .pt 文件的目录路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（.pt 格式）。如果不指定，只显示统计信息"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录路径（保存为文件夹格式：depth/, conf/, color/, camera/，并自动生成 depth_vis/）"
    )
    
    args = parser.parse_args()
    
    try:
        # 如果指定了 output_dir，不保存 .pt 文件
        output_pt = args.output if not args.output_dir else None
        data = extract_data_from_pt_files(args.pt_dir, output_pt)
        
        # 如果指定了 output_dir，保存为文件夹格式（会自动生成 depth_vis）
        if args.output_dir:
            save_to_directory_format(data, args.output_dir)
        
        print("\n" + "="*60)
        print("提取完成！")
        print("="*60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

