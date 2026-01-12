#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 output/HKU_Landmark 目录中的 .pt 文件内容
检查是否包含点云和位姿数据
"""

import os
import sys

try:
    import torch
except ImportError:
    print("错误: 未找到 torch 模块。")
    print("请确保已激活 conda 环境（例如: conda activate infinitevggt）")
    print("或者安装 PyTorch: pip install torch")
    sys.exit(1)

import numpy as np
from pathlib import Path
from typing import Dict, Any
import argparse


def analyze_pt_file(file_path: str) -> Dict[str, Any]:
    """
    分析单个 .pt 文件的内容
    
    Args:
        file_path: .pt 文件路径
        
    Returns:
        包含文件信息的字典
    """
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        info = {
            "file_path": file_path,
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "keys": list(data.keys()) if isinstance(data, dict) else "Not a dict",
            "structure": {}
        }
        
        if isinstance(data, dict):
            # 分析 pred 部分
            if "pred" in data:
                pred = data["pred"]
                info["structure"]["pred"] = {
                    "keys": list(pred.keys()) if isinstance(pred, dict) else "Not a dict",
                    "tensor_info": {}
                }
                if isinstance(pred, dict):
                    for key, value in pred.items():
                        if isinstance(value, torch.Tensor):
                            info["structure"]["pred"]["tensor_info"][key] = {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "min": float(value.min().item()) if value.numel() > 0 else None,
                                "max": float(value.max().item()) if value.numel() > 0 else None,
                            }
                        else:
                            info["structure"]["pred"]["tensor_info"][key] = {
                                "type": type(value).__name__,
                                "value": str(value)[:100] if not isinstance(value, (list, dict)) else f"{type(value).__name__} with {len(value)} items"
                            }
            
            # 分析 view 部分
            if "view" in data:
                view = data["view"]
                info["structure"]["view"] = {
                    "keys": list(view.keys()) if isinstance(view, dict) else "Not a dict",
                    "tensor_info": {}
                }
                if isinstance(view, dict):
                    for key, value in view.items():
                        if isinstance(value, torch.Tensor):
                            info["structure"]["view"]["tensor_info"][key] = {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "min": float(value.min().item()) if value.numel() > 0 else None,
                                "max": float(value.max().item()) if value.numel() > 0 else None,
                            }
                        else:
                            info["structure"]["view"]["tensor_info"][key] = {
                                "type": type(value).__name__,
                                "value": str(value)[:100] if not isinstance(value, (list, dict)) else f"{type(value).__name__} with {len(value)} items"
                            }
            
            # 分析 meta 部分
            if "meta" in data:
                meta = data["meta"]
                info["structure"]["meta"] = {
                    "keys": list(meta.keys()) if isinstance(meta, dict) else "Not a dict",
                    "content": {}
                }
                if isinstance(meta, dict):
                    for key, value in meta.items():
                        if isinstance(value, torch.Tensor):
                            info["structure"]["meta"]["content"][key] = {
                                "type": "tensor",
                                "shape": list(value.shape),
                                "value": value.item() if value.numel() == 1 else f"tensor with shape {list(value.shape)}"
                            }
                        else:
                            info["structure"]["meta"]["content"][key] = value
        
        return info
    
    except Exception as e:
        return {
            "file_path": file_path,
            "error": str(e)
        }


def check_point_cloud_data(data: Dict[str, Any]) -> Dict[str, bool]:
    """
    检查数据中是否包含点云相关信息
    
    Args:
        data: 分析后的文件信息
        
    Returns:
        包含检查结果的字典
    """
    result = {
        "has_point_cloud": False,
        "has_depth": False,
        "has_3d_points": False,
        "has_pose": False,
        "has_camera_pose": False,
        "has_extrinsic": False,
        "has_intrinsic": False
    }
    
    if "structure" not in data:
        return result
    
    structure = data["structure"]
    
    # 检查 pred 部分
    if "pred" in structure and "tensor_info" in structure["pred"]:
        pred_tensors = structure["pred"]["tensor_info"]
        
        # 检查深度图
        if "depth" in pred_tensors:
            result["has_depth"] = True
            depth_shape = pred_tensors["depth"]["shape"]
            if len(depth_shape) >= 2:  # 至少是 (H, W) 或 (B, H, W)
                result["has_point_cloud"] = True  # 深度图可以转换为点云
        
        # 检查 3D 点
        for key in ["pts3d", "points", "pts3d_in_other_view", "world_points"]:
            if key in pred_tensors:
                result["has_3d_points"] = True
                result["has_point_cloud"] = True
                break
        
        # 检查位姿
        if "camera_pose" in pred_tensors:
            result["has_camera_pose"] = True
            result["has_pose"] = True
        
        if "extrinsic_world_to_cam" in pred_tensors:
            result["has_extrinsic"] = True
            result["has_pose"] = True
        
        if "intrinsic" in pred_tensors:
            result["has_intrinsic"] = True
    
    # 检查 view 部分
    if "view" in structure and "tensor_info" in structure["view"]:
        view_tensors = structure["view"]["tensor_info"]
        
        if "camera_pose" in view_tensors:
            result["has_camera_pose"] = True
            result["has_pose"] = True
        
        if "world_to_cam" in view_tensors:
            result["has_extrinsic"] = True
            result["has_pose"] = True
        
        if "intrinsic" in view_tensors:
            result["has_intrinsic"] = True
    
    return result


def print_analysis(info: Dict[str, Any], check_result: Dict[str, bool]):
    """
    打印分析结果
    
    Args:
        info: 文件分析信息
        check_result: 点云和位姿检查结果
    """
    print("=" * 80)
    print(f"文件: {info['file_path']}")
    print(f"文件大小: {info.get('file_size_mb', 0):.2f} MB")
    
    if "error" in info:
        print(f"错误: {info['error']}")
        return
    
    print(f"\n顶层键: {info.get('keys', [])}")
    
    # 打印结构信息
    structure = info.get("structure", {})
    
    if "pred" in structure:
        print("\n【pred 部分】")
        print(f"  键: {structure['pred'].get('keys', [])}")
        if "tensor_info" in structure["pred"]:
            print("  张量信息:")
            for key, tensor_info in structure["pred"]["tensor_info"].items():
                if "shape" in tensor_info:
                    print(f"    - {key}: shape={tensor_info['shape']}, dtype={tensor_info['dtype']}")
                    if tensor_info.get('min') is not None:
                        print(f"      范围: [{tensor_info['min']:.4f}, {tensor_info['max']:.4f}]")
                else:
                    print(f"    - {key}: {tensor_info.get('type', 'unknown')}")
    
    if "view" in structure:
        print("\n【view 部分】")
        print(f"  键: {structure['view'].get('keys', [])}")
        if "tensor_info" in structure["view"]:
            print("  张量信息:")
            for key, tensor_info in structure["view"]["tensor_info"].items():
                if "shape" in tensor_info:
                    print(f"    - {key}: shape={tensor_info['shape']}, dtype={tensor_info['dtype']}")
                    if tensor_info.get('min') is not None:
                        print(f"      范围: [{tensor_info['min']:.4f}, {tensor_info['max']:.4f}]")
                else:
                    print(f"    - {key}: {tensor_info.get('type', 'unknown')}")
    
    if "meta" in structure:
        print("\n【meta 部分】")
        print(f"  键: {structure['meta'].get('keys', [])}")
        if "content" in structure["meta"]:
            for key, value in structure["meta"]["content"].items():
                print(f"    - {key}: {value}")
    
    # 打印检查结果
    print("\n【数据内容检查】")
    print(f"  包含点云数据: {'是' if check_result['has_point_cloud'] else '否'}")
    print(f"  包含深度图: {'是' if check_result['has_depth'] else '否'}")
    print(f"  包含3D点: {'是' if check_result['has_3d_points'] else '否'}")
    print(f"  包含位姿: {'是' if check_result['has_pose'] else '否'}")
    print(f"  包含相机位姿: {'是' if check_result['has_camera_pose'] else '否'}")
    print(f"  包含外参矩阵: {'是' if check_result['has_extrinsic'] else '否'}")
    print(f"  包含内参矩阵: {'是' if check_result['has_intrinsic'] else '否'}")


def main():
    parser = argparse.ArgumentParser(description="分析 .pt 文件内容")
    parser.add_argument(
        "--dir",
        type=str,
        default="output/HKU_Landmark",
        help="包含 .pt 文件的目录路径"
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=5,
        help="要分析的文件数量（默认分析前5个）"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="分析所有文件（可能很慢）"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="分析指定的单个文件"
    )
    
    args = parser.parse_args()
    
    dir_path = Path(args.dir)
    if not dir_path.exists():
        print(f"错误: 目录不存在: {dir_path}")
        return
    
    # 获取所有 .pt 文件
    pt_files = sorted(dir_path.glob("*.pt"))
    
    if not pt_files:
        print(f"错误: 在 {dir_path} 中未找到 .pt 文件")
        return
    
    print(f"找到 {len(pt_files)} 个 .pt 文件")
    
    # 确定要分析的文件
    if args.file:
        files_to_analyze = [dir_path / args.file]
        if not files_to_analyze[0].exists():
            print(f"错误: 文件不存在: {files_to_analyze[0]}")
            return
    elif args.all:
        files_to_analyze = pt_files
    else:
        files_to_analyze = pt_files[:args.num_files]
    
    print(f"将分析 {len(files_to_analyze)} 个文件\n")
    
    # 分析每个文件
    all_check_results = []
    for i, pt_file in enumerate(files_to_analyze, 1):
        print(f"\n[{i}/{len(files_to_analyze)}]")
        info = analyze_pt_file(str(pt_file))
        check_result = check_point_cloud_data(info)
        print_analysis(info, check_result)
        all_check_results.append(check_result)
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("【汇总统计】")
    total = len(all_check_results)
    print(f"分析的文件数: {total}")
    print(f"包含点云数据: {sum(1 for r in all_check_results if r['has_point_cloud'])}/{total}")
    print(f"包含深度图: {sum(1 for r in all_check_results if r['has_depth'])}/{total}")
    print(f"包含3D点: {sum(1 for r in all_check_results if r['has_3d_points'])}/{total}")
    print(f"包含位姿: {sum(1 for r in all_check_results if r['has_pose'])}/{total}")
    print(f"包含相机位姿: {sum(1 for r in all_check_results if r['has_camera_pose'])}/{total}")
    print(f"包含外参矩阵: {sum(1 for r in all_check_results if r['has_extrinsic'])}/{total}")
    print(f"包含内参矩阵: {sum(1 for r in all_check_results if r['has_intrinsic'])}/{total}")


if __name__ == "__main__":
    main()

