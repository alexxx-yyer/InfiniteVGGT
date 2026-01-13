#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据原始图片的顺序重新组织输出文件

这个脚本用于修复因为文件排序问题导致的输出文件顺序混乱。
它会读取原始图片目录的顺序，然后按照这个顺序重新组织输出目录的文件。

Usage:
    python scripts/reorder_output_by_source.py \
        --source_dir /path/to/original/images \
        --output_dir /path/to/output \
        --output_subdirs color depth conf camera \
        --target_dir /path/to/reordered/output
"""

import os
import sys
import argparse
import shutil
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from multiprocessing import Pool, cpu_count
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def _copy_file_task(args: Tuple[Path, Path, int]) -> Optional[str]:
    """
    复制单个文件的任务函数（用于多进程）
    
    Args:
        args: (source_file, target_file, index) 元组
    
    Returns:
        如果出错返回错误信息，否则返回None
    """
    output_file, target_file, correct_idx = args
    try:
        shutil.copy2(output_file, target_file)
        return None
    except Exception as e:
        return f"复制 {output_file.name} 到 {target_file.name} 时出错: {e}"


def natural_sort_key(filename: str) -> tuple:
    """
    自然排序键函数，能够正确处理文件名中的数字。
    
    例如：
    - img0.png -> (0,)
    - img10.png -> (10,)
    - img100.png -> (100,)
    
    这样排序后：img0.png < img10.png < img100.png < img1000.png
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # 使用正则表达式分割字符串和数字
    parts = re.split(r'(\d+)', name_without_ext)
    
    # 将数字部分转换为整数，非数字部分保持字符串
    key_parts = []
    for part in parts:
        if part.isdigit():
            key_parts.append(int(part))
        else:
            key_parts.append(part.lower())
    
    return tuple(key_parts)


def get_sorted_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    获取目录中的文件，按自然顺序排序
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，如 ['.png', '.jpg']，如果为None则获取所有文件
    
    Returns:
        排序后的文件路径列表
    """
    if not directory.exists():
        return []
    
    if extensions is None:
        files = [f for f in directory.iterdir() if f.is_file()]
    else:
        files = [f for f in directory.iterdir() 
                if f.is_file() and f.suffix.lower() in [e.lower() for e in extensions]]
    
    # 使用自然排序
    files = sorted(files, key=lambda x: natural_sort_key(str(x)))
    return files


def extract_index_from_filename(filename: str, pattern: str = None) -> Optional[int]:
    """
    从文件名中提取索引数字
    
    Args:
        filename: 文件名（不含路径）
        pattern: 可选的正则表达式模式，如果提供则使用该模式提取
    
    Returns:
        提取的索引，如果无法提取则返回None
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    if pattern:
        match = re.search(pattern, name_without_ext)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
    
    # 尝试提取所有数字，取最后一个（通常是索引）
    numbers = re.findall(r'\d+', name_without_ext)
    if numbers:
        try:
            return int(numbers[-1])  # 取最后一个数字
        except ValueError:
            pass
    
    return None


def reorder_output_files(
    source_dir: Path,
    output_dir: Path,
    target_dir: Path,
    subdirs: List[str],
    source_extensions: List[str] = None,
    num_workers: int = None
):
    """
    根据原始图片的顺序重新组织输出文件
    
    Args:
        source_dir: 原始图片目录
        output_dir: 输出文件目录（包含子目录如 color/, depth/ 等）
        target_dir: 目标目录（重新组织后的输出）
        subdirs: 需要重新组织的子目录列表，如 ['color', 'depth', 'conf', 'camera']
        source_extensions: 原始图片的扩展名列表，如 ['.png', '.jpg']
    """
    if source_extensions is None:
        source_extensions = ['.png', '.jpg', '.jpeg']
    
    print(f"读取原始图片目录: {source_dir}")
    
    # 第一步：获取原始图片文件（使用字符串排序，模拟处理时的混乱顺序）
    source_files_string_sorted = []
    if source_extensions is None:
        source_extensions = ['.png', '.jpg', '.jpeg']
    
    for ext in source_extensions:
        source_files_string_sorted.extend(list(source_dir.glob(f"*{ext}")))
        source_files_string_sorted.extend(list(source_dir.glob(f"*{ext.upper()}")))
    
    # 使用字符串排序（sorted），模拟处理时的混乱顺序
    source_files_string_sorted = sorted(source_files_string_sorted, key=lambda x: x.name)
    
    if not source_files_string_sorted:
        print(f"错误: 在 {source_dir} 中未找到图片文件")
        return
    
    print(f"找到 {len(source_files_string_sorted)} 个原始图片文件")
    print(f"字符串排序后的前5个文件（处理顺序）: {[f.name for f in source_files_string_sorted[:5]]}")
    print(f"字符串排序后的后5个文件: {[f.name for f in source_files_string_sorted[-5:]]}")
    
    # 第二步：获取原始图片的正确顺序（使用自然排序）
    source_files_natural_sorted = get_sorted_files(source_dir, source_extensions)
    print(f"自然排序后的前5个文件（正确顺序）: {[f.name for f in source_files_natural_sorted[:5]]}")
    print(f"自然排序后的后5个文件: {[f.name for f in source_files_natural_sorted[-5:]]}")
    
    # 第三步：建立映射关系
    # 映射1：字符串排序索引 -> 原始文件名
    string_sorted_to_filename = {i: f.name for i, f in enumerate(source_files_string_sorted)}
    
    # 映射2：原始文件名 -> 自然排序索引（正确顺序）
    filename_to_natural_index = {f.name: i for i, f in enumerate(source_files_natural_sorted)}
    
    # 映射3：字符串排序索引（处理顺序/输出文件索引） -> 自然排序索引（正确顺序）
    # 这个映射告诉我们：输出文件索引 i 应该对应原始图片的正确顺序索引是多少
    output_index_to_correct_index = {}
    for string_idx, filename in string_sorted_to_filename.items():
        if filename in filename_to_natural_index:
            output_index_to_correct_index[string_idx] = filename_to_natural_index[filename]
    
    print(f"\n建立映射关系完成")
    print(f"示例映射（前5个）:")
    for i in range(min(5, len(output_index_to_correct_index))):
        output_idx = i
        correct_idx = output_index_to_correct_index.get(i, -1)
        if correct_idx >= 0:
            print(f"  输出索引 {output_idx:06d} (对应处理时的 {source_files_string_sorted[i].name}) -> 正确索引 {correct_idx:06d} (对应 {source_files_natural_sorted[correct_idx].name})")
    
    # 建立反向映射：correct_idx -> output_idx（用于快速查找）
    correct_index_to_output_index = {cor_idx: out_idx for out_idx, cor_idx in output_index_to_correct_index.items()}
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个子目录
    for subdir in subdirs:
        output_subdir = output_dir / subdir
        target_subdir = target_dir / subdir
        
        if not output_subdir.exists():
            print(f"警告: 输出子目录 {output_subdir} 不存在，跳过")
            continue
        
        print(f"\n处理子目录: {subdir}")
        
        # 对于 pointclouds 目录，需要特殊处理：
        # 1. frame_*.pcd 文件需要重新组织
        # 2. merged_pointcloud.pcd 直接复制到目标目录根目录，不在子目录中
        if subdir == "pointclouds":
            # 处理 merged_pointcloud.pcd
            merged_pcd = output_subdir / "merged_pointcloud.pcd"
            if merged_pcd.exists():
                target_merged_pcd = target_dir / "merged_pointcloud.pcd"
                print(f"复制合并点云文件: {merged_pcd.name} -> {target_merged_pcd}")
                try:
                    shutil.copy2(merged_pcd, target_merged_pcd)
                    print(f"完成: {merged_pcd.name} 已复制到目标目录根目录")
                except Exception as e:
                    print(f"警告: 复制 {merged_pcd.name} 时出错: {e}")
            
            # 只处理 frame_*.pcd 文件
            output_files = [f for f in output_subdir.glob("frame_*.pcd")]
            output_files = sorted(output_files, key=lambda x: natural_sort_key(str(x)))
        else:
            # 获取输出文件
            output_files = get_sorted_files(output_subdir)
        
        if not output_files:
            print(f"警告: 在 {output_subdir} 中未找到需要处理的文件，跳过")
            continue
        
        if len(output_files) != len(source_files_string_sorted):
            print(f"警告: 输出文件数量 ({len(output_files)}) 与原始图片数量 ({len(source_files_string_sorted)}) 不匹配")
            print(f"      将使用较小的数量: {min(len(output_files), len(source_files_string_sorted))}")
        
        # 创建目标子目录
        target_subdir.mkdir(parents=True, exist_ok=True)
        
        # 确定文件扩展名（从输出文件中获取）
        output_ext = output_files[0].suffix if output_files else '.png'
        
        # 建立映射关系
        # 输出文件是按照处理顺序命名的（000000, 000001, ...）
        # 处理顺序是字符串排序的混乱顺序
        # 我们需要根据映射关系，将输出文件按照原始图片的正确顺序重新组织
        
        num_files = min(len(source_files_string_sorted), len(output_files))
        
        if len(output_files) != len(source_files_string_sorted):
            print(f"警告: 输出文件数量 ({len(output_files)}) 与原始图片数量 ({len(source_files_string_sorted)}) 不匹配")
            print(f"      将使用较小的数量: {num_files}")
        
        print(f"复制 {num_files} 个文件...")
        print(f"输出文件顺序（按处理顺序）: {[f.name for f in output_files[:3]]} ... {[f.name for f in output_files[-3:]]}")
        
        # 准备所有复制任务
        copy_tasks = []
        for correct_idx in range(num_files):
            # 使用反向映射快速查找
            output_idx = correct_index_to_output_index.get(correct_idx)
            
            if output_idx is None or output_idx >= len(output_files):
                continue
            
            # 获取对应的输出文件
            output_file = output_files[output_idx]
            
            # 生成目标文件名：使用正确顺序的索引
            # 对于 pointclouds 目录，保留 frame_ 前缀
            if subdir == "pointclouds":
                target_filename = f"frame_{correct_idx:06d}{output_ext}"
            else:
                target_filename = f"{correct_idx:06d}{output_ext}"
            target_file = target_subdir / target_filename
            
            copy_tasks.append((output_file, target_file, correct_idx))
        
        # 并行复制文件
        if num_workers is None:
            num_workers = max(1, min(cpu_count() - 1, len(copy_tasks)))  # 使用 CPU核心数 - 1，保留一个核心给系统
        else:
            num_workers = max(1, min(num_workers, len(copy_tasks)))
        
        if len(copy_tasks) > 1 and num_workers > 1:
            print(f"使用 {num_workers} 个进程并行复制文件...")
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(_copy_file_task, copy_tasks),
                    total=len(copy_tasks),
                    desc=f"复制 {subdir}",
                    unit="文件"
                ))
        else:
            # 单进程复制
            results = list(tqdm(
                map(_copy_file_task, copy_tasks),
                total=len(copy_tasks),
                desc=f"复制 {subdir}",
                unit="文件"
            ))
        
        # 检查错误
        errors = [r for r in results if r is not None]
        if errors:
            print(f"警告: {len(errors)} 个文件复制失败")
        
        print(f"完成: {subdir} -> {target_subdir}")
    
    print(f"\n重新组织完成！")
    print(f"目标目录: {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="根据原始图片的顺序重新组织输出文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="原始图片目录路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出文件目录路径（包含子目录如 color/, depth/ 等）"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="目标目录路径（重新组织后的输出）"
    )
    parser.add_argument(
        "--output_subdirs",
        type=str,
        nargs="+",
        default=["color", "depth", "conf", "camera", "depth_vis", "pointclouds"],
        help="需要重新组织的子目录列表"
    )
    parser.add_argument(
        "--source_extensions",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="原始图片的文件扩展名列表"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="并行处理的进程数，默认使用所有CPU核心"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    target_dir = Path(args.target_dir)
    
    if not source_dir.exists():
        print(f"错误: 原始图片目录不存在: {source_dir}")
        sys.exit(1)
    
    if not output_dir.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        sys.exit(1)
    
    reorder_output_files(
        source_dir=source_dir,
        output_dir=output_dir,
        target_dir=target_dir,
        subdirs=args.output_subdirs,
        source_extensions=args.source_extensions,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()

