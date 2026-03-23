#!/usr/bin/env python3
"""
修复的TrackingNet数据集转换为LMDB格式的脚本
主要修复：为每个TRAIN_X创建独立的LMDB数据库
"""

import os
import argparse
import lmdb
from tqdm import tqdm
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import glob


class TrackingNetToLMDBConverter:
    def __init__(self, input_dir, output_dir, map_size_gb=50):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.map_size = map_size_gb * 1024 * 1024 * 1024  # 每个分割的大小
        self.lock = threading.Lock()
        
        # TrackingNet有12个训练分割 (TRAIN_0 到 TRAIN_11)
        self.train_splits = [f"TRAIN_{i}" for i in range(12)]
    
    def get_trackingnet_splits(self):
        """获取TrackingNet可用的分割"""
        available_splits = []
        for split in self.train_splits:
            split_dir = os.path.join(self.input_dir, split)
            if os.path.exists(split_dir):
                available_splits.append(split)
        
        if not available_splits:
            raise ValueError(f"在 {self.input_dir} 中没有找到TrackingNet分割目录")
        
        return available_splits
    
    def get_sequences_in_split(self, split):
        """获取指定分割中的所有序列"""
        anno_dir = os.path.join(self.input_dir, split, "anno")
        if not os.path.exists(anno_dir):
            print(f"警告: 标注目录不存在 {anno_dir}")
            return []
        
        anno_files = glob.glob(os.path.join(anno_dir, "*.txt"))
        sequences = [os.path.splitext(os.path.basename(f))[0] for f in anno_files]
        
        return sorted(sequences)
    
    def convert_single_split(self, split, sequences, num_workers=4):
        """转换单个分割到独立的LMDB数据库"""
        split_id = int(split.split('_')[1])
        split_lmdb_dir = os.path.join(self.output_dir, f"{split}_lmdb")
        
        print(f"转换 {split} 到 {split_lmdb_dir} ({len(sequences)} 个序列)...")
        
        # 创建分割专用的LMDB数据库
        os.makedirs(split_lmdb_dir, exist_ok=True)
        env = lmdb.open(split_lmdb_dir, map_size=self.map_size)
        
        total_files = 0
        
        with env.begin(write=True) as txn:
            if num_workers > 1:
                # 多线程转换
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_seq = {
                        executor.submit(self.convert_sequence_to_split, split, seq_name, txn): seq_name
                        for seq_name in sequences
                    }
                    
                    for future in tqdm(as_completed(future_to_seq), 
                                     desc=f"转换{split}", 
                                     total=len(sequences)):
                        files_count = future.result()
                        total_files += files_count
            else:
                # 单线程转换
                for seq_name in tqdm(sequences, desc=f"转换{split}"):
                    files_count = self.convert_sequence_to_split(split, seq_name, txn)
                    total_files += files_count
        
        env.close()
        print(f"{split} 转换完成，处理了 {total_files} 个文件")
        return total_files, sequences
    
    def convert_sequence_to_split(self, split, seq_name, txn):
        """转换单个序列到分割的LMDB数据库"""
        frames_dir = os.path.join(self.input_dir, split, "frames", seq_name)
        anno_file = os.path.join(self.input_dir, split, "anno", f"{seq_name}.txt")
        
        if not os.path.exists(frames_dir):
            print(f"警告: 序列帧目录不存在 {frames_dir}")
            return 0
        
        if not os.path.exists(anno_file):
            print(f"警告: 序列标注文件不存在 {anno_file}")
            return 0
        
        converted_files = 0
        
        try:
            # 1. 转换图像帧 - 键名格式: frames/seq_name/frame.jpg
            frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')]
            frame_files.sort()
            
            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                if os.path.exists(frame_path):
                    with open(frame_path, 'rb') as f:
                        frame_data = f.read()
                    
                    # 关键修改：键名不包含TRAIN_X前缀
                    key = f'frames/{seq_name}/{frame_file}'.encode('utf-8')
                    with self.lock:
                        txn.put(key, frame_data)
                    converted_files += 1
            
            # 2. 转换标注文件 - 键名格式: anno/seq_name.txt
            with open(anno_file, 'r', encoding='utf-8', errors='ignore') as f:
                anno_content = f.read()
            
            anno_key = f'anno/{seq_name}.txt'.encode('utf-8')
            with self.lock:
                txn.put(anno_key, anno_content.encode('utf-8'))
            converted_files += 1
                    
        except Exception as e:
            print(f"转换序列 {split}/{seq_name} 时出错: {e}")
            
        return converted_files
    
    def convert_all_splits(self, num_workers=4, specific_splits=None):
        """转换所有TrackingNet分割"""
        print("开始转换 TrackingNet 数据...")
        
        # 获取可用的分割
        available_splits = self.get_trackingnet_splits()
        
        if specific_splits:
            splits_to_convert = [s for s in specific_splits if s in available_splits]
            if not splits_to_convert:
                raise ValueError(f"指定的分割都不存在: {specific_splits}")
        else:
            splits_to_convert = available_splits
        
        print(f"将转换以下分割: {splits_to_convert}")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 转换所有分割并收集序列信息
        all_sequence_info = []
        grand_total_files = 0
        
        for split in splits_to_convert:
            sequences = self.get_sequences_in_split(split)
            if sequences:
                files_count, converted_sequences = self.convert_single_split(split, sequences, num_workers)
                grand_total_files += files_count
                
                # 添加到总序列列表 - 格式: [split_id, seq_name]
                split_id = int(split.split('_')[1])
                for seq_name in converted_sequences:
                    all_sequence_info.append([split_id, seq_name])
        
        # 保存序列列表到主目录
        seq_list_file = os.path.join(self.output_dir, "seq_list.json")
        with open(seq_list_file, 'w') as f:
            json.dump(all_sequence_info, f, separators=(',', ':'))
        
        print(f"已保存序列列表: {seq_list_file} ({len(all_sequence_info)} 个序列)")
        print(f"TrackingNet 数据转换完成，总共转换 {grand_total_files} 个文件")
        
        return grand_total_files
    
    def verify_conversion(self, sample_size=5):
        """验证转换结果"""
        print("验证 TrackingNet 数据转换结果...")
        
        # 1. 检查seq_list.json
        seq_list_file = os.path.join(self.output_dir, "seq_list.json")
        if not os.path.exists(seq_list_file):
            print("错误: seq_list.json 不存在")
            return False
        
        with open(seq_list_file, 'r') as f:
            sequence_info = json.load(f)
        
        print(f"序列列表包含 {len(sequence_info)} 个序列")
        
        # 按分割统计
        split_counts = {}
        for split_id, seq_name in sequence_info:
            split_name = f"TRAIN_{split_id}"
            split_counts[split_name] = split_counts.get(split_name, 0) + 1
        
        for split_name, count in split_counts.items():
            print(f"  {split_name}: {count} 个序列")
        
        # 2. 验证LMDB数据库
        import random
        sample_sequences = random.sample(sequence_info, min(sample_size, len(sequence_info)))
        
        for split_id, seq_name in sample_sequences:
            split_name = f"TRAIN_{split_id}"
            split_lmdb_dir = os.path.join(self.output_dir, f"{split_name}_lmdb")
            
            if not os.path.exists(split_lmdb_dir):
                print(f"错误: LMDB数据库不存在 {split_lmdb_dir}")
                continue
            
            env = lmdb.open(split_lmdb_dir, readonly=True)
            with env.begin() as txn:
                # 检查第一帧
                frame_key = f'frames/{seq_name}/0.jpg'.encode('utf-8')
                frame_data = txn.get(frame_key)
                if frame_data is None:
                    frame_key = f'frames/{seq_name}/1.jpg'.encode('utf-8')
                    frame_data = txn.get(frame_key)
                
                if frame_data is None:
                    print(f"警告: 序列 {split_name}/{seq_name} 的首帧未找到")
                else:
                    try:
                        nparr = np.frombuffer(frame_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            print(f"✓ 序列 {split_name}/{seq_name}: 图像尺寸 {img.shape}")
                    except Exception as e:
                        print(f"错误: 序列 {split_name}/{seq_name} 图像处理失败: {e}")
                
                # 检查标注文件
                anno_key = f'anno/{seq_name}.txt'.encode('utf-8')
                anno_data = txn.get(anno_key)
                if anno_data is None:
                    print(f"警告: 序列 {split_name}/{seq_name} 的标注文件未找到")
                else:
                    anno_lines = anno_data.decode('utf-8').strip().split('\n')
                    print(f"✓ 序列 {split_name}/{seq_name}: 标注包含 {len(anno_lines)} 帧")
            
            env.close()
        
        print("验证完成")
        return True


def main():
    parser = argparse.ArgumentParser(description='将TrackingNet数据集转换为LMDB格式（修复版本）')
    parser.add_argument('--input_dir', type=str, required=True, help='TrackingNet原始数据集路径')
    parser.add_argument('--output_dir', type=str, required=True, help='LMDB输出路径')
    parser.add_argument('--splits', type=str, nargs='*', help='要转换的特定分割')
    parser.add_argument('--map_size_gb', type=int, default=50, help='每个LMDB数据库的大小(GB)')
    parser.add_argument('--workers', type=int, default=6, help='并行转换的线程数')
    parser.add_argument('--verify', action='store_true', help='转换后验证结果')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 {args.input_dir}")
        return
    
    converter = TrackingNetToLMDBConverter(args.input_dir, args.output_dir, args.map_size_gb)
    
    try:
        available_splits = converter.get_trackingnet_splits()
        print(f"发现可用分割: {available_splits}")
        
        total_files = converter.convert_all_splits(args.workers, args.splits)
        print(f"转换完成！共处理 {total_files} 个文件")
        
        if args.verify:
            converter.verify_conversion()
            
    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()