#!/usr/bin/env python3
"""
LaSOT数据集转换为LMDB格式的完整脚本
使用方法：
python convert_lasot_to_lmdb.py --input_dir /path/to/lasot --output_dir /path/to/lasot_lmdb
"""

import os
import argparse
import lmdb
from tqdm import tqdm
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import pandas as pd


class LaSOTToLMDBConverter:
    def __init__(self, input_dir, output_dir, map_size_gb=300):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.map_size = map_size_gb * 1024 * 1024 * 1024  # 转换为字节
        self.lock = threading.Lock()

    def get_sequence_list(self, split='train'):
        """获取LaSOT序列列表"""
        if split == 'train':
            # LaSOT训练集使用特定的分割文件
            split_file = os.path.join(os.path.dirname(__file__), 'data_specs', 'lasot_train_split.txt')
            if os.path.exists(split_file):
                sequence_list = pd.read_csv(split_file, header=None).squeeze("columns").values.tolist()
                return sequence_list
            else:
                print(f"未找到分割文件 {split_file}，将扫描所有序列")

        # 如果没有分割文件，扫描所有类别和序列
        sequences = []
        class_dirs = [d for d in os.listdir(self.input_dir)
                      if os.path.isdir(os.path.join(self.input_dir, d))]

        for class_name in sorted(class_dirs):
            class_dir = os.path.join(self.input_dir, class_name)
            seq_dirs = [d for d in os.listdir(class_dir)
                        if os.path.isdir(os.path.join(class_dir, d)) and d.startswith(class_name)]

            for seq_dir in sorted(seq_dirs):
                sequences.append(f"{class_name}/{seq_dir}")

        return sequences

    def convert_sequence(self, seq_path, txn):
        """转换单个LaSOT序列"""
        full_seq_path = os.path.join(self.input_dir, seq_path)
        if not os.path.exists(full_seq_path):
            print(f"警告: 序列目录不存在 {full_seq_path}")
            return 0

        converted_files = 0

        try:
            # 1. 转换图像文件（LaSOT图像在img子目录中）
            img_dir = os.path.join(full_seq_path, 'img')
            if os.path.exists(img_dir):
                img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
                img_files.sort()  # 确保顺序

                for img_file in img_files:
                    img_path = os.path.join(img_dir, img_file)
                    if os.path.exists(img_path):
                        with open(img_path, 'rb') as f:
                            img_data = f.read()

                        key = f'{seq_path}/img/{img_file}'.encode('utf-8')
                        with self.lock:
                            txn.put(key, img_data)
                        converted_files += 1

            # 2. 转换标注和元数据文件
            text_files = ['groundtruth.txt', 'full_occlusion.txt', 'out_of_view.txt', 'nlp.txt']
            for txt_file in text_files:
                txt_path = os.path.join(full_seq_path, txt_file)
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    key = f'{seq_path}/{txt_file}'.encode('utf-8')
                    with self.lock:
                        txn.put(key, content.encode('utf-8'))
                    converted_files += 1
                else:
                    print(f"警告: 文件不存在 {txt_path}")

        except Exception as e:
            print(f"转换序列 {seq_path} 时出错: {e}")

        return converted_files

    def convert_split(self, split='train', num_workers=4):
        """转换指定的数据分割"""
        print(f"开始转换 LaSOT {split} 数据...")

        # 获取序列列表
        sequences = self.get_sequence_list(split)
        print(f"找到 {len(sequences)} 个序列")

        # 创建LMDB环境
        os.makedirs(self.output_dir, exist_ok=True)
        env = lmdb.open(self.output_dir, map_size=self.map_size)

        with env.begin(write=True) as txn:
            # 1. 保存序列列表信息（创建一个与OSTrack兼容的格式）
            if split == 'train':
                # 创建LaSOT训练分割信息
                sequence_info = {
                    "info": {
                        "dataset": "LaSOT",
                        "split": split,
                        "num_sequences": len(sequences)
                    },
                    "sequences": sequences
                }

                # 保存为JSON格式，与OSTrack期望的格式兼容
                import json
                info_content = json.dumps(sequence_info, indent=2)
                info_key = 'LaSOTBenchmark.json'.encode('utf-8')
                txn.put(info_key, info_content.encode('utf-8'))
                print(f"已保存序列信息: LaSOTBenchmark.json")

            # 2. 转换所有序列
            total_files = 0

            if num_workers > 1:
                print(f"使用 {num_workers} 个线程并行转换...")
                # 多线程转换
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # 提交所有任务
                    futures = []
                    for seq_path in sequences:
                        future = executor.submit(self.convert_sequence, seq_path, txn)
                        futures.append(future)

                    # 收集结果并显示进度
                    for future in tqdm(futures, desc=f"转换LaSOT数据"):
                        files_count = future.result()
                        total_files += files_count
            else:
                print("使用单线程转换...")
                # 单线程转换
                for seq_path in tqdm(sequences, desc=f"转换LaSOT数据"):
                    files_count = self.convert_sequence(seq_path, txn)
                    total_files += files_count

            print(f"LaSOT {split} 数据转换完成，共转换 {total_files} 个文件")

        env.close()
        return total_files

    def verify_conversion(self, split='train', sample_size=5):
        """验证LaSOT转换结果"""
        print(f"验证 LaSOT {split} 数据转换结果...")

        env = lmdb.open(self.output_dir, readonly=True)

        with env.begin() as txn:
            # 验证序列信息
            info_key = 'LaSOTBenchmark.json'.encode('utf-8')
            info_data = txn.get(info_key)
            if info_data is None:
                print("错误: LaSOT序列信息未找到")
                return False

            import json
            sequence_info = json.loads(info_data.decode('utf-8'))
            sequences = sequence_info['sequences']
            print(f"序列信息包含 {len(sequences)} 个序列")

            # 随机验证几个序列
            import random
            sample_sequences = random.sample(sequences, min(sample_size, len(sequences)))

            for seq_path in sample_sequences:
                # 检查第一帧图像
                img_key = f'{seq_path}/img/00000001.jpg'.encode('utf-8')
                img_data = txn.get(img_key)
                if img_data is None:
                    print(f"警告: 序列 {seq_path} 的第一帧未找到")
                    continue

                # 尝试解码图像
                try:
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"警告: 序列 {seq_path} 的图像解码失败")
                    else:
                        print(f"✓ 序列 {seq_path}: 图像尺寸 {img.shape}")
                except Exception as e:
                    print(f"错误: 序列 {seq_path} 图像处理失败: {e}")

                # 检查标注文件
                gt_key = f'{seq_path}/groundtruth.txt'.encode('utf-8')
                gt_data = txn.get(gt_key)
                if gt_data is None:
                    print(f"警告: 序列 {seq_path} 的标注文件未找到")
                else:
                    gt_lines = gt_data.decode('utf-8').strip().split('\n')
                    print(f"✓ 序列 {seq_path}: 标注包含 {len(gt_lines)} 帧")

        env.close()
        print("验证完成")
        return True

    def get_lmdb_info(self):
        """获取LMDB数据库信息"""
        if not os.path.exists(self.output_dir):
            print("LMDB数据库不存在")
            return

        env = lmdb.open(self.output_dir, readonly=True)
        with env.begin() as txn:
            # 获取数据库统计信息
            stat = txn.stat()
            print(f"LaSOT LMDB统计信息:")
            print(f"  页面大小: {stat['psize']} bytes")
            print(f"  树深度: {stat['depth']}")
            print(f"  分支页面: {stat['branch_pages']}")
            print(f"  叶子页面: {stat['leaf_pages']}")
            print(f"  条目数量: {stat['entries']}")

            # 获取数据库大小
            db_size = os.path.getsize(os.path.join(self.output_dir, 'data.mdb'))
            print(f"  数据库大小: {db_size / (1024 ** 3):.2f} GB")

        env.close()


def main():
    parser = argparse.ArgumentParser(description='将LaSOT数据集转换为LMDB格式')
    parser.add_argument('--input_dir', type=str, required=True, help='LaSOT原始数据集路径')
    parser.add_argument('--output_dir', type=str, required=True, help='LMDB输出路径')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='要转换的数据分割 (默认: train)')
    parser.add_argument('--map_size_gb', type=int, default=300, help='LMDB映射大小(GB)')
    parser.add_argument('--workers', type=int, default=4, help='并行转换的线程数 (默认: 4)')
    parser.add_argument('--verify', action='store_true', help='转换后验证结果')
    parser.add_argument('--info', action='store_true', help='显示现有LMDB信息')

    args = parser.parse_args()

    # 检查输入路径
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 {args.input_dir}")
        return

    # 创建转换器
    converter = LaSOTToLMDBConverter(args.input_dir, args.output_dir, args.map_size_gb)

    if args.info:
        # 显示LMDB信息
        converter.get_lmdb_info()
        return

    try:
        # 执行转换
        total_files = converter.convert_split(args.split, args.workers)
        print(f"转换完成！共处理 {total_files} 个文件")

        # 显示LMDB信息
        converter.get_lmdb_info()

        # 验证转换结果
        if args.verify:
            converter.verify_conversion(args.split)

    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()