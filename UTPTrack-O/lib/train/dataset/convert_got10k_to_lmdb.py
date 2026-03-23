#!/usr/bin/env python3
"""
GOT-10K数据集转换为LMDB格式的完整脚本
使用方法：
python convert_got10k_to_lmdb.py --input_dir /path/to/got10k --output_dir /path/to/got10k_lmdb
"""

import os
import argparse
import lmdb
from tqdm import tqdm
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading


class GOT10KToLMDBConverter:
    def __init__(self, input_dir, output_dir, map_size_gb=200):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.map_size = map_size_gb * 1024 * 1024 * 1024  # 转换为字节
        self.lock = threading.Lock()

    def get_sequence_list(self, split='train'):
        """获取序列列表"""
        list_file = os.path.join(self.input_dir, split, 'list.txt')
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"序列列表文件不存在: {list_file}")

        with open(list_file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        return sequences

    def convert_sequence(self, seq_name, split, txn):
        """转换单个序列"""
        seq_dir = os.path.join(self.input_dir, split, seq_name)
        if not os.path.exists(seq_dir):
            print(f"警告: 序列目录不存在 {seq_dir}")
            return 0

        converted_files = 0

        try:
            # 1. 转换图像文件
            img_files = [f for f in os.listdir(seq_dir) if f.endswith('.jpg')]
            img_files.sort()  # 确保顺序

            for img_file in img_files:
                img_path = os.path.join(seq_dir, img_file)
                if os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        img_data = f.read()

                    key = f'{split}/{seq_name}/{img_file}'.encode('utf-8')
                    with self.lock:
                        txn.put(key, img_data)
                    converted_files += 1

            # 2. 转换标注和元数据文件
            text_files = ['groundtruth.txt', 'absence.label', 'cover.label', 'meta_info.ini']
            for txt_file in text_files:
                txt_path = os.path.join(seq_dir, txt_file)
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    key = f'{split}/{seq_name}/{txt_file}'.encode('utf-8')
                    with self.lock:
                        txn.put(key, content.encode('utf-8'))
                    converted_files += 1
                else:
                    print(f"警告: 文件不存在 {txt_path}")

        except Exception as e:
            print(f"转换序列 {seq_name} 时出错: {e}")

        return converted_files

    def convert_split(self, split='train', num_workers=4):
        """转换指定的数据分割"""
        print(f"开始转换 {split} 数据...")

        # 获取序列列表
        sequences = self.get_sequence_list(split)
        print(f"找到 {len(sequences)} 个序列")

        # 创建LMDB环境
        os.makedirs(self.output_dir, exist_ok=True)
        env = lmdb.open(self.output_dir, map_size=self.map_size)

        with env.begin(write=True) as txn:
            # 1. 保存序列列表文件
            list_file = os.path.join(self.input_dir, split, 'list.txt')
            with open(list_file, 'r') as f:
                list_content = f.read()

            list_key = f'{split}/list.txt'.encode('utf-8')
            txn.put(list_key, list_content.encode('utf-8'))
            print(f"已保存序列列表: {list_key.decode()}")

            # 2. 转换所有序列
            total_files = 0

            if num_workers > 1:
                print(f"使用 {num_workers} 个线程并行转换...")
                # 多线程转换
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # 提交所有任务
                    futures = []
                    for seq_name in sequences:
                        future = executor.submit(self.convert_sequence, seq_name, split, txn)
                        futures.append(future)

                    # 收集结果并显示进度
                    for future in tqdm(futures, desc=f"转换{split}数据"):
                        files_count = future.result()
                        total_files += files_count
            else:
                print("使用单线程转换...")
                # 单线程转换
                for seq_name in tqdm(sequences, desc=f"转换{split}数据"):
                    files_count = self.convert_sequence(seq_name, split, txn)
                    total_files += files_count

            print(f"{split} 数据转换完成，共转换 {total_files} 个文件")

        env.close()
        return total_files

    def verify_conversion(self, split='train', sample_size=5):
        """验证转换结果"""
        print(f"验证 {split} 数据转换结果...")

        env = lmdb.open(self.output_dir, readonly=True)

        with env.begin() as txn:
            # 验证序列列表
            list_key = f'{split}/list.txt'.encode('utf-8')
            list_data = txn.get(list_key)
            if list_data is None:
                print("错误: 序列列表未找到")
                return False

            sequences = list_data.decode('utf-8').strip().split('\n')
            print(f"序列列表包含 {len(sequences)} 个序列")

            # 随机验证几个序列
            import random
            sample_sequences = random.sample(sequences, min(sample_size, len(sequences)))

            for seq_name in sample_sequences:
                # 检查第一帧图像
                img_key = f'{split}/{seq_name}/00000001.jpg'.encode('utf-8')
                img_data = txn.get(img_key)
                if img_data is None:
                    print(f"警告: 序列 {seq_name} 的第一帧未找到")
                    continue

                # 尝试解码图像
                try:
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"警告: 序列 {seq_name} 的图像解码失败")
                    else:
                        print(f"✓ 序列 {seq_name}: 图像尺寸 {img.shape}")
                except Exception as e:
                    print(f"错误: 序列 {seq_name} 图像处理失败: {e}")

                # 检查标注文件
                gt_key = f'{split}/{seq_name}/groundtruth.txt'.encode('utf-8')
                gt_data = txn.get(gt_key)
                if gt_data is None:
                    print(f"警告: 序列 {seq_name} 的标注文件未找到")
                else:
                    gt_lines = gt_data.decode('utf-8').strip().split('\n')
                    print(f"✓ 序列 {seq_name}: 标注包含 {len(gt_lines)} 帧")

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
            print(f"LMDB统计信息:")
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
    parser = argparse.ArgumentParser(description='将GOT-10K数据集转换为LMDB格式')
    parser.add_argument('--input_dir', type=str, required=True, help='GOT-10K原始数据集路径')
    parser.add_argument('--output_dir', type=str, required=True, help='LMDB输出路径')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='要转换的数据分割 (默认: train)')
    parser.add_argument('--map_size_gb', type=int, default=200, help='LMDB映射大小(GB)')
    parser.add_argument('--workers', type=int, default=4, help='并行转换的线程数 (默认: 4)')
    parser.add_argument('--verify', action='store_true', help='转换后验证结果')
    parser.add_argument('--info', action='store_true', help='显示现有LMDB信息')

    args = parser.parse_args()

    # 检查输入路径
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 {args.input_dir}")
        return

    if not os.path.exists(os.path.join(args.input_dir, args.split)):
        print(f"错误: {args.split} 目录不存在于 {args.input_dir}")
        return

    # 创建转换器
    converter = GOT10KToLMDBConverter(args.input_dir, args.output_dir, args.map_size_gb)

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