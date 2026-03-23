#!/usr/bin/env python3
"""
COCO数据集转换为LMDB格式的完整脚本
使用方法：
python convert_coco_to_lmdb.py --input_dir /path/to/coco --output_dir /path/to/coco_lmdb --split train2017
"""

import os
import argparse
import lmdb
from tqdm import tqdm
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import shutil


class COCOToLMDBConverter:
    def __init__(self, input_dir, output_dir, map_size_gb=200):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.map_size = map_size_gb * 1024 * 1024 * 1024  # 转换为字节
        self.lock = threading.Lock()
        
    def get_coco_info(self, split='train2017'):
        """获取COCO数据集信息"""
        # 确定标注文件路径
        if split == 'train2017':
            ann_file = os.path.join(self.input_dir, 'annotations', 'instances_train2017.json')
        elif split == 'val2017':
            ann_file = os.path.join(self.input_dir, 'annotations', 'instances_val2017.json')
        elif split == 'test2017':
            ann_file = os.path.join(self.input_dir, 'annotations', 'image_info_test2017.json')
        else:
            raise ValueError(f"不支持的分割: {split}")
        
        # 图像目录
        img_dir = os.path.join(self.input_dir, 'images', split)
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"标注文件不存在: {ann_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
            
        return ann_file, img_dir
    
    def convert_images_batch(self, image_files, img_dir, split, txn, start_idx=0):
        """批量转换图像"""
        converted_files = 0
        
        for i, img_file in enumerate(image_files):
            try:
                img_path = os.path.join(img_dir, img_file)
                if os.path.exists(img_path):
                    # 读取图像数据
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                    
                    # 构造LMDB键名
                    key = f'images/{split}/{img_file}'.encode('utf-8')
                    
                    # 线程安全的写入LMDB
                    with self.lock:
                        txn.put(key, img_data)
                    converted_files += 1
                else:
                    print(f"警告: 图像文件不存在 {img_path}")
                    
            except Exception as e:
                print(f"转换图像 {img_file} 时出错: {e}")
        
        return converted_files
    
    def convert_split(self, split='train2017', num_workers=4):
        """转换指定的COCO数据分割"""
        print(f"开始转换 COCO {split} 数据...")
        
        # 获取COCO数据信息
        ann_file, img_dir = self.get_coco_info(split)
        
        # 读取标注文件
        print("读取COCO标注文件...")
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        print(f"数据集信息:")
        print(f"  图像数量: {len(coco_data.get('images', []))}")
        print(f"  类别数量: {len(coco_data.get('categories', []))}")
        print(f"  标注数量: {len(coco_data.get('annotations', []))}")
        
        # 获取所有图像文件
        if 'images' in coco_data:
            image_files = [img_info['file_name'] for img_info in coco_data['images']]
        else:
            # 如果标注文件中没有图像信息，扫描目录
            image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 创建LMDB环境
        os.makedirs(self.output_dir, exist_ok=True)
        env = lmdb.open(self.output_dir, map_size=self.map_size)
        
        with env.begin(write=True) as txn:
            # 1. 保存标注文件
            ann_key = f'annotations/instances_{split}.json'.encode('utf-8')
            ann_content = json.dumps(coco_data, separators=(',', ':')).encode('utf-8')
            txn.put(ann_key, ann_content)
            print(f"已保存标注文件: annotations/instances_{split}.json")
            
            # 2. 转换图像文件
            total_files = 0
            
            if num_workers > 1:
                print(f"使用 {num_workers} 个线程并行转换图像...")
                
                # 将图像文件分批
                batch_size = max(1, len(image_files) // (num_workers * 4))  # 每个线程多个批次
                batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
                
                # 多线程转换
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for i, batch in enumerate(batches):
                        future = executor.submit(
                            self.convert_images_batch, 
                            batch, img_dir, split, txn, i * batch_size
                        )
                        futures.append(future)
                    
                    # 收集结果并显示进度
                    for future in tqdm(futures, desc=f"转换{split}图像"):
                        files_count = future.result()
                        total_files += files_count
            else:
                print("使用单线程转换图像...")
                # 单线程转换
                total_files = self.convert_images_batch(
                    image_files, img_dir, split, txn
                )
            
            print(f"COCO {split} 数据转换完成，共转换 {total_files} 个文件")
        
        env.close()
        return total_files
    
    def verify_conversion(self, split='train2017', sample_size=5):
        """验证COCO转换结果"""
        print(f"验证 COCO {split} 数据转换结果...")
        
        env = lmdb.open(self.output_dir, readonly=True)
        
        with env.begin() as txn:
            # 验证标注文件
            ann_key = f'annotations/instances_{split}.json'.encode('utf-8')
            ann_data = txn.get(ann_key)
            if ann_data is None:
                print("错误: COCO标注文件未找到")
                return False
            
            # 解析标注数据
            coco_data = json.loads(ann_data.decode('utf-8'))
            print(f"标注文件包含 {len(coco_data.get('images', []))} 个图像")
            
            # 随机验证几个图像
            import random
            if 'images' in coco_data and len(coco_data['images']) > 0:
                sample_images = random.sample(
                    coco_data['images'], 
                    min(sample_size, len(coco_data['images']))
                )
                
                for img_info in sample_images:
                    img_file = img_info['file_name']
                    img_key = f'images/{split}/{img_file}'.encode('utf-8')
                    img_data = txn.get(img_key)
                    
                    if img_data is None:
                        print(f"警告: 图像 {img_file} 未找到")
                        continue
                    
                    # 尝试解码图像
                    try:
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is None:
                            print(f"警告: 图像 {img_file} 解码失败")
                        else:
                            expected_shape = (img_info['height'], img_info['width'])
                            actual_shape = img.shape[:2]
                            if expected_shape != actual_shape:
                                print(f"警告: 图像 {img_file} 尺寸不匹配: "
                                      f"期望{expected_shape}, 实际{actual_shape}")
                            else:
                                print(f"✓ 图像 {img_file}: 尺寸 {actual_shape}, ID {img_info['id']}")
                    except Exception as e:
                        print(f"错误: 图像 {img_file} 处理失败: {e}")
            
            # 验证类别信息
            if 'categories' in coco_data:
                categories = coco_data['categories']
                print(f"✓ 数据集包含 {len(categories)} 个类别")
                sample_categories = random.sample(categories, min(3, len(categories)))
                for cat in sample_categories:
                    print(f"  - {cat['name']} (ID: {cat['id']})")
        
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
            print(f"COCO LMDB统计信息:")
            print(f"  页面大小: {stat['psize']} bytes")
            print(f"  树深度: {stat['depth']}")
            print(f"  分支页面: {stat['branch_pages']}")
            print(f"  叶子页面: {stat['leaf_pages']}")
            print(f"  条目数量: {stat['entries']}")
            
            # 获取数据库大小
            db_size = os.path.getsize(os.path.join(self.output_dir, 'data.mdb'))
            print(f"  数据库大小: {db_size / (1024**3):.2f} GB")
            
            # 尝试读取标注文件获取详细信息
            for split in ['train2017', 'val2017']:
                ann_key = f'annotations/instances_{split}.json'.encode('utf-8')
                ann_data = txn.get(ann_key)
                if ann_data is not None:
                    coco_data = json.loads(ann_data.decode('utf-8'))
                    print(f"  {split}: {len(coco_data.get('images', []))} 图像, "
                          f"{len(coco_data.get('annotations', []))} 标注")
        
        env.close()
    
    def create_symlinks(self):
        """为兼容性创建符号链接"""
        # 有些代码可能期望特定的目录结构
        links_dir = os.path.join(self.output_dir, 'links')
        os.makedirs(links_dir, exist_ok=True)
        
        # 创建指向LMDB的符号链接
        lmdb_link = os.path.join(links_dir, 'data.mdb')
        if not os.path.exists(lmdb_link):
            os.symlink(os.path.join(self.output_dir, 'data.mdb'), lmdb_link)
            print(f"创建符号链接: {lmdb_link}")


def main():
    parser = argparse.ArgumentParser(description='将COCO数据集转换为LMDB格式')
    parser.add_argument('--input_dir', type=str, required=True, help='COCO原始数据集路径 (包含images和annotations目录)')
    parser.add_argument('--output_dir', type=str, required=True, help='LMDB输出路径')
    parser.add_argument('--split', type=str, default='train2017',  choices=['train2017', 'val2017', 'test2017'], help='要转换的数据分割 (默认: train2017)')
    parser.add_argument('--map_size_gb', type=int, default=200, help='LMDB映射大小(GB) (默认: 200)')
    parser.add_argument('--workers', type=int, default=4, help='并行转换的线程数 (默认: 4)')
    parser.add_argument('--verify', action='store_true', help='转换后验证结果')
    parser.add_argument('--info', action='store_true', help='显示现有LMDB信息')
    parser.add_argument('--create_links', action='store_true', help='创建兼容性符号链接')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 {args.input_dir}")
        return
    
    # 检查COCO目录结构
    images_dir = os.path.join(args.input_dir, 'images')
    annotations_dir = os.path.join(args.input_dir, 'annotations')
    
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在 {images_dir}")
        print("COCO数据集应该包含 'images' 和 'annotations' 目录")
        return
        
    if not os.path.exists(annotations_dir):
        print(f"错误: 标注目录不存在 {annotations_dir}")
        print("COCO数据集应该包含 'images' 和 'annotations' 目录")
        return
    
    # 创建转换器
    converter = COCOToLMDBConverter(args.input_dir, args.output_dir, args.map_size_gb)
    
    if args.info:
        # 显示LMDB信息
        converter.get_lmdb_info()
        return
    
    try:
        # 执行转换
        total_files = converter.convert_split(args.split, args.workers)
        print(f"转换完成！共处理 {total_files} 个文件")
        
        # 创建符号链接
        if args.create_links:
            converter.create_symlinks()
        
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