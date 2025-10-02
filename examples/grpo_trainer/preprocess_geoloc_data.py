#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
预处理geoloc数据集，转换为训练所需的格式
"""

import argparse
import os
import json
import pandas as pd
from typing import Dict, List, Any


def create_geoloc_prompt(image_path: str = None, context: str = None) -> str:
    """
    创建geoloc任务的提示词
    """
    base_prompt = (
        "请根据提供的图像和上下文信息，预测图像拍摄的地理位置。"
        "请以经纬度坐标的形式给出答案，格式为 (纬度, 经度)。"
        "请先进行推理分析，然后给出最终的坐标预测。"
        "推理过程请用 <think> </think> 标签包围。"
        "最终答案请用坐标格式：(latitude, longitude)"
    )
    
    if context:
        base_prompt += f"\n\n上下文信息：{context}"
    
    return base_prompt


def process_geoloc_dataset(
    input_file: str,
    output_dir: str,
    data_source: str = "geoloc",
    image_column: str = "image",
    lat_column: str = "latitude", 
    lon_column: str = "longitude",
    context_column: str = None,
    split_ratio: float = 0.8
) -> None:
    """
    处理geoloc数据集
    
    Args:
        input_file: 输入的CSV或JSON文件路径
        output_dir: 输出目录
        data_source: 数据源名称
        image_column: 图像列名
        lat_column: 纬度列名
        lon_column: 经度列名
        context_column: 上下文信息列名（可选）
        split_ratio: 训练集比例
    """
    
    # 读取数据
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json') or input_file.endswith('.jsonl'):
        df = pd.read_json(input_file, lines=input_file.endswith('.jsonl'))
    else:
        raise ValueError("Unsupported file format. Please use CSV or JSON/JSONL.")
    
    print(f"Loaded {len(df)} samples from {input_file}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 检查必要的列是否存在
    required_columns = [lat_column, lon_column]
    if image_column and image_column not in df.columns:
        print(f"Warning: Image column '{image_column}' not found. Will proceed without images.")
        image_column = None
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset.")
    
    # 处理数据
    processed_data = []
    
    for idx, row in df.iterrows():
        # 获取坐标
        latitude = float(row[lat_column])
        longitude = float(row[lon_column])
        
        # 获取图像路径（如果有）
        image_path = None
        if image_column and pd.notna(row[image_column]):
            image_path = str(row[image_column])
        
        # 获取上下文信息（如果有）
        context = None
        if context_column and context_column in df.columns and pd.notna(row[context_column]):
            context = str(row[context_column])
        
        # 创建提示词
        prompt = create_geoloc_prompt(image_path, context)
        
        # 创建ground truth
        ground_truth = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "coordinates": [float(latitude), float(longitude)]
        }
        
        # 构建数据项
        data_item = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "ability": "geoloc",
            "reward_model": {
                "style": "rule", 
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": "unknown",  # 稍后会设置
                "index": idx,
                "latitude": latitude,
                "longitude": longitude,
            }
        }
        
        # 添加图像信息（如果有）
        if image_path:
            data_item["images"] = [image_path]
        
        # 添加上下文信息到extra_info（如果有）
        if context:
            data_item["extra_info"]["context"] = context
        
        processed_data.append(data_item)
    
    # 分割训练集和测试集
    total_samples = len(processed_data)
    train_size = int(total_samples * split_ratio)
    
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]
    
    # 设置split标签
    for item in train_data:
        item["extra_info"]["split"] = "train"
    for item in test_data:
        item["extra_info"]["split"] = "test"
    
    print(f"Split data: {len(train_data)} train samples, {len(test_data)} test samples")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为parquet格式
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Saved train data to: {train_path}")
    print(f"Saved test data to: {test_path}")
    
    # 保存示例数据用于检查
    sample_path = os.path.join(output_dir, "sample.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(train_data[:3], f, indent=2, ensure_ascii=False)
    
    print(f"Saved sample data to: {sample_path}")


def create_synthetic_geoloc_data(output_dir: str, num_samples: int = 1000) -> None:
    """
    创建合成的geoloc数据用于测试
    """
    import random
    
    print(f"Creating {num_samples} synthetic geoloc samples...")
    
    # 一些知名地点的坐标
    famous_locations = [
        {"name": "北京天安门", "lat": 39.9042, "lon": 116.4074},
        {"name": "上海外滩", "lat": 31.2304, "lon": 121.4737},
        {"name": "广州塔", "lat": 23.1291, "lon": 113.3240},
        {"name": "深圳平安金融中心", "lat": 22.5431, "lon": 114.0579},
        {"name": "杭州西湖", "lat": 30.2741, "lon": 120.1551},
        {"name": "成都宽窄巷子", "lat": 30.6598, "lon": 104.0633},
        {"name": "西安大雁塔", "lat": 34.2216, "lon": 108.9640},
        {"name": "南京中山陵", "lat": 32.0675, "lon": 118.8484},
    ]
    
    processed_data = []
    
    for idx in range(num_samples):
        # 随机选择一个地点或生成随机坐标
        if idx < len(famous_locations) * 10:  # 前面一部分使用知名地点
            location = random.choice(famous_locations)
            # 添加一些噪声
            latitude = location["lat"] + random.uniform(-0.01, 0.01)
            longitude = location["lon"] + random.uniform(-0.01, 0.01)
            context = f"这是{location['name']}附近的图像"
        else:
            # 生成中国境内的随机坐标
            latitude = random.uniform(18.0, 54.0)  # 中国纬度范围
            longitude = random.uniform(73.0, 135.0)  # 中国经度范围
            context = "这是一张地理位置图像"
        
        # 创建提示词
        prompt = create_geoloc_prompt(context=context)
        
        # 创建ground truth
        ground_truth = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "coordinates": [float(latitude), float(longitude)]
        }
        
        # 构建数据项
        data_item = {
            "data_source": "synthetic_geoloc",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "ability": "geoloc",
            "reward_model": {
                "style": "rule", 
                "ground_truth": ground_truth
            },
            "extra_info": {
                "split": "unknown",
                "index": idx,
                "latitude": latitude,
                "longitude": longitude,
                "context": context,
                "is_synthetic": True,
            }
        }
        
        processed_data.append(data_item)
    
    # 分割数据
    train_size = int(num_samples * 0.8)
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]
    
    # 设置split标签
    for item in train_data:
        item["extra_info"]["split"] = "train"
    for item in test_data:
        item["extra_info"]["split"] = "test"
    
    # 保存数据
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Created synthetic geoloc dataset:")
    print(f"  Train: {len(train_data)} samples -> {train_path}")
    print(f"  Test: {len(test_data)} samples -> {test_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess geoloc dataset for training")
    parser.add_argument("--input_file", type=str, help="Input CSV or JSON file path")
    parser.add_argument("--output_dir", type=str, default="~/data/geoloc", 
                        help="Output directory for processed data")
    parser.add_argument("--data_source", type=str, default="geoloc", 
                        help="Data source name")
    parser.add_argument("--image_column", type=str, default="image", 
                        help="Column name for image paths")
    parser.add_argument("--lat_column", type=str, default="latitude", 
                        help="Column name for latitude")
    parser.add_argument("--lon_column", type=str, default="longitude", 
                        help="Column name for longitude") 
    parser.add_argument("--context_column", type=str, default=None,
                        help="Column name for context information")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train/test split ratio")
    parser.add_argument("--create_synthetic", action="store_true",
                        help="Create synthetic data for testing")
    parser.add_argument("--num_synthetic", type=int, default=1000,
                        help="Number of synthetic samples to create")
    
    args = parser.parse_args()
    
    # 展开用户目录
    output_dir = os.path.expanduser(args.output_dir)
    
    if args.create_synthetic:
        create_synthetic_geoloc_data(output_dir, args.num_synthetic)
    elif args.input_file:
        process_geoloc_dataset(
            input_file=args.input_file,
            output_dir=output_dir,
            data_source=args.data_source,
            image_column=args.image_column,
            lat_column=args.lat_column,
            lon_column=args.lon_column,
            context_column=args.context_column,
            split_ratio=args.split_ratio
        )
    else:
        print("Please specify --input_file or use --create_synthetic")
        parser.print_help()


if __name__ == "__main__":
    main() 