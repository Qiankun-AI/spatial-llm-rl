#!/usr/bin/env python3
"""
测试geoloc reward函数
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from geoloc_reward_function import (
    geoloc_euclidean_reward_function,
    extract_coordinates,
    calculate_euclidean_distance,
    calculate_haversine_distance
)

def test_coordinate_extraction():
    """测试坐标提取功能"""
    print("=== 测试坐标提取功能 ===")
    
    test_cases = [
        "(39.9042, 116.4074)",
        "[31.2304, 121.4737]",
        '{"latitude": 23.1291, "longitude": 113.3240}',
        '{"lat": 22.5431, "lon": 114.0579}',
        "坐标是 30.2741, 120.1551",
        "最终答案：(30.6598, 104.0633)",
        "无效的坐标格式",
    ]
    
    for i, test_str in enumerate(test_cases):
        coords = extract_coordinates(test_str)
        print(f"测试 {i+1}: {test_str}")
        print(f"结果: {coords}")
        print()

def test_distance_calculation():
    """测试距离计算功能"""
    print("=== 测试距离计算功能 ===")
    
    # 北京天安门和上海外滩
    beijing = (39.9042, 116.4074)
    shanghai = (31.2304, 121.4737)
    
    euclidean_dist = calculate_euclidean_distance(beijing, shanghai)
    haversine_dist = calculate_haversine_distance(beijing, shanghai)
    
    print(f"北京天安门: {beijing}")
    print(f"上海外滩: {shanghai}")
    print(f"欧式距离: {euclidean_dist:.4f} 度")
    print(f"Haversine距离: {haversine_dist:.2f} 公里")
    print()

def test_reward_function():
    """测试reward函数"""
    print("=== 测试Reward函数 ===")
    
    # 测试用例：预测坐标和真实坐标
    test_cases = [
        {
            "name": "完全匹配",
            "prediction": "(39.9042, 116.4074)",
            "ground_truth": {"latitude": 39.9042, "longitude": 116.4074},
        },
        {
            "name": "轻微偏差",
            "prediction": "(39.9050, 116.4080)",
            "ground_truth": {"latitude": 39.9042, "longitude": 116.4074},
        },
        {
            "name": "较大偏差",
            "prediction": "(40.0000, 117.0000)",
            "ground_truth": {"latitude": 39.9042, "longitude": 116.4074},
        },
        {
            "name": "无效预测",
            "prediction": "无法确定位置",
            "ground_truth": {"latitude": 39.9042, "longitude": 116.4074},
        },
    ]
    
    for case in test_cases:
        print(f"测试案例: {case['name']}")
        print(f"预测: {case['prediction']}")
        print(f"真实: {case['ground_truth']}")
        
        # 测试欧式距离reward
        euclidean_reward = geoloc_euclidean_reward_function(
            data_source="test",
            solution_str=case['prediction'],
            ground_truth=case['ground_truth'],
            use_haversine=False,
            max_distance=1.0,
            distance_penalty_factor=2.0
        )
        
        # 测试haversine距离reward
        haversine_reward = geoloc_euclidean_reward_function(
            data_source="test",
            solution_str=case['prediction'],
            ground_truth=case['ground_truth'],
            use_haversine=True,
            max_distance=50.0,
            distance_penalty_factor=0.1
        )
        
        print(f"欧式距离Reward: {euclidean_reward:.4f}")
        print(f"Haversine距离Reward: {haversine_reward:.4f}")
        print()

def test_different_ground_truth_formats():
    """测试不同的ground_truth格式"""
    print("=== 测试不同Ground Truth格式 ===")
    
    prediction = "(39.9042, 116.4074)"
    
    ground_truth_formats = [
        {"latitude": 39.9042, "longitude": 116.4074},
        {"lat": 39.9042, "lon": 116.4074},
        {"coordinates": [39.9042, 116.4074]},
        "(39.9042, 116.4074)",  # 字符串格式
    ]
    
    for i, gt in enumerate(ground_truth_formats):
        reward = geoloc_euclidean_reward_function(
            data_source="test",
            solution_str=prediction,
            ground_truth=gt
        )
        print(f"格式 {i+1}: {gt}")
        print(f"Reward: {reward:.4f}")
        print()

def main():
    """主测试函数"""
    print("开始测试Geoloc Reward函数...")
    print("=" * 50)
    
    test_coordinate_extraction()
    test_distance_calculation()
    test_reward_function()
    test_different_ground_truth_formats()
    
    print("测试完成！")

if __name__ == "__main__":
    main() 