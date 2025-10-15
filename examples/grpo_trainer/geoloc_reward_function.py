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
Geoloc reward function based on euclidean distance
"""

import re
import math
import json
from typing import Union, Tuple, Optional


def extract_coordinates(solution_str: str) -> Optional[Tuple[float, float]]:
    """
    从解决方案字符串中提取坐标信息
    支持多种格式：
    - (lat, lon) 
    - [lat, lon]
    - {"latitude": lat, "longitude": lon}
    - {"lat": lat, "lon": lon}
    - lat, lon
    """
    # 尝试提取JSON格式
    json_pattern = r'\{[^}]*"(?:lat(?:itude)?|lng|lon(?:gitude)?)"\s*:\s*[-+]?\d*\.?\d+[^}]*\}'
    json_matches = re.findall(json_pattern, solution_str, re.IGNORECASE)
    for match in json_matches:
        try:
            data = json.loads(match)
            lat = None
            lon = None
            
            # 尝试不同的键名
            for lat_key in ['latitude', 'lat']:
                if lat_key in data:
                    lat = float(data[lat_key])
                    break
            
            for lon_key in ['longitude', 'lon', 'lng']:
                if lon_key in data:
                    lon = float(data[lon_key])
                    break
            
            if lat is not None and lon is not None:
                return (lat, lon)
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    
    # 尝试提取括号或方括号格式 (lat, lon) 或 [lat, lon]
    bracket_pattern = r'[\[\(]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\]\)]'
    bracket_matches = re.findall(bracket_pattern, solution_str)
    if bracket_matches:
        try:
            lat, lon = float(bracket_matches[0][0]), float(bracket_matches[0][1])
            return (lat, lon)
        except ValueError:
            pass
    
    # 尝试提取简单的数字对格式 lat, lon
    number_pattern = r'([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)'
    number_matches = re.findall(number_pattern, solution_str)
    if number_matches:
        try:
            # 取最后一对数字，通常是最终答案
            lat, lon = float(number_matches[-1][0]), float(number_matches[-1][1])
            return (lat, lon)
        except ValueError:
            pass
    
    return None


def calculate_euclidean_distance(pred_coords: Tuple[float, float], true_coords: Tuple[float, float]) -> float:
    """
    计算两个坐标点之间的欧式距离
    Args:
        pred_coords: 预测的坐标 (lat, lon)
        true_coords: 真实的坐标 (lat, lon)
    Returns:
        欧式距离（单位：度）
    """
    lat1, lon1 = pred_coords
    lat2, lon2 = true_coords
    
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def calculate_haversine_distance(pred_coords: Tuple[float, float], true_coords: Tuple[float, float]) -> float:
    """
    计算两个坐标点之间的haversine距离（更适合地理坐标）
    Args:
        pred_coords: 预测的坐标 (lat, lon)
        true_coords: 真实的坐标 (lat, lon)
    Returns:
        距离（单位：公里）
    """
    lat1, lon1 = pred_coords
    lat2, lon2 = true_coords
    
    # 转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # 地球半径（公里）
    R = 6371.0
    
    return R * c


def geoloc_euclidean_reward_function(
    data_source: str, 
    solution_str: str, 
    ground_truth: Union[str, dict], 
    extra_info: Optional[dict] = None,
    use_haversine: bool = False,
    max_distance: float = 1.0,
    distance_penalty_factor: float = 1.0
) -> float:
    """
    基于欧式距离的geoloc reward函数
    
    Args:
        data_source: 数据源标识
        solution_str: 模型生成的解决方案字符串
        ground_truth: 真实的坐标，可以是字符串或字典格式
        extra_info: 额外信息
        use_haversine: 是否使用haversine距离而不是欧式距离
        max_distance: 最大距离阈值，超过此距离reward为0
        distance_penalty_factor: 距离惩罚因子
        
    Returns:
        reward值，范围[0, 1]，距离越近reward越高
    """
    try:
        # 从solution_str中提取预测坐标
        pred_coords = extract_coordinates(solution_str)
        if pred_coords is None:
            return 0.0
        
        # 解析ground_truth坐标
        true_coords = None
        if isinstance(ground_truth, str):
            true_coords = extract_coordinates(ground_truth)
        elif isinstance(ground_truth, dict):
            if 'latitude' in ground_truth and 'longitude' in ground_truth:
                true_coords = (float(ground_truth['latitude']), float(ground_truth['longitude']))
            elif 'lat' in ground_truth and 'lon' in ground_truth:
                true_coords = (float(ground_truth['lat']), float(ground_truth['lon']))
            elif 'coordinates' in ground_truth:
                coords = ground_truth['coordinates']
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    true_coords = (float(coords[0]), float(coords[1]))
        
        if true_coords is None:
            return 0.0
        
        # 计算距离
        if use_haversine:
            distance = calculate_haversine_distance(pred_coords, true_coords)
            # 对于haversine距离，max_distance单位是公里
            if max_distance < 1:
                max_distance = max_distance * 111.0  # 将度转换为大致的公里数
        else:
            distance = calculate_euclidean_distance(pred_coords, true_coords)
        
        # 如果距离超过阈值，返回0
        if distance > max_distance:
            return 0.0
        
        # 计算基于距离的reward
        # reward = exp(-distance * penalty_factor)
        # 这样距离为0时reward为1，距离越大reward越小
        reward = math.exp(-distance * distance_penalty_factor)
        
        return float(reward)
        
    except Exception as e:
        print(f"Error in geoloc_euclidean_reward_function: {e}")
        print(f"solution_str: {solution_str}")
        print(f"ground_truth: {ground_truth}")
        return 0.0



def geoloc_haversine_reward_function(
    data_source: str, 
    solution_str: str, 
    ground_truth: Union[str, dict], 
    extra_info: Optional[dict] = None,
    use_haversine: bool = False,
    max_distance: float = 1.0,
    distance_penalty_factor: float = 1.0
) -> float:
    """
    基于欧式距离的geoloc reward函数
    
    Args:
        data_source: 数据源标识
        solution_str: 模型生成的解决方案字符串
        ground_truth: 真实的坐标，可以是字符串或字典格式
        extra_info: 额外信息
        use_haversine: 是否使用haversine距离而不是欧式距离
        max_distance: 最大距离阈值，超过此距离reward为0
        distance_penalty_factor: 距离惩罚因子
        
    Returns:
        reward值，范围[0, 1]，距离越近reward越高
    """
    try:
        # 从solution_str中提取预测坐标
        pred_coords = extract_coordinates(solution_str)
        if pred_coords is None:
            return 0.0
        
        # 解析ground_truth坐标
        true_coords = None
        if isinstance(ground_truth, str):
            true_coords = extract_coordinates(ground_truth)
        elif isinstance(ground_truth, dict):
            if 'latitude' in ground_truth and 'longitude' in ground_truth:
                true_coords = (float(ground_truth['latitude']), float(ground_truth['longitude']))
            elif 'lat' in ground_truth and 'lon' in ground_truth:
                true_coords = (float(ground_truth['lat']), float(ground_truth['lon']))
            elif 'coordinates' in ground_truth:
                coords = ground_truth['coordinates']
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    true_coords = (float(coords[0]), float(coords[1]))
        
        if true_coords is None:
            return 0.0
        
        # 计算距离
        # if use_haversine:
        distance = calculate_haversine_distance(pred_coords, true_coords)
        # 对于haversine距离，max_distance单位是公里
        if max_distance < 1:
            max_distance = max_distance * 111.0  # 将度转换为大致的公里数
        # else:
        #     distance = calculate_euclidean_distance(pred_coords, true_coords)
        
        # 如果距离超过阈值，返回0
        if distance > max_distance:
            return 0.0
        
        # 计算基于距离的reward
        # reward = exp(-distance * penalty_factor)
        # 这样距离为0时reward为1，距离越大reward越小
        reward = math.exp(-distance * distance_penalty_factor)
        
        return float(reward)
        
    except Exception as e:
        print(f"Error in geoloc_euclidean_reward_function: {e}")
        print(f"solution_str: {solution_str}")
        print(f"ground_truth: {ground_truth}")
        return 0.0

def geoloc_distance_reward_function(
    data_source: str, 
    solution_str: str, 
    ground_truth: Union[str, dict], 
    extra_info: Optional[dict] = None
) -> float:
    """
    基于距离的简单reward函数，返回负距离作为reward
    距离越小，reward越高（越接近0）
    """
    try:
        pred_coords = extract_coordinates(solution_str)
        if pred_coords is None:
            return -10.0  # 大的惩罚
        
        # 解析ground_truth坐标
        true_coords = None
        if isinstance(ground_truth, str):
            true_coords = extract_coordinates(ground_truth)
        elif isinstance(ground_truth, dict):
            if 'latitude' in ground_truth and 'longitude' in ground_truth:
                true_coords = (float(ground_truth['latitude']), float(ground_truth['longitude']))
            elif 'lat' in ground_truth and 'lon' in ground_truth:
                true_coords = (float(ground_truth['lat']), float(ground_truth['lon']))
        
        if true_coords is None:
            return -10.0
        
        # 计算欧式距离并返回负值
        distance = calculate_euclidean_distance(pred_coords, true_coords)
        return -distance
        
    except Exception as e:
        print(f"Error in geoloc_distance_reward_function: {e}")
        return -10.0 