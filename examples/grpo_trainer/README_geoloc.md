# Geoloc任务训练指南

本目录包含了用于地理位置预测（geoloc）任务的训练脚本和工具，使用基于欧式距离的自定义reward函数。

## 文件说明

### 核心文件

- `geoloc_reward_function.py` - 自定义的reward函数，支持欧式距离和haversine距离计算
- `run_geoloc_qwen2_5_vl-7b.sh` - 使用欧式距离的训练脚本
- `run_geoloc_qwen2_5_vl-7b_haversine.sh` - 使用haversine距离的训练脚本
- `preprocess_geoloc_data.py` - 数据预处理脚本

### 工具特性

#### Reward函数特性

1. **多种距离计算方式**：
   - 欧式距离：适用于小范围区域的快速计算
   - Haversine距离：适用于地理坐标的精确距离计算（单位：公里）

2. **灵活的坐标提取**：
   - 支持多种坐标格式：`(lat, lon)`、`[lat, lon]`、JSON格式等
   - 自动识别和提取模型输出中的坐标信息

3. **可配置的reward计算**：
   - `max_distance`：最大距离阈值，超过此距离reward为0
   - `distance_penalty_factor`：距离惩罚因子，控制reward衰减速度
   - `use_haversine`：是否使用haversine距离

## 使用方法

### 1. 数据准备

#### 方法一：使用现有数据集

如果您有包含经纬度信息的CSV或JSON数据集：

```bash
# 处理CSV数据集
python preprocess_geoloc_data.py \
    --input_file /path/to/your/dataset.csv \
    --output_dir ~/data/geoloc \
    --lat_column latitude \
    --lon_column longitude \
    --image_column image_path \
    --context_column description

# 处理JSON数据集
python preprocess_geoloc_data.py \
    --input_file /path/to/your/dataset.json \
    --output_dir ~/data/geoloc \
    --data_source your_dataset_name
```

#### 方法二：创建合成数据集（用于测试）

```bash
# 创建1000个合成样本用于测试
python preprocess_geoloc_data.py \
    --create_synthetic \
    --num_synthetic 1000 \
    --output_dir ~/data/geoloc
```

### 2. 训练模型

#### 使用欧式距离reward

```bash
# 基本训练
bash run_geoloc_qwen2_5_vl-7b.sh

# 使用sglang引擎
bash run_geoloc_qwen2_5_vl-7b.sh sglang
```

#### 使用haversine距离reward

```bash
# 使用地理上更精确的haversine距离
bash run_geoloc_qwen2_5_vl-7b_haversine.sh
```

### 3. 自定义配置

您可以通过修改脚本中的参数来自定义训练：

```bash
# 自定义reward参数
python3 -m verl.trainer.main_ppo \
    ... \
    reward_model.custom_reward_function.reward_kwargs.use_haversine=True \
    reward_model.custom_reward_function.reward_kwargs.max_distance=10.0 \
    reward_model.custom_reward_function.reward_kwargs.distance_penalty_factor=0.5 \
    ...
```

## 配置参数说明

### 数据配置

- `data.train_files`: 训练数据文件路径
- `data.val_files`: 验证数据文件路径
- `data.max_response_length`: 最大响应长度（geoloc任务通常较短）
- `data.image_key`: 图像数据的键名

### Reward函数配置

- `reward_model.custom_reward_function.path`: reward函数文件路径
- `reward_model.custom_reward_function.name`: 函数名称
- `reward_model.custom_reward_function.reward_kwargs.use_haversine`: 是否使用haversine距离
- `reward_model.custom_reward_function.reward_kwargs.max_distance`: 最大有效距离
- `reward_model.custom_reward_function.reward_kwargs.distance_penalty_factor`: 距离惩罚系数

### 训练配置

- `trainer.project_name`: wandb项目名称
- `trainer.experiment_name`: 实验名称
- `trainer.total_epochs`: 训练轮数

## Reward函数详解

### 欧式距离模式

```python
distance = sqrt((lat1 - lat2)² + (lon1 - lon2)²)
reward = exp(-distance * penalty_factor)
```

- 优点：计算快速，适合小范围区域
- 缺点：不考虑地球曲率，大范围时误差较大

### Haversine距离模式

```python
# 使用球面几何计算真实地理距离
distance = haversine_distance(coord1, coord2)  # 单位：公里
reward = exp(-distance * penalty_factor)
```

- 优点：地理上精确，适合全球范围
- 缺点：计算稍慢

## 数据格式要求

### 输入数据格式

CSV文件应包含以下列：
- `latitude`: 纬度
- `longitude`: 经度  
- `image` (可选): 图像路径
- `description` (可选): 上下文描述

### 处理后的数据格式

```json
{
  "data_source": "geoloc",
  "prompt": [
    {
      "role": "user", 
      "content": "请根据提供的图像和上下文信息，预测图像拍摄的地理位置..."
    }
  ],
  "images": ["path/to/image.jpg"],
  "ability": "geoloc",
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "latitude": 39.9042,
      "longitude": 116.4074,
      "coordinates": [39.9042, 116.4074]
    }
  }
}
```

## 监控和调试

### 使用wandb监控训练

脚本已配置wandb日志记录，训练过程中会记录：
- 平均reward值
- 距离误差统计
- 训练损失和学习率

### 调试reward函数

如果reward函数出现问题，检查：
1. 模型输出格式是否包含坐标信息
2. 坐标提取正则表达式是否匹配
3. ground_truth格式是否正确

## 性能优化建议

1. **批次大小调整**：根据GPU内存调整`ppo_micro_batch_size_per_gpu`
2. **学习率调优**：地理位置任务可能需要较小的学习率
3. **距离阈值设置**：根据任务精度要求调整`max_distance`
4. **惩罚因子调整**：较大的`distance_penalty_factor`会使reward更快衰减

## 故障排除

### 常见问题

1. **坐标提取失败**：检查模型输出格式，可能需要调整提示词
2. **Reward始终为0**：检查`max_distance`设置是否过小
3. **训练不收敛**：尝试调整学习率和惩罚因子

### 调试命令

```bash
# 测试reward函数
python -c "
from geoloc_reward_function import geoloc_euclidean_reward_function
result = geoloc_euclidean_reward_function('test', '(39.9, 116.4)', {'latitude': 39.9042, 'longitude': 116.4074})
print(f'Reward: {result}')
"
```

## 扩展和自定义

### 添加新的距离计算方法

在`geoloc_reward_function.py`中添加新的距离计算函数，并在主函数中添加相应的选项。

### 支持新的坐标格式

修改`extract_coordinates`函数中的正则表达式，支持更多坐标格式。

### 自定义reward策略

可以实现更复杂的reward策略，例如：
- 基于距离区间的分段reward
- 考虑置信度的加权reward
- 多目标优化的组合reward 