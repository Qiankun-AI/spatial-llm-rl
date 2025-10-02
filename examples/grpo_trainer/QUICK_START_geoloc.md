# Geoloc训练快速开始指南

## 🚀 快速开始

### 1. 创建测试数据

```bash
# 进入脚本目录
cd examples/grpo_trainer

# 创建合成数据用于测试
python preprocess_geoloc_data.py --create_synthetic --num_synthetic 1000 --output_dir ~/data/geoloc
```

### 2. 测试Reward函数

```bash
# 运行测试确保reward函数工作正常
python test_geoloc_reward.py
```

### 3. 开始训练

#### 选项A：使用欧式距离（适合小范围区域）

```bash
# 基本训练
bash run_geoloc_qwen2_5_vl-7b.sh

# 或使用sglang引擎
bash run_geoloc_qwen2_5_vl-7b.sh sglang
```

#### 选项B：使用Haversine距离（适合全球范围）

```bash
bash run_geoloc_qwen2_5_vl-7b_haversine.sh
```

## 📊 监控训练

训练过程中可以通过wandb查看：
- 平均reward值
- 距离误差统计  
- 训练损失和学习率

访问 https://wandb.ai 查看训练日志。

## 🛠 自定义数据

### 准备您的数据

您的CSV文件应该包含：
- `latitude`: 纬度
- `longitude`: 经度
- `image`: 图像路径（可选）
- `description`: 描述信息（可选）

### 处理数据

```bash
python preprocess_geoloc_data.py \
    --input_file /path/to/your/data.csv \
    --output_dir ~/data/geoloc \
    --lat_column latitude \
    --lon_column longitude \
    --image_column image_path \
    --context_column description
```

## ⚙️ 调整参数

### 修改reward函数参数

编辑训练脚本中的这些参数：

```bash
# 欧式距离模式
reward_model.custom_reward_function.reward_kwargs.use_haversine=False \
reward_model.custom_reward_function.reward_kwargs.max_distance=1.0 \
reward_model.custom_reward_function.reward_kwargs.distance_penalty_factor=2.0 \

# Haversine距离模式  
reward_model.custom_reward_function.reward_kwargs.use_haversine=True \
reward_model.custom_reward_function.reward_kwargs.max_distance=50.0 \
reward_model.custom_reward_function.reward_kwargs.distance_penalty_factor=0.1 \
```

### 参数说明

- `max_distance`: 最大有效距离，超过此距离reward为0
  - 欧式距离：建议0.1-2.0度
  - Haversine距离：建议10-100公里

- `distance_penalty_factor`: 距离惩罚因子
  - 值越大，reward衰减越快
  - 欧式距离：建议1.0-5.0
  - Haversine距离：建议0.01-0.5

## 🔧 故障排除

### 常见问题

1. **模型输出无法提取坐标**
   - 检查模型输出格式
   - 调整提示词让模型输出标准格式

2. **Reward始终为0**
   - 检查`max_distance`是否设置过小
   - 确认ground_truth格式正确

3. **训练不收敛**
   - 降低学习率
   - 调整`distance_penalty_factor`
   - 增加训练数据量

### 调试命令

```bash
# 检查数据格式
python -c "
import pandas as pd
df = pd.read_parquet('~/data/geoloc/train.parquet')
print('数据形状:', df.shape)
print('第一个样本prompt:', df.iloc[0]['prompt'])
"

# 测试reward函数
python -c "
from geoloc_reward_function import geoloc_euclidean_reward_function
result = geoloc_euclidean_reward_function('test', '(39.9, 116.4)', {'latitude': 39.9042, 'longitude': 116.4074})
print(f'Reward: {result}')
"
```

## 📈 性能优化

### GPU内存优化

根据您的GPU内存调整：

```bash
# 8GB GPU
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

# 16GB GPU  
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8

# 24GB+ GPU
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12
```

### 加速训练

```bash
# 使用更大的batch size
data.train_batch_size=1024

# 启用tensor并行
actor_rollout_ref.rollout.tensor_model_parallel_size=4

# 使用更快的引擎
bash run_geoloc_qwen2_5_vl-7b.sh sglang
```

## 📝 输出格式

训练后的模型应该能够输出如下格式：

```
<think>
根据图像中的建筑风格和地理特征，这看起来像是中国的一个城市。
从建筑物的样式和周围环境来看，可能是北京地区。
考虑到图像中的具体特征，我估计位置大约在北京市中心附近。
</think>

最终坐标预测：(39.9042, 116.4074)
```

## 🎯 下一步

1. 使用真实的geoloc数据集
2. 尝试不同的模型（如Qwen2.5-VL-32B）
3. 调优reward函数参数
4. 添加更多上下文信息
5. 实验不同的提示词策略 