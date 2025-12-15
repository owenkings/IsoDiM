
# 优化方案

## 📁 文件修改清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `models/DiffTransformer.py` | ✨ **新建** | JiT-style 1D Transformer 扩散模块 |
| `models/DiffMLPs.py` | ✏️ 修改 | 注册 DiffTransformer 模型 |
| `models/MARDM.py` | ✏️ 修改 | 添加 4 个新的 FSQ-MARDM-DiT 变体 |
| `train_MARDM.py` | ✏️ 修改 | 更新模型选项 |
| `evaluation_MARDM.py` | ✏️ 修改 | 更新模型选项 |

---

## 🆕 新增模型

```
可用模型: [
    'MARDM-DDPM-XL',        # 原版 MARDM
    'MARDM-SiT-XL',          
    'FSQ-MARDM-SiT-XL',     # FSQ + MLP (原版)
    'FSQ-MARDM-DDPM-XL',     
    'FSQ-MARDM-DiT-S',      # FSQ + Transformer (新 JiT-style)
    'FSQ-MARDM-DiT-B',      
    'FSQ-MARDM-DiT-L',      
    'FSQ-MARDM-DiT-XL'      # 推荐：与原版对标
]
```

---

## 🚀 使用方法

### 训练新模型 (JiT-style DiffTransformer)

```bash
python train_MARDM.py \
    --name FSQ_MARDM_DiT_XL \
    --ae_name FSQ_AE_High \
    --ae_model FSQ_AE_High \
    --model FSQ-MARDM-DiT-XL \
    --dataset_name t2m \
    --batch_size 64 \
    --epoch 500 \
    --need_evaluation
```

### 评估模型

```bash
python evaluation_MARDM.py \
    --name FSQ_MARDM_DiT_XL \
    --ae_name FSQ_AE_High \
    --ae_model FSQ_AE_High \
    --model FSQ-MARDM-DiT-XL \
    --dataset_name t2m
```

---

## 📊 架构对比

| 特性 | FSQ-MARDM-SiT-XL (原版) | FSQ-MARDM-DiT-XL (JiT) |
|------|-------------------------|------------------------|
| 扩散骨干 | MLP (ResBlock) | 1D Transformer |
| 帧间交互 | ❌ 无（独立处理） | ✅ Self-Attention |
| 条件注入 | Add | adaLN + Token-wise Add |
| 序列结构 | Flatten (B*L, D) | 保持 (B, L, D) |
| 参数量 | ~330M | ~330M |

---

## 说明

**`DiffTransformer_XPred` 是完全独立实现的 x-prediction**，没有依赖 `transport.py`！

```python:446:486:models/DiffTransformer.py
    def forward(self, target: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ...
        # 3. 构造含噪输入 x_t = t * x_1 + (1-t) * x_0
        t_expand = t.view(-1, 1)
        x_t = t_expand * target + (1 - t_expand) * x_0
        
        # 4. 模型预测 x_1 (不是 velocity!)
        x_1_pred = self.net(x_t, t, z)
        
        # 5. 计算 x-prediction 损失: MSE(pred, x_1)
        loss = (x_1_pred - target) ** 2  # ✅ target 就是 x_1
```

**当前状态：**
- ✅ `DiffTransformer_XPred` - 自己实现的 x-prediction，**不依赖 transport.py**
- ⚠️ `DiffTransformer_SiT` - 使用 transport.py 的 v-prediction (对照组)

---
## x-prediction

### 清单

| 文件 | 修改内容 |
|------|----------|
| `diffusions/transport/transport.py` | 添加 `ModelType.DATA` 枚举，修改 `training_losses`、`get_drift`、`get_score` |
| `diffusions/transport/__init__.py` | 支持 `prediction="x"` / `prediction="data"` 参数 |
| `models/DiffTransformer.py` | 添加 `DiffTransformer_Transport_XPred` 类 |

### 当前 x-prediction 实现

| 实现 | 类名 | 特点 |
|------|------|------|
| ⭐ **推荐** | `DiffTransformer_XPred` | 独立实现，完全控制训练/采样逻辑 |
| 替代方案 | `DiffTransformer_Transport_XPred` | 基于 transport.py 框架 |

### 验证结果

```
[1] DiffTransformer_XPred (独立实现):     Loss: 1.3408
[2] DiffTransformer_Transport_XPred:       Loss: 1.3408  ← 一致！
```

### transport.py 新增支持

```python
# 现在可以这样使用
transport_x = create_transport(prediction="x")  # 或 "data"
# transport.py 会自动:
# - 训练时: loss = MSE(model_output, x1)  ← 直接预测干净数据
# - 采样时: velocity = (x1_pred - x_t) / (1-t)  ← 自动转换
```

# v-prediction vs x-prediction 详解

## 1. 基本定义

在扩散模型/Flow Matching 中，从噪声 $x_0$ 到干净数据 $x_1$ 的插值过程为：

$$x_t = t \cdot x_1 + (1-t) \cdot x_0$$

模型需要预测**某个目标**来完成去噪。两种范式的区别：

| | **v-prediction** | **x-prediction** |
|--|------------------|------------------|
| **预测目标** | $v = x_1 - x_0$ (velocity) | $x_1$ (干净数据) |
| **物理含义** | 预测"噪声到数据的方向" | 直接预测"最终目标" |
| **输出范围** | $(-\infty, +\infty)$ | 有界（FSQ: $[-1, 1]$） |

## 2. 数学对比

### v-prediction (SiT 默认)
```
训练: Loss = ||model(x_t, t) - (x_1 - x_0)||²
采样: dx/dt = model(x_t, t)  → 直接积分
```
模型学习的是"如何从噪声走向数据"的方向向量。

### x-prediction (JiT 风格)
```
训练: Loss = ||model(x_t, t) - x_1||²
采样: velocity = (x̂_1 - x_t) / (1-t)  → 需要转换
```
模型直接学习"干净数据长什么样"。

## 3. 各自的优缺点

### v-prediction
| 优点 | 缺点 |
|------|------|
| 采样数学简洁（直接积分） | 输出无界，可能数值不稳定 |
| 广泛使用，代码成熟 | 需要学习噪声的复杂分布 |
| 对高斯噪声建模效果好 | 在离散/有界空间可能次优 |

### x-prediction
| 优点 | 缺点 |
|------|------|
| 目标明确，直接预测最终结果 | 采样需要额外转换步骤 |
| 输出有界（FSQ: [-1,1]），训练稳定 | t→1 时数值可能不稳定 |
| 在离散/量化空间效果更好 | 相对较新，代码成熟度低 |
| 模型可以"直接看到目标" | |

## 4. 为什么 FSQ 空间更适合 x-prediction？

```
FSQ 空间特点：
- 离散网格点（如 [-1, -0.5, 0, 0.5, 1]）
- 有界范围 [-1, 1]
- 拓扑结构清晰

v-prediction 问题：
- 预测的 velocity 可能很大（如 x_0 = 3, x_1 = -0.5 → v = -3.5）
- 模型需要学习"噪声的分布"，这很复杂

x-prediction 优势：
- 直接预测"哪个网格点"
- 输出天然在 [-1, 1] 范围内
- 任务更简单：分类/回归到有限网格
```

## 5. 推荐

**对于 FSQ-MARDM 项目，推荐 x-prediction**，原因：

| 因素        | x-prediction 优势      |
| --------- | -------------------- |
| **目标空间**  | FSQ 坐标天然有界           |
| **任务难度**  | "预测正确的网格" < "预测噪声方向" |
| **训练稳定性** | 输出有界，不会爆炸            |
| **理论支持**  | JiT 论文验证了在离散空间的有效性   |

## 6. 实验建议

如果你有时间，可以对比实验：

```bash
# x-prediction (推荐)
python train_MARDM.py --model FSQ-MARDM-DiT-XL ...

# v-prediction (对照)
python train_MARDM.py --model FSQ-MARDM-DiT-VPred-XL ...
```

预期：x-prediction 在 FSQ 空间应该收敛更快、FID 更低。

---

**总结**：v-prediction 是更通用的选择，但在 FSQ 这种有界离散空间，x-prediction 更合适，因为任务从"预测噪声方向"变成了"预测正确的网格坐标"，这是一个更简单、更直接的目标。