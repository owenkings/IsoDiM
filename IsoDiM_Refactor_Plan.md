# IsoDiM 项目重构与迁移指南

**目标：** 基于原 MARDM 项目中验证成功的 "FSQ + JiT (x-prediction)" 架构，构建一个全新的、干净的、无冗余的独立项目，命名为 **IsoDiM** (Isometric Discrete Diffusion Model)。

**核心原则：**
1. **全新开始：** 所有新生成的文件创建日期均为今天。
2. **彻底改名：** 移除所有 "MARDM" 命名痕迹，替换为 "IsoDiM" 或对应模块名。
3. **去冗余：** 仅保留 FSQ 和 JiT 相关的核心代码，移除原版连续 AE、MLP Diffusion 等不再使用的旧逻辑。
4. **逻辑继承：** 核心算法逻辑（FSQ量化、JiT Transformer、x-prediction、Grid Snapping）必须与之前验证通过的代码完全一致。
5. **完整性：** 保留所有必要的数据处理、评估工具（Glove, Evaluators等）。

---

## 1. 📁 目录结构映射 (File Mapping)

我们将从左侧 (Old) 迁移并重命名到右侧 (New)：

| 原文件/路径 | 新文件/路径 | 说明 |
| :--- | :--- | :--- |
| `MARDM/` | `IsoDiM/` | 项目根目录 |
| `models/FSQ.py` | `models/Quantizer.py` | 核心 FSQ 模块 |
| `models/AE.py` | `models/Tokenizer.py` | 仅保留 FSQ-AE 逻辑，改名为 Tokenizer |
| `models/DiffTransformer.py` | `models/Transformer.py` | JiT 核心 Backbone (1D DiT) |
| `models/MARDM.py` | `models/IsoDiM.py` | 模型主入口 (集成 Tokenizer + Transformer) |
| `train_AE.py` | `train_Tokenizer.py` | 训练 FSQ Tokenizer 的脚本 |
| `train_MARDM.py` | `train_IsoDiM.py` | 训练主模型的脚本 |
| `evaluation_MARDM.py` | `evaluation_IsoDiM.py` | 评估脚本 |
| `evaluation_AE.py` | `evaluation_Tokenizer.py` | 重建质量评估脚本 |
| `sample.py` | `sample.py` | 推理可视化脚本 (需适配新类名) |
| `models/DiffMLPs.py` | **[删除]** | 不再需要 MLP 架构 |
| `utils/*` | `utils/*` | 工具库全量保留 (eval_utils, datasets等) |
| `diffusions/*` | `diffusions/*` | 扩散库保留 (需适配 x-prediction) |
| `datasets/*` | `datasets/*` | 数据集加载逻辑保留 |

---

## 2. 🧱 代码重构细节 (Refactoring Details)

### A. 模型定义 (`models/`)

1.  **`Tokenizer.py` (原 AE.py):**
    * **清理：** 移除原版连续 `AE` 类，移除 `VQ_AE` 类。
    * **保留：** 仅保留 `FSQ_AE` 的逻辑（现在作为主类）。
    * **重命名：** 类名从 `FSQ_AE` 改为 `IsoDiM_Tokenizer`。
    * **功能：** 必须包含 `encode_with_fsq_output` (用于训练) 和 `decode_from_fsq`。

2.  **`Transformer.py` (原 DiffTransformer.py):**
    * **保留：** 之前实现的 JiT-style 1D Transformer。
    * **类名：** 统一为 `IsoDiM_Transformer`。
    * **特性：** 确保保留 `adaLN` 和 Token-wise condition 注入逻辑。

3.  **`IsoDiM.py` (原 MARDM.py):**
    * **重命名：** 类名从 `FSQ_MARDM` 改为 `IsoDiM`。
    * **集成：** 初始化时直接实例化 `IsoDiM_Transformer`，不再支持旧的 MLP 选项。
    * **Grid Snapping：** 必须保留 `snap_to_fsq_grid` 函数，并在 `generate` 和 `edit` 方法中调用。

### B. 扩散底层 (`diffusions/`)

* **x-prediction：** 确保 `transport.py` 中包含 `ModelType.DATA` 和 x-prediction 的损失计算逻辑。这是 IsoDiM 的核心驱动。

### C. 训练脚本 (`train_*.py`)

1.  **`train_Tokenizer.py`：**
    * 默认模型参数改为新的 `IsoDiM_Tokenizer`。
    * 移除旧 AE 相关的冗余参数。

2.  **`train_IsoDiM.py`：**
    * 引用新的 `IsoDiM` 和 `IsoDiM_Tokenizer` 类。
    * **强制配置：** 默认使用 `x-prediction` (prediction='data')。
    * 移除 MLP 相关的参数（如 `diffmlps_model`）。

---

## 3. 🛡️ 质量保证 (Quality Assurance)

* **Logic Check:** 重构后的 `IsoDiM` 在输入输出维度、Forward 流程上必须与之前的 `FSQ-MARDM-DiT-XL` 此时此刻的代码完全一致。
* **Imports:** 修正所有文件中的 import 路径，例如 `from models.MARDM import ...` 改为 `from models.IsoDiM import ...`。
* **Defaults:** 将推荐配置（如 FSQ-High, DiT-XL, x-prediction）设为脚本的默认参数。

---

这个计划旨在创建一个生产级、干净的代码库，即刻执行。