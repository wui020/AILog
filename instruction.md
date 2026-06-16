# 说明

## [run_cosmos3_nano](run_cosmos3_nano.sh)

2026年6月16日
1. 在3090 24G显卡上部署 英伟达的 nvidia/Cosmos3-Nano 这个世界模型 16B 33G 的BF16模型
2. 目前只有文生视频功能 需要扩充五种输入输出 在此基础上


### 一、 这个脚本做了什么才让 Cosmos3-Nano 跑通跑成功？

`nvidia/Cosmos3-Nano` 是一个拥有 **160 亿参数（16B）** 的混合 Transformer（Mixture-of-Transformers, MoT）物理世界基础模型。在不进行优化的情况下，仅加载其 BF16 格式的完整权重就需要消耗约 **33GB** 显存，进行一次推理很容易突破 **40~48GB** 显存限制，普通的消费级显卡（如 RTX 3090、4090 的 24GB 显存）会直接报错（OOM）。

本脚本之所以能让该模型在有限的显存（甚至低于 24GB）上成功跑通，是因为实施了以下多重硬核优化策略：

1. **8-bit 量化技术（BitsAndBytes 8-bit Quantization）**：
   脚本通过配置 `PipelineQuantizationConfig`，在加载时将原本庞大的 16B 变体压缩为 8-bit 精度。这使得模型的权重显存占用折半（降至约 16GB 左右），是能够在消费级显卡运行的前提。
2. **VAE 空间与时间分块技术（VAE Tiling + Slicing）**：
   在视频生成中，VAE 编解码阶段（将潜空间张量还原为视频像素）是显存消耗的超级黑洞。通过开启 `pipe.vae.enable_tiling()` 与 `pipe.vae.enable_slicing()`，脚本强制 VAE 将视频在空间（宽、高）和时间（帧维度）上拆分为极小的“瓦片”和“切片”逐一解码，彻底消除了还原视频时的 VRAM 峰值。
3. **安全拦截补丁（Safety Checker Bypass Patch）**：
   Cosmos 3 默认集成了基于网络黑名单和多模态分类的安全过滤器 `cosmos_guardrail`。在本地局域网或受限的算力节点上，它不仅会因为尝试联网更新而报错，还会额外加载轻量分类模型占用显存。脚本通过 Python 的动态特性（Monkey Patch）劫持了其 `__init__` 函数，不加载安全过滤器，同时节省了显存并杜绝了报错。
4. **PyTorch 显存碎片整理与预分配（Expandable Segments）**：
   通过设置环境变量 `expandable_segments:True`，强制 PyTorch 动态扩展内存块而非频繁申请释放。这有效阻止了在多步扩散（Denoising Steps）迭代中因显存碎片化导致的虚假 OOM。
5. **Cosmos 专有的时空 4 倍数帧数修正（VAE Frame Constraint）**：
   Cosmos 的 VAE 具有严格的时空下采样率。帧数 $N$ 必须严格满足 $(N-1) \pmod 4 == 0$（如 1, 5, 9, 13...）才能被 VAE 正确解码，否则会导致底层 CUDA 报错或画面严重撕裂。脚本中的 `check_frames` 辅助函数在后台自动完成了这套物理对齐数学计算。

---

## [run_cosmos4_nano](run_cosmos4_nano.sh)

1. 增加图生视频入口
2. 去掉无效图标