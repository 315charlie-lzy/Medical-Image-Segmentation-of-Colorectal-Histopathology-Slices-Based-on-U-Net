myKiU-Net: Medical Image Segmentation of Colorectal Histopathology Slices

基于 U-Net / KiU-Net 的结肠组织学切片像素级分割（腺体 vs 背景），在轻量结构上引入光谱池化、改进激活、注意力门、残差与密集连接等模块，提升细胞腺体边界与形态结构的分割质量。
本项目提供多种 U-Net / KiU-Net 变体在 2D 医学图像二类语义分割上的训练与验证脚本。支持选择不同模型、单/多 GPU 训练（DataParallel）、按周期评估与导出可视化预测，并计算 Dice(F1)、mIoU、像素精度 等指标。适用于如结肠腺体、视网膜血管等前景/背景分割任务，开箱即用、易于复现与对比。

This repository offers PyTorch training/evaluation scripts for multiple U-Net / KiU-Net variants on 2D binary medical image segmentation. It supports model selection, single/multi-GPU training (DataParallel), periodic validation with visualized mask exports, and metrics including Dice(F1), mIoU, and Pixel Accuracy. Ideal for foreground/background tasks (e.g., glands, retinal vessels), designed for quick reproduction and fair comparison.

✨ Highlights

任务：结肠组织切片的像素级语义分割（Gland vs Background）。

数据：GlaS Challenge 数据集（共 165 张）。本项目将图像统一 resize 到 128×128，并采用 train(85) / val(40) / test(40) 的自定义划分；同时在 RITE 视网膜血管数据集上做泛化验证。

指标：IoU（交并比）与 F1-score（精确率与召回率的调和均值）。

方法基线：U-Net、KiU-Net。

我们的模型：myKiU-Net（基于 KiU-Net 的改进实现），在不显著增加复杂度的前提下对结构与训练细节做多处优化。

🧠 Method (myKiU-Net)

在 KiU-Net 框架基础上进行以下改造与消融评估：

卷积块与跳连策略微调

调整卷积块层序（Conv2D → ReLU → Upsampling/MaxPooling），并在编码器/解码器间采用改进的 CRFB（卷积残差融合块） 进行跨层信息融合。

Loss 优化

由纯交叉熵 CE 切换为组合损失（例如 CE + 边界/区域项），以兼顾腺体内部与边界质量。

Pooling 优化：MaxPool → Hartley 光谱池化

在频域进行能量更平滑的下采样，抑制 aliasing，保留形态结构细节，有利于腺体边界与细丝状结构。

激活函数替换：ReLU → ELiSH / HardELiSH

连续且带有指数/线性混合特性，缓解梯度消失并提升特征表达的平滑性。

注意力门（Attention Gate）

对 skip/融合特征进行相关性筛选，抑制无关背景噪声，突出腺体区域。

残差连接（Residual）与密集块（Dense Block）

提升梯度流动与特征复用能力，进一步改善细粒度结构的恢复。

注：以上模块均做了可选与组合消融，以验证各部分对 IoU / F1 的贡献与可叠加性。

📊 Results & Evaluation

定量：在 GlaS 的验证与测试划分上，IoU / F1 相比基线（U-Net / KiU-Net）呈稳定提升趋势（详见项目中的表格与日志）。

可视化：在边界贴合度、腺体整体形态保持以及细丝状结构的连贯性方面，myKiU-Net 的预测更接近标注；在 RITE 上的扩展实验也显示出良好的跨数据集泛化。
