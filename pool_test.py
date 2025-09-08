import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.nn.modules.utils import _pair
from Hartley_Spectral_Pooling import *
from hybrid_pooling import HybridPoolLayerValid
import torch.nn.functional as F
from Hartley_Spectral_Pooling import *
# 加载图片
#image_path = r"C:\Users\27612\Desktop\Glas_dataset\resized_128x128\train\img\12.png"
#image_path = r"C:\Users\27612\Desktop\Glas_dataset\resized_128x128\Glas\train\img\12.png"
image_path = r"C:\Users\27612\Desktop\KiU-Net-pytorch-master\KiU-Net-pytorch-master\Glas\train\img\12.png"
#image_path = r"C:\Users\27612\Desktop\Glas_dataset\resized_128x128\train\img\25.png"
image = Image.open(image_path)

# 图片预处理
transform = transforms.Compose([  # 调整图片大小
    transforms.ToTensor()            # 将图片转换为Tensor
])
input_image = transform(image)
# plt.imshow(np.transpose(input_image, (1, 2, 0)))
# plt.show()
input_image= input_image.unsqueeze(0)  # 在第0维增加一个维度，表示batch_size为1

# max_pool_output = F.max_pool2d(x, 2,2)
#
# # Spectral pooling
# # spectral_pool = SpectralPoolingFunction.apply(x, x.size(-2)//2, x.size(-1)//2)
# spectral_pool_layer = SpectralPool2d(scale_factor=(0.5,0.5))
# spectral_pool_output = spectral_pool_layer(x)
# ##Concatenate max pooling and spectral pooling results
# concatenated = torch.cat((max_pool_output, spectral_pool_output), dim=1)
# # # 逐元素相加并除以2
# # final_output = torch.add(max_pool_output, spectral_pool_output)
# # final_output = torch.div(final_output, 2)
#
# # Apply 1x1 convolution
# conv = nn.Conv2d(6, 3, kernel_size=1)
# final_output = conv(concatenated)
#
#  # 将张量规范化到0到1之间
# # tensor_normalized = torch.clamp(output, 0, 1)
# # 将张量转换为NumPy数组并交换通道顺序
#
# output_np =final_output.squeeze(0).detach().numpy()
#
# # Display the output
# plt.imshow(np.transpose(output_np, (1, 2, 0)))
# plt.show()




# 定义池化层
maxpool_layer = torch.nn.MaxPool2d(kernel_size=2)
spectral_pool_layer = SpectralPool2d(scale_factor=(0.5, 0.5))

# 对图片进行池化操作
maxpool_output = maxpool_layer(input_image)
spectral_pool_output = spectral_pool_layer(input_image)

# 显示结果
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# 显示原始图片
input_np = input_image.squeeze().permute(1, 2, 0).numpy()
axes[0].imshow(input_np)
axes[0].set_title("Original Image")
axes[0].axis('off')

# 显示经过MaxPool2d和Hartley Spectral Pooling的结果图
maxpool_np = maxpool_output.squeeze().permute(1, 2, 0).detach().numpy()
spectral_pool_np = spectral_pool_output.squeeze().permute(1, 2, 0).detach().numpy()


axes[1].imshow(maxpool_np)
axes[1].set_title("MaxPool2d Output")#(scale_factor=(2, 2))
axes[1].axis('off')

axes[2].imshow(spectral_pool_np)
axes[2].set_title("Spectral Pooling Output")#(scale_factor=(2, 2))
axes[2].axis('off')

plt.tight_layout()
plt.show()











# # 定义不同的池化层
# maxpool_layers = [torch.nn.MaxPool2d(kernel_size=2),
#                   torch.nn.MaxPool2d(kernel_size=4),
#                   torch.nn.MaxPool2d(kernel_size=8),
#                   torch.nn.MaxPool2d(kernel_size=16)]
#
# spectral_pool_layers = [SpectralPool2d(scale_factor=(2, 2)),
#                         SpectralPool2d(scale_factor=(4, 4)),
#                         SpectralPool2d(scale_factor=(8, 8)),
#                         SpectralPool2d(scale_factor=(16, 16))]
#
# # 对图片进行不同的池化操作
# maxpool_outputs = [layer(input_image) for layer in maxpool_layers]
# spectral_pool_outputs = [layer(input_image) for layer in spectral_pool_layers]
#
# # 显示结果
# fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 20))
#
# # 显示原始图片
# input_np = input_image.squeeze().permute(1, 2, 0).numpy()
# axes[0].imshow(input_np)
# axes[0].set_title("Original Image")
# axes[0].axis('off')
#
# # 显示不同缩放因子下的结果图
# for i in range(len(maxpool_outputs)):
#     maxpool_np = maxpool_outputs[i].squeeze().permute(1, 2, 0).detach().numpy()
#     spectral_pool_np = spectral_pool_outputs[i].squeeze().permute(1, 2, 0).detach().numpy()
#
#     axes[i+1].imshow(np.hstack((maxpool_np, spectral_pool_np)))
#     axes[i+1].set_title(f"MaxPool2d vs Spectral Pooling (scale_factor={maxpool_layers[i].kernel_size})")
#     axes[i+1].axis('off')
#
# plt.tight_layout()
# plt.show()


