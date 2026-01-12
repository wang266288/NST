# quick_test.py - 快速测试脚本
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 创建简单的测试图像
def create_simple_images():
    """创建简单的测试图像"""
    # 创建内容图像（渐变）
    content = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        content[:, i, 0] = i  # 红色通道渐变
        content[i, :, 1] = i  # 绿色通道渐变
    
    # 创建风格图像（条纹）
    style = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 32):
        style[:, i:i+16, 0] = 255  # 红色条纹
        style[:, i+16:i+32, 2] = 255  # 蓝色条纹
    
    return content, style

# 加载VGG19并测试特征提取
def test_vgg_features():
    """测试VGG19特征提取"""
    print("测试VGG19特征提取...")
    
    # 创建测试图像
    content_array, style_array = create_simple_images()
    
    # 显示测试图像
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(content_array)
    axes[0].set_title('内容图像')
    axes[0].axis('off')
    
    axes[1].imshow(style_array)
    axes[1].set_title('风格图像')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 转换为Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    content_tensor = transform(content_array).unsqueeze(0)
    style_tensor = transform(style_array).unsqueeze(0)
    
    print(f"内容图像形状: {content_tensor.shape}")
    print(f"风格图像形状: {style_tensor.shape}")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载VGG19
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    # 测试特征提取
    content_features = {}
    style_features = {}
    
    # 我们关心的层
    target_layers = [1, 6, 11, 20, 22, 29]  # relu1_1, relu2_1, relu3_1, relu4_1, relu4_2, relu5_1
    
    # 提取内容图像特征
    x = content_tensor.to(device)
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in target_layers:
            layer_name = {
                1: 'relu1_1',
                6: 'relu2_1',
                11: 'relu3_1',
                20: 'relu4_1',
                22: 'relu4_2',
                29: 'relu5_1'
            }[i]
            content_features[layer_name] = x
            print(f"内容图像 - {layer_name}: {x.shape}")
    
    # 提取风格图像特征
    x = style_tensor.to(device)
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in target_layers:
            layer_name = {
                1: 'relu1_1',
                6: 'relu2_1',
                11: 'relu3_1',
                20: 'relu4_1',
                22: 'relu4_2',
                29: 'relu5_1'
            }[i]
            style_features[layer_name] = x
            print(f"风格图像 - {layer_name}: {x.shape}")
    
    print("\n特征提取测试成功！")
    print("现在可以运行 main.py 进行完整的风格迁移测试。")

if __name__ == "__main__":
    test_vgg_features()
    
    print("\n" + "="*60)
    print("运行风格迁移命令:")
    print("python main.py --size 128 --iterations 50")
    print("="*60)