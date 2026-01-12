import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

class ImageProcessor:
    """图像处理工具类"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 图像预处理的变换（用于VGG网络输入）
        self.preprocess = transforms.Compose([
            transforms.Resize(512),  # 调整图像大小
            transforms.ToTensor(),    # 转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
        
        # 图像后处理的逆变换
        self.deprocess = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                              std=[1/0.229, 1/0.224, 1/0.225]),
        ])
    
    def load_image(self, image_path, size=None):
        """加载图像并进行预处理"""
        image = Image.open(image_path).convert('RGB')
        
        if size is not None:
            image = image.resize((size, size))
        
        # 添加批量维度
        image = self.preprocess(image).unsqueeze(0)
        return image.to(self.device)
    
    def save_image(self, tensor, filename, size=None):
        """保存Tensor为图像文件"""
        # 移除批量维度并移动到CPU
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        
        # 反标准化
        image = self.deprocess(image)
        
        # 分离梯度并转换为PIL图像
        image = transforms.ToPILImage()(image.detach())
        
        if size is not None:
            image = image.resize((size, size))
        
        image.save(filename)
        print(f"图像已保存到: {filename}")
    
    def imshow(self, tensor, title=None, size=None):
        """显示Tensor图像"""
        # 移除批量维度
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        
        # 反标准化
        image = self.deprocess(image)
        
        # 分离梯度并转换为numpy数组用于显示
        image = image.detach().numpy().transpose(1, 2, 0)
        
        # 确保值在[0,1]范围内
        image = np.clip(image, 0, 1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        if title:
            plt.title(title, fontsize=16)
        plt.axis('off')
        plt.show()
    
    def tensor_to_numpy(self, tensor):
        """将Tensor转换为numpy数组"""
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.deprocess(image)
        return image.detach().numpy().transpose(1, 2, 0)
    
    def create_initial_image(self, content_image, noise_ratio=0.6):
        """创建初始图像（内容图像和白噪声的混合）"""
        # 生成与内容图像相同大小的随机噪声
        noise = torch.randn_like(content_image) * 0.1
        
        # 混合内容图像和噪声
        initial_image = content_image * (1 - noise_ratio) + noise * noise_ratio
        initial_image.requires_grad_(True)
        
        return initial_image
    
    def compute_image_size(self, content_path, style_path, max_size=512):
        """计算合适的图像大小"""
        content_img = Image.open(content_path)
        style_img = Image.open(style_path)
        
        # 获取较小的维度
        content_min_dim = min(content_img.size)
        style_min_dim = min(style_img.size)
        
        # 取两者中较小的，但不超过max_size
        size = min(content_min_dim, style_min_dim, max_size)
        
        # 确保是偶数（某些模型需要）
        size = size - (size % 2) if size % 2 != 0 else size
        
        return size