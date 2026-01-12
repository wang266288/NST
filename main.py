import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

class GatysStyleTransfer:
    """Gatys风格迁移实现"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        self.vgg = models.vgg19(weights='IMAGENET1K_V1').features.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.content_layers = ['relu4_2']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        self.layer_indices = {
            'relu1_1': 1,
            'relu2_1': 6,
            'relu3_1': 11,
            'relu4_1': 20,
            'relu4_2': 22,
            'relu5_1': 29
        }
        
        # 损失权重
        self.content_weight = 1
        self.style_weight = 1e6
    
    def get_features(self, x):
        features = {}
        
        # 获取特征
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            for layer_name, layer_idx in self.layer_indices.items():
                if i == layer_idx:
                    features[layer_name] = x
        
        return features
    
    def gram_matrix(self, x):
        batch_size, channels, h, w = x.size()
        features = x.view(batch_size, channels, h * w)
        
        gram = torch.bmm(features, features.transpose(1, 2))
        
        return gram.div(batch_size * channels * h * w)
    
    def transfer_style(self, content_img, style_img, iterations=300, lr=0.1):
        """执行风格迁移"""
        input_img = content_img.clone().requires_grad_(True)
        
        # 获取内容和风格特征
        print("提取内容图像特征...")
        content_features = self.get_features(content_img)
        print("提取风格图像特征...")
        style_features = self.get_features(style_img)
        
        # 计算风格特征的Gram矩阵
        style_grams = {}
        for layer in self.style_layers:
            style_grams[layer] = self.gram_matrix(style_features[layer])
        
        # 使用Adam优化器
        optimizer = torch.optim.Adam([input_img], lr=lr)
        
        print(f"开始风格迁移优化 ({iterations}次迭代)...")
        
        # 记录损失历史
        loss_history = []
        content_loss_history = []
        style_loss_history = []
        
        for i in range(iterations):
            # 前向传播
            input_features = self.get_features(input_img)
            
            # 计算内容损失
            content_loss = 0
            for layer in self.content_layers:
                content_loss += F.mse_loss(input_features[layer], content_features[layer])
            
            # 计算风格损失
            style_loss = 0
            for layer in self.style_layers:
                input_gram = self.gram_matrix(input_features[layer])
                style_gram = style_grams[layer]
                style_loss += F.mse_loss(input_gram, style_gram)
            
            # 总损失
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                input_img.data.clamp_(0, 1)
            
            # 记录损失
            loss_history.append(total_loss.item())
            content_loss_history.append(content_loss.item())
            style_loss_history.append(style_loss.item())
            
            if i % 50 == 0:
                print(f"迭代 {i}/{iterations}: 总损失 = {total_loss.item():.2e}, "
                      f"内容损失 = {content_loss.item():.2e}, "
                      f"风格损失 = {style_loss.item():.2e}")
        
        return input_img, {
            'total_loss': loss_history,
            'content_loss': content_loss_history,
            'style_loss': style_loss_history
        }

def load_image(image_path, size=512, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    
    image = image.detach().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    if title:
        plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

def save_image(tensor, filename):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    
    image = transforms.ToPILImage()(image.detach())
    image.save(filename)
    print(f"图像已保存: {filename}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Gatys风格迁移')
    parser.add_argument('--content', type=str, default='content.jpg', help='内容图像文件名')
    parser.add_argument('--style', type=str, default='style.jpg', help='风格图像文件名')
    parser.add_argument('--size', type=int, default=256, help='图像大小')
    parser.add_argument('--iterations', type=int, default=100, help='迭代次数')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')
    parser.add_argument('--output', type=str, default='output', help='输出目录')

    parser.add_argument('--content-dir', type=str, default='examples/content', help='内容图像目录')
    parser.add_argument('--style-dir', type=str, default='examples/style', help='风格图像目录')

    args = parser.parse_args()
    content_path = os.path.join(args.content_dir, args.content)
    style_path = os.path.join(args.style_dir, args.style)
    
    print("=" * 60)
    print("GATYS风格迁移")
    print("=" * 60)

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 加载图像
        print(f"加载内容图像: {args.content}")
        content_img = load_image(content_path, size=args.size, device=device)
        
        print(f"加载风格图像: {args.style}")
        style_img = load_image(style_path, size=args.size, device=device)
    except Exception as e:
        print(f"加载图像失败: {e}")
        print("请确保图像文件存在且格式正确。")
        return
    
    # 显示原始图像
    print("显示原始图像...")
    imshow(content_img, "content image")
    imshow(style_img, "style image")
    
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化模型
    print("初始化风格迁移模型...")
    model = GatysStyleTransfer(device=device)
    
    # 执行风格迁移
    print("开始风格迁移...")
    generated_img, loss_history = model.transfer_style(
        content_img, style_img, 
        iterations=args.iterations, 
        lr=args.lr
    )
    
    # 显示结果
    print("显示结果图像...")
    imshow(generated_img, "generated image")
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f"result_{timestamp}.jpg")
    save_image(generated_img, output_path)
    
    # 绘制损失曲线
    if loss_history['total_loss']:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(loss_history['total_loss'])
        plt.title('Total Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(loss_history['content_loss'])
        plt.title('Content Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(loss_history['style_loss'])
        plt.title('Style Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_curve_path = os.path.join(args.output, f"loss_curves_{timestamp}.png")
        plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
        print(f"损失曲线已保存: {loss_curve_path}")
        plt.show()
    
    print("=" * 60)
    print("风格迁移完成！")
    print(f"结果保存到: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
