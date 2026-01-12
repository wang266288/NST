import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GatysStyleTransfer:
    """Gatys风格迁移实现"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 加载预训练的VGG19模型
        self.vgg = models.vgg19(weights='IMAGENET1K_V1').features.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 定义内容和风格层
        self.content_layers = ['relu4_2']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        # VGG19层索引映射
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
        """获取VGG网络中间层特征"""
        features = {}
        
        # 获取特征
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            for layer_name, layer_idx in self.layer_indices.items():
                if i == layer_idx:
                    features[layer_name] = x
        
        return features
    
    def gram_matrix(self, x):
        """计算Gram矩阵"""
        batch_size, channels, h, w = x.size()
        features = x.view(batch_size, channels, h * w)
        
        # 计算Gram矩阵
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # 归一化
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
        print(f"损失权重: 内容={self.content_weight}, 风格={self.style_weight}")
        
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
            
            # 限制像素值在[0,1]范围内
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
    
    def set_weights(self, content_weight=1, style_weight=1e6):
        """设置损失权重"""
        self.content_weight = content_weight
        self.style_weight = style_weight
        return self
