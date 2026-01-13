"""
继承自GatysStyleTransfer
"""
import torch
import torch.nn.functional as F
from gatysstyle import GatysStyleTransfer

class LapStyleTransfer(GatysStyleTransfer):
    """LapStyle风格迁移实现"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        
        # 拉普拉斯损失权重
        self.lap_weight = 0.5e3
        
        # 定义拉普拉斯滤波器核
        self.laplacian_kernel = self.create_laplacian_kernel().to(device)
    
    def create_laplacian_kernel(self):
        """创建拉普拉斯滤波器核"""
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32)
        
        # 扩展为3通道
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        kernel = kernel.expand(3, 1, 3, 3)  # [3, 1, 3, 3]
        kernel = kernel.contiguous()
        
        return kernel
    
    def laplacian_loss(self, img1, img2):
        """计算拉普拉斯损失"""
        # 确保输入在[0, 1]范围内
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        
        # 应用拉普拉斯滤波器
        lap1 = F.conv2d(img1, self.laplacian_kernel, padding=1, groups=3)
        lap2 = F.conv2d(img2, self.laplacian_kernel, padding=1, groups=3)
        
        # 计算均方误差
        loss = F.mse_loss(lap1, lap2)
        
        return loss
    
    def transfer_style(self, content_img, style_img, iterations=300, lr=0.1):
        """执行风格迁移（包含拉普拉斯损失）"""
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
        print(f"损失权重: 内容={self.content_weight}, 风格={self.style_weight}, 拉普拉斯={self.lap_weight}")
        
        # 记录损失历史
        loss_history = []
        content_loss_history = []
        style_loss_history = []
        lap_loss_history = []

        # 动态权重调整
        content_weight = self.content_weight
        style_weight = self.style_weight
        lap_weight = self.lap_weight
        
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
            
            # 计算拉普拉斯损失
            lap_loss = self.laplacian_loss(input_img, content_img)
            
            # 根据迭代阶段调整权重
            progress = i / iterations
            
            # 早期：强调风格损失
            if progress < 0.3:
                effective_style_weight = style_weight * 1.5
                effective_lap_weight = lap_weight * 0.5
            # 中期：平衡所有损失
            elif progress < 0.7:
                effective_style_weight = style_weight * 1.2
                effective_lap_weight = lap_weight * 0.8
            # 后期：强调内容和拉普拉斯损失
            else:
                effective_style_weight = style_weight * 0.8
                effective_lap_weight = lap_weight * 1.2
            
            # 计算总损失
            total_loss = (content_weight * content_loss + 
                        effective_style_weight * style_loss + 
                        effective_lap_weight * lap_loss)
            
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
            lap_loss_history.append(lap_loss.item())
            
            if i % 50 == 0:
                print(f"迭代 {i}/{iterations}: 总损失 = {total_loss.item():.2e}, "
                      f"内容损失 = {content_loss.item():.2e}, "
                      f"风格损失 = {style_loss.item():.2e}, "
                      f"拉普拉斯损失 = {lap_loss.item():.2e}")
        
        return input_img, {
            'total_loss': loss_history,
            'content_loss': content_loss_history,
            'style_loss': style_loss_history,
            'lap_loss': lap_loss_history
        }
    
    def set_weights(self, content_weight=1, style_weight=1e6, lap_weight=0.5e3):
        """设置损失权重"""
        super().set_weights(content_weight, style_weight)
        self.lap_weight = lap_weight
        return self
