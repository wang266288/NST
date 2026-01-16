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
        self.style_weight = 1e4
    
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
        """执行风格迁移，使用随机初始化"""
        # 随机初始化输入图像
        input_img = torch.rand_like(content_img, requires_grad=True)

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
    
    def set_weights(self, content_weight=1, style_weight=1e4):
        """设置损失权重"""
        self.content_weight = content_weight
        self.style_weight = style_weight
        return self


class MultiStyleMultiContentTransfer(GatysStyleTransfer):
    """通用多风格多内容风格迁移"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.content_weights = []  # 内容权重列表
        self.style_weights = []    # 风格权重列表
        
    def set_multi_weights(self, content_weights=None, style_weights=None):
        """设置多内容和多风格的权重"""
        if content_weights is not None:
            sum_content_weights = sum(content_weights)
            self.content_weights = [w/sum_content_weights for w in content_weights]
        if style_weights is not None:
            sum_style_weights = sum(style_weights)
            self.style_weights = [w/sum_style_weights for w in style_weights]
        return self
    
    def transfer_multi_style_multi_content(self, content_imgs, style_imgs, iterations=300, lr=0.1, initial_img=None):
        """执行多风格多内容迁移
        
        参数:
            content_imgs: 内容图像列表 [content_img1, content_img2, ...]
            style_imgs: 风格图像列表 [style_img1, style_img2, ...]
            iterations: 迭代次数
            lr: 学习率
            initial_img: 初始图像 (可选，默认为内容图像的平均)
        """
        # 设置默认权重
        if not self.content_weights:
            self.content_weights = [1/len(content_imgs)] * len(content_imgs)
        else:
            self.content_weights = [w * self.content_weight for w in self.content_weights]
        if not self.style_weights:
            self.style_weights = [1e4/len(style_imgs)] * len(style_imgs)
        else:
            self.style_weights = [w * self.style_weight for w in self.style_weights]
        # 创建初始图像，使用随机初始化
        if initial_img is None:
            input_img = torch.rand_like(content_imgs[0], requires_grad=True)
        else:
            input_img = initial_img.clone().requires_grad_(True)
        
        # 获取所有内容特征
        content_features_list = []
        print(f"提取 {len(content_imgs)} 个内容图像特征...")
        for idx, content_img in enumerate(content_imgs):
            print(f"  内容图像 {idx+1}...")
            content_features_list.append(self.get_features(content_img))
        
        # 获取所有风格特征
        style_grams_list = []
        print(f"提取 {len(style_imgs)} 个风格图像特征...")
        for idx, style_img in enumerate(style_imgs):
            print(f"  风格图像 {idx+1}...")
            style_features = self.get_features(style_img)
            style_grams = {}
            for layer in self.style_layers:
                style_grams[layer] = self.gram_matrix(style_features[layer])
            style_grams_list.append(style_grams)
        
        # 使用Adam优化器
        optimizer = torch.optim.Adam([input_img], lr=lr)
        
        print(f"开始多风格多内容迁移优化 ({iterations}次迭代)...")
        print(f"内容权重: {self.content_weights}")
        print(f"风格权重: {self.style_weights}")
        
        # 记录损失历史
        loss_history = []
        content_losses_history = [[] for _ in range(len(content_imgs))]
        style_losses_history = [[] for _ in range(len(style_imgs))]
        
        for i in range(iterations):
            # 前向传播
            input_features = self.get_features(input_img)
            
            # 计算多内容损失
            content_losses = []
            total_content_loss = 0
            
            for idx, content_features in enumerate(content_features_list):
                content_loss = 0
                for layer in self.content_layers:
                    content_loss += F.mse_loss(input_features[layer], content_features[layer])
                content_losses.append(content_loss)
                total_content_loss += self.content_weights[idx] * content_loss
                content_losses_history[idx].append(content_loss.item())
            
            # 计算多风格损失
            style_losses = []
            total_style_loss = 0
            
            for idx, style_grams in enumerate(style_grams_list):
                style_loss = 0
                for layer in self.style_layers:
                    input_gram = self.gram_matrix(input_features[layer])
                    style_gram = style_grams[layer]
                    style_loss += F.mse_loss(input_gram, style_gram)
                style_losses.append(style_loss)
                total_style_loss += self.style_weights[idx] * style_loss
                style_losses_history[idx].append(style_loss.item())
            
            # 总损失
            total_loss = total_content_loss + total_style_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 限制像素值在[0,1]范围内
            with torch.no_grad():
                input_img.data.clamp_(0, 1)
            
            # 记录总损失
            loss_history.append(total_loss.item())
            
            if i % 50 == 0:
                print(f"迭代 {i}/{iterations}: 总损失 = {total_loss.item():.2e}")
                for idx, loss in enumerate(content_losses):
                    print(f"  内容{idx+1}损失 = {loss.item():.2e}", end="  ")
                print()
                for idx, loss in enumerate(style_losses):
                    print(f"  风格{idx+1}损失 = {loss.item():.2e}", end="  ")
                print()
        
        # 创建损失字典
        loss_dict = {
            'total_loss': loss_history,
        }
        
        for idx in range(len(content_imgs)):
            loss_dict[f'content{idx+1}_loss'] = content_losses_history[idx]
        for idx in range(len(style_imgs)):
            loss_dict[f'style{idx+1}_loss'] = style_losses_history[idx]
        
        return input_img, loss_dict

