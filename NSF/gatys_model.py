import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple

# 定义一个命名的元组来存储VGG网络的输出
VGGOutputs = namedtuple('VGGOutputs', ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1',
                                       'relu4_2'])

class VGG19Features(nn.Module):
    """提取VGG19特征的自定义模型"""
    
    def __init__(self, requires_grad=False):
        super(VGG19Features, self).__init__()
        
        # 加载预训练的VGG19模型
        vgg_pretrained = models.vgg19(pretrained=True).features
        
        # VGG19的层切片 - 修正索引
        # VGG19 features的层索引:
        # 0: conv1_1, 1: relu1_1, 2: conv1_2, 3: relu1_2, 4: maxpool
        # 5: conv2_1, 6: relu2_1, 7: conv2_2, 8: relu2_2, 9: maxpool
        # 10: conv3_1, 11: relu3_1, 12: conv3_2, 13: relu3_2, 14: conv3_3, 15: relu3_3, 16: conv3_4, 17: relu3_4, 18: maxpool
        # 19: conv4_1, 20: relu4_1, 21: conv4_2, 22: relu4_2, 23: conv4_3, 24: relu4_3, 25: conv4_4, 26: relu4_4, 27: maxpool
        # 28: conv5_1, 29: relu5_1, 30: conv5_2, 31: relu5_2, 32: conv5_3, 33: relu5_3, 34: conv5_4, 35: relu5_4, 36: maxpool
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()  # 到relu4_1
        self.slice5 = nn.Sequential()  # 到relu5_1
        self.slice_content = nn.Sequential()  # 到relu4_2
        
        # 第一段: 到relu1_1 (层0-1)
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained[x])
        
        # 第二段: 到relu2_1 (层5-6)
        for x in range(5, 7):
            self.slice2.add_module(str(x), vgg_pretrained[x])
        
        # 第三段: 到relu3_1 (层10-11)
        for x in range(10, 12):
            self.slice3.add_module(str(x), vgg_pretrained[x])
        
        # 第四段: 到relu4_1 (层19-20)
        for x in range(19, 21):
            self.slice4.add_module(str(x), vgg_pretrained[x])
        
        # 内容层: 到relu4_2 (层19-22)
        for x in range(19, 23):
            self.slice_content.add_module(str(x), vgg_pretrained[x])
        
        # 第五段: 到relu5_1 (层28-29)
        for x in range(28, 30):
            self.slice5.add_module(str(x), vgg_pretrained[x])
        
        # 冻结参数
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # 提取各个层的特征
        h = self.slice1(x)
        relu1_1 = h
        
        h = self.slice2(h)
        relu2_1 = h
        
        h = self.slice3(h)
        relu3_1 = h
        
        h = self.slice4(h)
        relu4_1 = h
        
        # 为内容层单独处理
        h_content = self.slice_content(x)  # 从原始输入开始
        relu4_2 = h_content
        
        h = self.slice5(h)
        relu5_1 = h
        
        return VGGOutputs(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1,
                          relu4_2)


class GatysStyleTransfer(nn.Module):
    """Gatys风格迁移模型"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(GatysStyleTransfer, self).__init__()
        self.device = device
        
        # 特征提取器
        self.feature_extractor = VGG19Features(requires_grad=False).to(device)
        
        # 风格和内容层配置（按照Gatys原始论文）
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.content_layers = ['relu4_2']
        
        # 风格层权重
        self.style_weights = {'relu1_1': 1.0,
                             'relu2_1': 0.8,
                             'relu3_1': 0.5,
                             'relu4_1': 0.3,
                             'relu5_1': 0.1}
        
        # 损失权重
        self.content_weight = 1e0
        self.style_weight = 1e3
    
    def compute_gram_matrix(self, x):
        """计算Gram矩阵"""
        batch_size, channels, height, width = x.size()
        
        # 重塑为二维矩阵
        features = x.view(batch_size * channels, height * width)
        
        # 计算Gram矩阵
        gram = torch.mm(features, features.t())
        
        # 归一化
        return gram.div(batch_size * channels * height * width)
    
    def extract_features(self, image):
        """提取图像特征 - 简化版本"""
        # 直接使用特征提取器
        outputs = self.feature_extractor(image)
        
        # 构建特征字典
        features = {
            'relu1_1': outputs.relu1_1,
            'relu2_1': outputs.relu2_1,
            'relu3_1': outputs.relu3_1,
            'relu4_1': outputs.relu4_1,
            'relu5_1': outputs.relu5_1,
            'relu4_2': outputs.relu4_2
        }
        
        return features
    
    def compute_content_loss(self, content_features, generated_features):
        """计算内容损失"""
        content_loss = 0
        for layer in self.content_layers:
            content_loss += F.mse_loss(generated_features[layer], 
                                      content_features[layer])
        return content_loss
    
    def compute_style_loss(self, style_features, generated_features):
        """计算风格损失"""
        style_loss = 0
        for layer in self.style_layers:
            # 计算Gram矩阵
            style_gram = self.compute_gram_matrix(style_features[layer])
            generated_gram = self.compute_gram_matrix(generated_features[layer])
            
            # 计算该层的风格损失
            layer_loss = F.mse_loss(generated_gram, style_gram)
            
            # 加权求和
            style_loss += self.style_weights[layer] * layer_loss
        
        return style_loss
    
    def compute_total_loss(self, content_image, style_image, generated_image):
        """计算总损失"""
        # 提取特征
        content_features = self.extract_features(content_image)
        style_features = self.extract_features(style_image)
        generated_features = self.extract_features(generated_image)
        
        # 计算各项损失
        content_loss = self.compute_content_loss(content_features, generated_features)
        style_loss = self.compute_style_loss(style_features, generated_features)
        
        # 总损失
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        return total_loss, content_loss, style_loss
    
    def transfer_style(self, content_image, style_image, 
                      num_iterations=300, 
                      learning_rate=1.0,
                      show_progress=True):
        """执行风格迁移"""
        
        # 初始化生成图像（使用内容图像作为起点）
        generated_image = content_image.clone().requires_grad_(True)
        
        # 使用L-BFGS优化器（论文推荐）
        optimizer = torch.optim.LBFGS([generated_image], lr=learning_rate, max_iter=20)
        
        # 记录损失历史
        loss_history = []
        content_loss_history = []
        style_loss_history = []
        
        # 迭代优化
        iteration = [0]
        def closure():
            optimizer.zero_grad()
            total_loss, content_loss, style_loss = self.compute_total_loss(
                content_image, style_image, generated_image
            )
            total_loss.backward()
            
            # 记录损失
            if iteration[0] % 10 == 0:
                loss_history.append(total_loss.item())
                content_loss_history.append(content_loss.item())
                style_loss_history.append(style_loss.item())
                
                if show_progress:
                    print(f"Iteration {iteration[0]}: Total Loss = {total_loss.item():.2e}, "
                          f"Content Loss = {content_loss.item():.2e}, "
                          f"Style Loss = {style_loss.item():.2e}")
            
            iteration[0] += 1
            return total_loss
        
        # 优化循环
        while iteration[0] < num_iterations:
            optimizer.step(closure)
            
            # 限制像素值范围
            with torch.no_grad():
                generated_image.data.clamp_(0, 1)
        
        # 返回结果和损失历史
        return generated_image, {
            'total_loss': loss_history,
            'content_loss': content_loss_history,
            'style_loss': style_loss_history
        }


# 简化的Gatys实现（如果上面的有问题，可以尝试这个简化版本）
class SimpleGatysStyleTransfer:
    """简化的Gatys风格迁移实现"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
        
        # 冻结所有参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 内容层和风格层
        self.content_layers = ['relu4_2']
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        # 层名称到索引的映射
        self.layer_names = {
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
        """获取指定层的特征"""
        features = {}
        for name, idx in self.layer_names.items():
            # 前向传播到指定层
            for i in range(idx + 1):
                x = self.vgg[i](x)
            features[name] = x
            # 重置x为原始输入以获取下一层特征
            x = features['relu1_1'] if name == 'relu1_1' else self.get_input_for_layer(name)
        return features
    
    def get_input_for_layer(self, layer_name):
        """获取指定层的输入（简化实现）"""
        # 这是一个简化的实现，实际使用时需要更复杂的逻辑
        pass
    
    def gram_matrix(self, x):
        """计算Gram矩阵"""
        batch_size, channels, h, w = x.size()
        features = x.view(batch_size * channels, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * h * w)
    
    def style_transfer_simple(self, content, style, iterations=300, lr=0.01):
        """简化的风格迁移"""
        # 生成初始图像
        input_img = content.clone().requires_grad_(True)
        
        # 获取内容和风格特征
        content_features = self.get_features(content)
        style_features = self.get_features(style)
        
        # 计算风格特征的Gram矩阵
        style_grams = {layer: self.gram_matrix(style_features[layer]) 
                      for layer in self.style_layers}
        
        # 优化器
        optimizer = torch.optim.Adam([input_img], lr=lr)
        
        print("开始优化...")
        for i in range(iterations):
            # 前向传播
            input_features = self.get_features(input_img)
            
            # 计算内容损失
            content_loss = 0
            for layer in self.content_layers:
                content_loss += torch.mean((input_features[layer] - content_features[layer])**2)
            
            # 计算风格损失
            style_loss = 0
            for layer in self.style_layers:
                input_gram = self.gram_matrix(input_features[layer])
                style_gram = style_grams[layer]
                style_loss += torch.mean((input_gram - style_gram)**2)
            
            # 总损失
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 限制像素值
            with torch.no_grad():
                input_img.data.clamp_(0, 1)
            
            if i % 50 == 0:
                print(f"Iteration {i}: Total Loss = {total_loss.item():.2e}")
        
        return input_img