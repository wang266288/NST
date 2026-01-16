import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
import datetime
from gatysstyle import MultiStyleMultiContentTransfer

def load_image(image_path, size=512, device='cpu'):
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

def denormalize(tensor):
    """反标准化图像"""
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return tensor * std + mean

def imshow(tensor, title=None, figsize=(8, 8)):
    """显示图像"""
    image = tensor.cpu().clone()
    
    # 反标准化
    image = denormalize(image)
    
    image = image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if title:
        plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

def save_image(tensor, filename):
    """保存图像"""
    image = tensor.cpu().clone()
    
    # 反标准化
    image = denormalize(image)
    
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image.detach())
    image.save(filename)
    print(f"图像已保存: {filename}")
    return filename

def plot_loss_curves(loss_history, title_prefix, save_dir):
    """绘制损失曲线"""
    if not loss_history:
        return None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 根据损失类型确定子图数量
    loss_types = list(loss_history.keys())
    n_plots = len(loss_types)
    
    fig = plt.figure(figsize=(5 * n_plots, 4))
    
    for i, loss_type in enumerate(loss_types, 1):
        plt.subplot(1, n_plots, i)
        plt.plot(loss_history[loss_type])
        plt.title(f'{loss_type.replace("_", " ").title()}')
        plt.xlabel('Iterations')
        plt.ylabel('Loss Value')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title_prefix} Loss Curves', fontsize=16)
    plt.tight_layout()
    
    # 保存图像
    loss_curve_path = os.path.join(save_dir, f"loss_curves_{timestamp}.png")
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return loss_curve_path

def compare_results(results_dict, save_dir=None):
    """不同方法对比"""
    n_methods = len(results_dict)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5 * (n_methods + 1), 10))
    
    # 获取内容图像用于显示
    content_img = list(results_dict.values())[0]['content_img']
    style_img = list(results_dict.values())[0]['style_img']
    
    # 原始图像
    content_display = denormalize(content_img).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    content_display = np.clip(content_display, 0, 1)
    
    style_display = denormalize(style_img).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    style_display = np.clip(style_display, 0, 1)
    
    # 显示内容图像和风格图像
    axes[0, 0].imshow(content_display)
    axes[0, 0].set_title('Content Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(style_display)
    axes[1, 0].set_title('Style Image', fontsize=12)
    axes[1, 0].axis('off')
    
    # 显示每种方法的结果
    for idx, (method_name, result) in enumerate(results_dict.items(), 1):
        generated_img = result['generated_img']
        generated_display = denormalize(generated_img).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        generated_display = np.clip(generated_display, 0, 1)
        
        axes[0, idx].imshow(generated_display)
        axes[0, idx].set_title(f'{method_name}\nGenerated', fontsize=12)
        axes[0, idx].axis('off')
        
        # 显示损失曲线
        if 'loss_history' in result and 'total_loss' in result['loss_history']:
            axes[1, idx].plot(result['loss_history']['total_loss'][:100])
            axes[1, idx].set_title(f'{method_name}\nLoss Curve', fontsize=10)
            axes[1, idx].set_xlabel('Iterations')
            axes[1, idx].set_ylabel('Loss')
            axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(save_dir, f"comparison_{timestamp}.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存: {comparison_path}")
        return comparison_path
    
    plt.show()
    return None

def create_style_transfer_model(transfer_type, device, **kwargs):
    """创建风格迁移模型"""
    if transfer_type.lower() == 'gatysstyle':
        from gatysstyle import GatysStyleTransfer
        model = GatysStyleTransfer(device=device)
        if 'content_weight' in kwargs or 'style_weight' in kwargs:
            model.set_weights(
                content_weight=kwargs.get('content_weight', 1.0),
                style_weight=kwargs.get('style_weight', 1e4)
            )
        return model

    elif transfer_type.lower() == 'lapstyle':
        from lapstyle import LapStyleTransfer
        model = LapStyleTransfer(device=device)
        if any(k in kwargs for k in ['content_weight', 'style_weight', 'lap_weight']):
            model.set_weights(
                content_weight=kwargs.get('content_weight', 1.0),
                style_weight=kwargs.get('style_weight', 1e4),
                lap_weight=kwargs.get('lap_weight', 0.5e3)
            )
        return model

    elif transfer_type.lower() == 'multigatysstyle':
        model = MultiStyleMultiContentTransfer(device=device)
        if 'content_weights' in kwargs or 'style_weights' in kwargs:
            model.set_multi_weights(
                content_weights=kwargs.get('content_weights', None),
                style_weights=kwargs.get('style_weights', None)
            )
        return model

    else:
        raise ValueError(f"未知的风格迁移类型: {transfer_type}")

def run_style_transfer(args):
    """运行风格迁移"""
    print("=" * 60)
    print(f"{args.transfer.upper()} 风格迁移")
    print("=" * 60)

    # 设置设备和路径
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"使用设备: {device}")

    # 检查是否为多图像模式
    if args.content_images and args.style_images:
        content_paths = [os.path.join(args.content_dir, img) for img in args.content_images]
        style_paths = [os.path.join(args.style_dir, img) for img in args.style_images]

        content_imgs = [load_image(path, size=args.size, device=device) for path in content_paths]
        style_imgs = [load_image(path, size=args.size, device=device) for path in style_paths]

        # 创建风格迁移模型
        print(f"初始化{args.transfer}风格迁移模型...")
        model = create_style_transfer_model(
            args.transfer,
            device,
            content_weights=args.content_weights,
            style_weights=args.style_weights
        )

        # 执行风格迁移
        print("开始风格迁移...")
        generated_img, loss_history = model.transfer_multi_style_multi_content(
            content_imgs, style_imgs,
            iterations=args.iterations,
            lr=args.lr
        )

        # 保存结果
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output, args.transfer)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{args.transfer}_result_{timestamp}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        save_image(generated_img, output_path)

        # 显示生成图像
        if args.show_images:
            imshow(generated_img, title="Generated Image")

        # 绘制损失曲线
        if args.plot_loss:
            plot_loss_curves(loss_history, title_prefix=args.transfer, save_dir=output_dir)

        print("=" * 60)
        print(f"{args.transfer} 风格迁移完成！")
        print(f"结果保存到: {output_path}")
        print("=" * 60)

        return {
            'transfer_type': args.transfer,
            'generated_img': generated_img,
            'content_imgs': content_imgs,
            'style_imgs': style_imgs,
            'loss_history': loss_history,
            'output_path': output_path
        }

    else:
        # 单图像模式
        content_path = os.path.join(args.content_dir, args.content)
        style_path = os.path.join(args.style_dir, args.style)

        content_img = load_image(content_path, size=args.size, device=device)
        style_img = load_image(style_path, size=args.size, device=device)

        # 创建风格迁移模型
        print(f"初始化{args.transfer}风格迁移模型...")
        model = create_style_transfer_model(
            args.transfer,
            device,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            lap_weight=args.lap_weight
        )

        # 执行风格迁移
        print("开始风格迁移...")
        generated_img, loss_history = model.transfer_style(
            content_img, style_img,
            iterations=args.iterations,
            lr=args.lr
        )

        # 保存结果
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output, args.transfer)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{args.transfer}_result_{timestamp}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        save_image(generated_img, output_path)

        # 显示生成图像
        if args.show_images:
            imshow(generated_img, title="Generated Image")

        # 绘制损失曲线
        if args.plot_loss:
            plot_loss_curves(loss_history, title_prefix=args.transfer, save_dir=output_dir)

        print("=" * 60)
        print(f"{args.transfer} 风格迁移完成！")
        print(f"结果保存到: {output_path}")
        print("=" * 60)

        return {
            'transfer_type': args.transfer,
            'generated_img': generated_img,
            'content_img': content_img,
            'style_img': style_img,
            'loss_history': loss_history,
            'output_path': output_path
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='神经风格迁移系统')

    # 风格迁移参数
    parser.add_argument('--transfer', type=str, choices=['gatysstyle', 'lapstyle', 'multigatysstyle'], default='gatysstyle',
                       help='风格迁移方法: gatysstyle, lapstyle, multigatysstyle')

    # 图像参数
    parser.add_argument('--content', type=str, default='content.jpg',
                       help='内容图像文件名')
    parser.add_argument('--style', type=str, default='style.jpg',
                       help='风格图像文件名')
    parser.add_argument('--content-images', type=str, nargs='+',
                       help='多内容图像文件名列表')
    parser.add_argument('--style-images', type=str, nargs='+',
                       help='多风格图像文件名列表')
    parser.add_argument('--content-dir', type=str, default='examples/content',
                       help='内容图像目录')
    parser.add_argument('--style-dir', type=str, default='examples/style',
                       help='风格图像目录')

    # 超参数
    parser.add_argument('--size', type=int, default=256,
                       help='图像大小')
    parser.add_argument('--iterations', type=int, default=200,
                       help='迭代次数')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='学习率')

    # 损失权重
    parser.add_argument('--content-weight', type=float, default=1.0,
                       help='内容损失权重')
    parser.add_argument('--style-weight', type=float, default=1e4,
                       help='风格损失权重')
    parser.add_argument('--lap-weight', type=float, default=0.5e3,
                       help='拉普拉斯损失权重')
    parser.add_argument('--content-weights', type=float, nargs='+',
                       help='多内容损失权重列表（仅多内容方法使用）')
    parser.add_argument('--style-weights', type=float, nargs='+',
                       help='多风格损失权重列表（仅多风格方法使用）')

    # 输出选项
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--plot-loss', action='store_true',
                       help='绘制损失曲线')
    parser.add_argument('--show-images', action='store_true',
                       help='显示图像')
    parser.add_argument('--cpu', action='store_true',
                       help='强制使用CPU')

    # 对比模式
    parser.add_argument('--compare', action='store_true',
                       help='对比多种方法')
    parser.add_argument('--compare-methods', type=str, nargs='+',
                       default=['gatysstyle', 'lapstyle'],
                       help='要对比的方法列表')

    args = parser.parse_args()

    # 启用对比模式
    if args.compare:
        print("=" * 60)
        print("多方法对比模式")
        print("=" * 60)
        
        results = {}
        
        for method in args.compare_methods:
            print(f"\n运行 {method}...")
            # 临时修改参数
            args_copy = argparse.Namespace(**vars(args))
            args_copy.transfer = method
            
            # 运行风格迁移
            result = run_style_transfer(args_copy)
            results[method] = result
        
        # 生成对比图
        print("\n生成对比图...")
        comparison_dir = os.path.join(args.output, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        compare_results({
            method: {
                'generated_img': results[method]['generated_img'],
                'content_img': results[method]['content_img'],
                'style_img': results[method]['style_img'],
                'loss_history': results[method]['loss_history']
            }
            for method in results
        }, comparison_dir)
        
        print("\n" + "=" * 60)
        print("对比完成！")
        print("=" * 60)
        
    else:
        # 单一方法模式
        run_style_transfer(args)

if __name__ == "__main__":
    main()
