import argparse
import os
import shutil
from pathlib import Path
from PIL import Image

def resize_image(image_path, output_path, resolution_ratio):
    """
    调整图片分辨率
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        resolution_ratio: 分辨率压缩比例 (如8表示压缩到1/8)
    """
    try:
        with Image.open(image_path) as img:
            # 计算新的分辨率
            width, height = img.size
            new_width = width // resolution_ratio
            new_height = height // resolution_ratio
            
            # 确保最小分辨率为1
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            print(f"调整分辨率: {width}x{height} -> {new_width}x{new_height}")
            
            # 调整图片大小，使用LANCZOS高质量下采样
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 保存图片，保持原始质量
            resized_img.save(output_path, quality=95)
            
            return True
            
    except Exception as e:
        print(f"错误: 处理图片 {image_path} 时发生异常: {e}")
        return False

def copy_other_files(scene_path, output_path, other_extensions=['.ply', '.npy']):
    """
    复制其他类型的文件（如.ply, .npy等）
    
    Args:
        scene_path: 原始场景路径
        output_path: 输出场景路径
        other_extensions: 需要复制的文件扩展名列表
    """
    copied_files = []
    for ext in other_extensions:
        for file_path in scene_path.glob(f'*{ext}'):
            if file_path.is_file():
                output_file = output_path / file_path.name
                shutil.copy2(file_path, output_file)
                copied_files.append(file_path.name)
                print(f"复制文件: {file_path.name}")
    
    return copied_files

def process_scene(scene_path, resolution_ratio):
    """
    处理整个场景
    
    Args:
        scene_path: 场景根目录路径
        resolution_ratio: 分辨率压缩比例
    """
    scene_path = Path(scene_path)
    
    # 检查输入目录是否存在
    if not scene_path.exists():
        print(f"错误: 场景目录不存在 - {scene_path}")
        return False
    
    # 创建输出目录
    scene_name = scene_path.name
    output_dir_name = f"{scene_name}_{resolution_ratio*2}"
    output_path = scene_path.parent / output_dir_name
    
    # 如果输出目录已存在，询问是否覆盖
    if output_path.exists():
        response = input(f"输出目录 {output_path} 已存在，是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            return False
        else:
            # 清空现有目录
            shutil.rmtree(output_path)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"创建输出目录: {output_path}")
    
    # 统计变量
    total_images = 0
    successful_images = 0
    
    # 遍历所有相机目录
    camera_dirs = [d for d in scene_path.iterdir() if d.is_dir() and d.name.startswith('cam')]
    
    if not camera_dirs:
        print(f"警告: 在 {scene_path} 中未找到相机目录 (cam01, cam02, ...)")
    
    for camera_dir in sorted(camera_dirs):
        camera_name = camera_dir.name
        images_dir = camera_dir / 'images'
        
        # 检查images目录是否存在
        if not images_dir.exists():
            print(f"警告: {images_dir} 不存在，跳过")
            continue
        
        # 创建输出目录结构
        output_camera_dir = output_path / camera_name / 'images'
        output_camera_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n处理相机: {camera_name}")
        
        # 处理该相机下的所有图片
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
        
        for image_file in sorted(image_files):
            total_images += 1
            output_image_path = output_camera_dir / image_file.name
            
            if resize_image(image_file, output_image_path, resolution_ratio):
                successful_images += 1
            else:
                print(f"失败: {image_file}")
    
    # 复制其他文件（.ply, .npy等）
    print(f"\n复制其他文件...")
    copied_files = copy_other_files(scene_path, output_path)
    
    # 输出统计信息
    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"场景: {scene_name}")
    print(f"分辨率压缩比例: 1/{resolution_ratio}")
    print(f"图片处理: {successful_images}/{total_images} 成功")
    print(f"复制的其他文件: {len(copied_files)} 个")
    if copied_files:
        print(f"文件列表: {', '.join(copied_files)}")
    print(f"输出目录: {output_path}")
    print(f"{'='*50}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='处理场景图片分辨率并复制相关文件')
    parser.add_argument('--scene_path', '-s', type=str, required=True,
                       help='场景目录路径')
    parser.add_argument('--resolution', '-r', type=int, required=True,
                       help='分辨率压缩比例 (如8表示压缩到1/8分辨率)')
    
    args = parser.parse_args()
    
    # 验证分辨率参数
    if args.resolution <= 0:
        print("错误: 分辨率比例必须为正整数")
        return
    
    print("开始处理场景...")
    success = process_scene(args.scene_path, args.resolution)
    
    if success:
        print("\n操作成功完成!")
    else:
        print("\n操作失败!")
        exit(1)

if __name__ == "__main__":
    main()