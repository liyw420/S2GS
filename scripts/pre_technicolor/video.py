import os
import shutil
from pathlib import Path
import argparse
from collections import defaultdict

def reorganize_by_camera(scene_path):
    """
    按相机编号重组图片路径结构
    
    原结构: scene/images/camXX_YYYY.png
    新结构: scene/camXX/images/YYYY.png
    """
    images_dir = Path(scene_path) / "images"
    
    # 检查原目录是否存在
    if not images_dir.exists():
        print(f"错误: 目录 {images_dir} 不存在")
        return
    
    # 创建新的scene目录
    scene_dir = Path(scene_path)
    
    # 遍历所有图片文件
    for image_file in images_dir.glob("cam*_*.png"):
        # 解析文件名
        filename = image_file.name
        try:
            # 提取相机编号和帧编号
            base_name = filename.replace(".png", "")
            parts = base_name.split("_")
            if len(parts) == 2:
                camera_id = parts[0]  # 如 cam00
                frame_num = parts[1]  # 如 0000
                
                # 构建新的目录结构
                camera_dir = scene_dir / camera_id / "images"
                camera_dir.mkdir(parents=True, exist_ok=True)
                
                # 构建目标文件路径
                new_filename = f"{frame_num}.png"
                target_path = camera_dir / new_filename
                
                # 复制文件到新位置
                shutil.copy2(image_file, target_path)
                print(f"已处理: {filename} -> {camera_id}/images/{new_filename}")
                
        except (ValueError, IndexError) as e:
            print(f"跳过无法解析的文件: {filename}, 错误: {e}")
    
    print("文件重组完成！")

def reorganize_by_camera_efficient(scene_path):
    """
    高效版本：使用字典先收集再批量处理
    """
    images_dir = Path(scene_path) / "images"
    scene_dir = Path(scene_path)
    
    if not images_dir.exists():
        print(f"错误: 目录 {images_dir} 不存在")
        return
    
    # 使用字典收集文件映射关系
    camera_files = defaultdict(list)
    
    # 收集所有文件信息
    for image_file in images_dir.glob("cam*_*.png"):
        filename = image_file.name
        try:
            base_name = filename.replace(".png", "")
            camera_id, frame_num = base_name.split("_")
            
            camera_files[camera_id].append((frame_num, image_file))
            
        except (ValueError, IndexError) as e:
            print(f"跳过无法解析的文件: {filename}, 错误: {e}")
    
    # 批量创建目录和复制文件
    for camera_id, frames in camera_files.items():
        camera_dir = scene_dir / camera_id / "images"
        camera_dir.mkdir(parents=True, exist_ok=True)
        
        for frame_num, source_path in frames:
            target_path = camera_dir / f"{frame_num}.png"
            shutil.copy2(source_path, target_path)
        
        print(f"创建 {camera_id}/images，包含 {len(frames)} 个帧")
    
    print(f"\n重组完成！共处理 {len(camera_files)} 个相机目录")

# 移动文件版本（会删除原文件）
def reorganize_by_camera_move(scene_path):
    """
    移动文件版本（会删除原文件）
    """
    images_dir = Path(scene_path) / "images"
    scene_dir = Path(scene_path)
    
    if not images_dir.exists():
        print(f"错误: 目录 {images_dir} 不存在")
        return
    
    for image_file in images_dir.glob("cam*_*.png"):
        filename = image_file.name
        try:
            base_name = filename.replace(".png", "")
            camera_id, frame_num = base_name.split("_")
            
            # 创建相机目录
            camera_dir = scene_dir / camera_id / "images"
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # 移动文件
            target_path = camera_dir / f"{frame_num}.png"
            shutil.move(str(image_file), str(target_path))
            print(f"已移动: {filename} -> {camera_id}/images/{frame_num}.png")
            
        except (ValueError, IndexError) as e:
            print(f"跳过无法解析的文件: {filename}, 错误: {e}")
    
    print("文件移动完成！")

def verify_structure(scene_path):
    """
    验证重组后的文件结构
    """
    scene_dir = Path(scene_path)
    
    # 检查相机目录
    camera_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("cam")])
    print(f"\n验证结果:")
    print(f"找到 {len(camera_dirs)} 个相机目录")
    
    for camera_dir in camera_dirs:
        images_dir = camera_dir / "images"
        if images_dir.exists():
            frame_files = sorted([f.name for f in images_dir.iterdir() if f.is_file()])
            print(f"{camera_dir.name}: {len(frame_files)} 个帧文件 - {frame_files[:3]}...")  # 显示前3个文件
        else:
            print(f"{camera_dir.name}: images目录不存在")


def main():
    parser = argparse.ArgumentParser(description='重组图像文件')
    parser.add_argument('--source', '-s', required=True, help='源目录')
    
    args = parser.parse_args()
    reorganize_by_camera(args.source)

if __name__ == "__main__":
    main()



# if __name__ == "__main__":
#     from collections import defaultdict
    
#     # 设置你的scene目录路径
#     scene_directory = "data/technicolor/birthday"  # 修改为你的实际路径
    
#     print("开始按相机重组文件结构...")
    
#     # 选择其中一种方法执行：
    
#     # 方法1: 复制文件（推荐，保留原文件）
#     reorganize_by_camera(scene_directory)
    
#     # 方法2: 高效复制版本
#     # reorganize_by_camera_efficient(scene_directory)
    
#     # 方法3: 移动文件（会删除原文件，谨慎使用）
#     # reorganize_by_camera_move(scene_directory)
    
#     # 验证结果
#     verify_structure(scene_directory)