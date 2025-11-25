import os
import shutil
from pathlib import Path
import argparse

def reorganize_image_structure(scene_path):

    scene_path = Path(scene_path)
    scene_name = scene_path.name
    
    print(f"开始重组场景: {scene_name}")
    
    # 获取所有帧文件夹
    frame_folders = sorted([f for f in scene_path.iterdir() if f.is_dir() and f.name.startswith('frame')])
    print(f"找到 {len(frame_folders)} 个帧文件夹")
    
    if not frame_folders:
        print("未找到帧文件夹，请检查路径")
        return
    
    # 获取相机列表（从第一帧中获取）
    first_frame_images = list(frame_folders[0].glob("images_4/*.png"))
    camera_names = sorted(list(set(img.stem for img in first_frame_images)))  # cam00, cam01, etc.
    print(f"找到 {len(camera_names)} 个相机: {camera_names}")
    
    # 为每个相机创建目标文件夹
    for cam_name in camera_names:
        cam_dir = scene_path / cam_name / "images"
        cam_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有帧，重组图片
    for frame_idx, frame_folder in enumerate(frame_folders):
        images_dir = frame_folder / "images_4"
        
        if not images_dir.exists():
            print(f"警告: {images_dir} 不存在，跳过")
            continue
        
        # 遍历该帧的所有图片
        for img_path in images_dir.glob("*.png"):
            cam_name = img_path.stem  # cam00, cam01, etc.
            
            if cam_name not in camera_names:
                continue
            
            # 生成新的文件名（4位数字，从0000开始）
            new_filename = f"{frame_idx:04d}.png"
            target_path = scene_path / cam_name / "images" / new_filename
            
            # 复制文件（使用复制而不是移动，以防出错）
            shutil.copy2(img_path, target_path)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"已处理 {frame_idx + 1}/{len(frame_folders)} 帧")
    
    print(f"重组完成！新结构已创建在: {scene_path}")

def reorganize_with_verification(scene_path, delete_original=False):
    """
    带验证功能的重组，可选择删除原文件
    """
    scene_path = Path(scene_path)
    
    # 先执行重组
    reorganize_image_structure(scene_path)
    
    # 验证重组结果
    print("\n验证重组结果...")
    
    # 检查新结构
    camera_folders = sorted([f for f in scene_path.iterdir() if f.is_dir() and f.name.startswith('cam')])
    total_images = 0
    
    for cam_folder in camera_folders:
        images_dir = cam_folder / "images"
        if images_dir.exists():
            image_count = len(list(images_dir.glob("*.png")))
            total_images += image_count
            print(f"  {cam_folder.name}: {image_count} 张图片")
    
    # 计算原结构中的图片总数
    original_images = 0
    frame_folders = sorted([f for f in scene_path.iterdir() if f.is_dir() and f.name.startswith('frame')])
    for frame_folder in frame_folders:
        images_dir = frame_folder / "images_4"
        if images_dir.exists():
            original_images += len(list(images_dir.glob("*.png")))
    
    print(f"\n统计信息:")
    print(f"  原结构图片总数: {original_images}")
    print(f"  新结构图片总数: {total_images}")
    
    if original_images == total_images:
        print("✅ 验证成功: 图片数量匹配")
        
        if delete_original:
            # 删除原帧文件夹
            for frame_folder in frame_folders:
                shutil.rmtree(frame_folder)
            print("🗑️  原帧文件夹已删除")
        else:
            print("💾 原帧文件夹保留")
    else:
        print("❌ 验证失败: 图片数量不匹配，请检查")
    
    return original_images == total_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='重组图像目录结构')
    parser.add_argument('--scene', '-s', required=True, help='场景路径')
    args = parser.parse_args()
    
    reorganize_image_structure(args.scene)



    
