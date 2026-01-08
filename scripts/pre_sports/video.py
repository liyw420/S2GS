import os
import shutil
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  
import argparse

def copy_image_task(image_file, frame_dir, cam_id):
    target_filename = f"cam{cam_id:02d}.png"
    target_file = frame_dir / target_filename
    shutil.copy2(image_file, target_file)
    return (frame_dir, target_filename)

def reorganize_images(source_dir, target_dir, max_workers=8):
    """
    将图像从相机分组重新组织为帧分组（多线程加速版，带进度条显示）
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    images_dir = Path(source_dir) / "images"
    if not images_dir.exists():
        print(f"错误：找不到图像目录 {images_dir}")
        return

    image_files = list(images_dir.glob("*.png"))
    if not image_files:
        print("错误：没有找到PNG图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件")
    pattern = re.compile(r'cam(\d+)_(\d+)\.png')
    frame_dict = {}

    for image_file in image_files:
        match = pattern.match(image_file.name)
        if match:
            cam_id = int(match.group(1))
            frame_id = int(match.group(2))
            if frame_id not in frame_dict:
                frame_dict[frame_id] = {}
            frame_dict[frame_id][cam_id] = image_file

    total_frames = len(frame_dict)
    print(f"发现 {total_frames} 个帧")

    # 先批量创建所有帧目录
    for frame_id in frame_dict:
        frame_dir_name = f"frame{frame_id + 1:06d}"
        frame_dir = target_path / frame_dir_name
        frame_dir.mkdir(exist_ok=True)

    # 多线程复制图片，并用tqdm显示进度条
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for frame_id, cameras in frame_dict.items():
            frame_dir_name = f"frame{frame_id + 1:06d}"
            frame_dir = target_path / frame_dir_name
            for cam_id, image_file in cameras.items():
                tasks.append(executor.submit(copy_image_task, image_file, frame_dir, cam_id))

        # tqdm进度条
        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="复制图片进度"):
            pass

    print(f"\n重组完成！")
    print(f"总帧数: {total_frames}")
    print(f"目标目录: {target_dir}")

def main():
    parser = argparse.ArgumentParser(description='重组图像文件')
    parser.add_argument('--source', '-s', required=True, help='源目录')
    parser.add_argument('--target', '-t', required=True, help='目标目录') 
    
    args = parser.parse_args()
    reorganize_images(args.source, args.target, max_workers=16)

if __name__ == "__main__":
    main()