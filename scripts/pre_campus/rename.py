import os
import re
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='重命名视频文件')
    parser.add_argument('--source', type=str, required=True, help='源目录路径')
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    prefix = source_dir.name  # 自动从路径获取前缀，如"fountain"
    
    # 获取所有匹配的文件并按数字排序
    files = []
    for f in source_dir.glob(f"{prefix}_*.mp4"):
        match = re.search(rf"{prefix}_(\d+)\.mp4", f.name)
        if match:
            files.append((int(match.group(1)), f))
    
    files.sort()  # 按数字排序
    
    # 重命名文件
    for i, (num, old_path) in enumerate(files):
        new_name = f"cam{i:02d}.mp4"
        new_path = source_dir / new_name
        old_path.rename(new_path)
        print(f"{old_path.name} -> {new_name}")
    
    print(f"重命名完成! 共处理 {len(files)} 个文件")

if __name__ == "__main__":
    main()