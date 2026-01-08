import os
import shutil
import argparse

def reorganize_images(src_dir, dst_dir):

    ims_path = os.path.join(src_dir, 'images')

    # 创建目标 images 文件夹
    dst_images_path = os.path.join(dst_dir, 'images')
    os.makedirs(dst_images_path, exist_ok=True)

    # 遍历 ims 下的每个编号文件夹
    for cam_folder in os.listdir(ims_path):
        cam_path = os.path.join(ims_path, cam_folder)
        if not os.path.isdir(cam_path):
            continue
        cam_id = int(cam_folder)
        cam_prefix = f'cam{cam_id:02d}_'

        # 遍历图片
        for img_file in sorted(os.listdir(cam_path)):
            if not img_file.lower().endswith('.jpg'):
                continue
            img_idx = int(img_file.split('.')[0][-6:])  # 假设文件名为 ---000000.jpg
            if img_idx >= 300:
                continue  # 跳过大于等于300的图片
            new_img_name = f'{cam_prefix}{img_idx:04d}.png'
            src_img_path = os.path.join(cam_path, img_file)
            dst_img_path = os.path.join(dst_images_path, new_img_name)
            # 转换为 png 格式
            from PIL import Image
            with Image.open(src_img_path) as im:
                im.save(dst_img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorganize image dataset structure.')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--dst', type=str, required=True, help='Destination directory')
    args = parser.parse_args()

    reorganize_images(args.src, args.dst)
