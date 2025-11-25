import cv2
import os

# 设置读取图像的文件夹路径
image_folder = '/media/vincent/HDD-01/queen_lightweight/output/dynerf/flame_salmon/spiral'
# 设置保存视频的文件名
video_filename = '/media/vincent/HDD-01/queen_lightweight/output/dynerf/flame_salmon/spiral/output_video.mp4'

# 获取所有PNG文件的列表
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# 确保文件名按照顺序排列
images.sort()

# 获取第一张图片以确定视频的宽度和高度
first_image_path = os.path.join(image_folder, images[0])
first_image = cv2.imread(first_image_path)
height, width, layers = first_image.shape

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用XVID编码
video = cv2.VideoWriter(video_filename, fourcc, 30, (width, height))  # 30帧每秒

# 遍历所有图像并写入视频
for image in images:
    image_path = os.path.join(image_folder, image)
    video.write(cv2.imread(image_path))

# 释放视频写入对象
video.release()
cv2.destroyAllWindows()

print(f'视频已保存为: {video_filename}')
