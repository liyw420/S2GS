import open3d as o3d
import sys

def process_ply_file(input_file, output_file):
    # 读取输入的ply文件
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Total points: {len(pcd.points)}")
    
    # 设置目标点数（最多200000点）
    target_points = min(200000, len(pcd.points))
    
    # # 使用最远点采样
    # if len(pcd.points) > target_points:
    #     # 方法1: 使用sample_points_farthest_point_sampling函数
    #     # 首先需要将点云转换为Tensor形式
    #     pcd_tensor = o3d.t.geometry.PointCloud.from_legacy(pcd)
        
    #     # 执行最远点采样
    #     sampled_pcd_tensor = pcd_tensor.farthest_point_down_sample(target_points)
        
    #     # 转换回Legacy格式
    #     pcd = sampled_pcd_tensor.to_legacy()
        
    #     print(f"Farthest point sampled points: {len(pcd.points)}")
    # else:
    #     print(f"Point cloud already has {len(pcd.points)} points, no need to downsample.")

    # 使用均匀采样
    # 使用均匀下采样
    if len(pcd.points) > target_points:
        # 方法：使用 uniform_down_sample 函数
        # 计算采样间隔：每 K 个点中取1个
        every_k_points = max(int(len(pcd.points)/100000), len(pcd.points) // target_points) # 确保 K 至少为 1
        
        # 执行均匀下采样
        downsampled_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
        
        # 由于均匀采样的点数可能不完全等于 target_points，我们确保不超过目标点数
        # 如果采样后点数仍多于目标，可进行二次随机采样（或其他处理），但通常均匀采样的结果已接近目标
        print(f"Uniform downsampled points: {len(downsampled_pcd.points)} (target was {target_points}, every_k_points={every_k_points})")
        pcd = downsampled_pcd
    else:
        print(f"Point cloud already has {len(pcd.points)} points, no need to downsample.")   

    # 将结果保存到输出的路径中
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Saved downsampled point cloud to {output_file}")

# 使用函数
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.ply output.ply")
        sys.exit(1)
    
    process_ply_file(sys.argv[1], sys.argv[2])