import open3d as o3d
import sys

def process_ply_file(input_file, output_file):
    # 读取输入的ply文件
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Total points: {len(pcd.points)}")
    
    # 设置目标点数（最多40000点）
    target_points = min(40000, len(pcd.points))
    
    # 使用最远点采样
    if len(pcd.points) > target_points:
        # 方法1: 使用sample_points_farthest_point_sampling函数
        # 首先需要将点云转换为Tensor形式
        pcd_tensor = o3d.t.geometry.PointCloud.from_legacy(pcd)
        
        # 执行最远点采样
        sampled_pcd_tensor = pcd_tensor.farthest_point_down_sample(target_points)
        
        # 转换回Legacy格式
        pcd = sampled_pcd_tensor.to_legacy()
        
        print(f"Farthest point sampled points: {len(pcd.points)}")
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