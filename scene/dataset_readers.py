#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat,\
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getRT
import numpy as np
import torch
import json
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.pose_utils import get_spiral, get_spiral_immersive
from scene.gaussian_model import BasicPointCloud
from threading import Lock
import imagesize
from multiprocessing.pool import ThreadPool

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SequentialCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int
    znear: float
    zfar: float
    principle_point: tuple
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)

    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = None
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=100):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:    
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    raise NotImplementedError("Currently not implemented for blender format!")

def getPose(R, T, world_scale):
    pose = np.eye(4)
    pose[:3,3] = - np.dot(T[np.newaxis,:]*world_scale,R.T)[0]
    pose[:3,3] /= world_scale
    pose[:3,:3] = R
    pose[:3,0] = -pose[:3,0]
    pose[:3,:3] = -pose[:3,:3]

    pose_pre = np.eye(4)
    pose_pre[1, 1] *= -1
    pose_pre[2, 2] *= -1
    pose = pose_pre @ pose @ pose_pre

    return pose[:3,:]

def readImmersiveCameras(cam_extrinsics, cam_intrinsics, image_wh, world_scale=1.0):
    cam_infos = []
    poses = -np.ones((len(cam_extrinsics), 3, 4))
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        scale = intr.width/image_wh[0] # Scaled down image w.r.t camera intrinsics
        height = intr.height/scale
        width = intr.width/scale
        assert width == image_wh[0] and height == image_wh[1]

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]/scale
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]/scale
            focal_length_y = intr.params[1]/scale
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only datasets"+\
                   "(PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        cam_info = SequentialCameraInfo(
            uid=uid, R=R, T=T*world_scale, FovY=FovY, FovX=FovX,
            width=int(width), height=int(height),
            znear=.1, zfar=1000.,
            principle_point=(intr.params[2]/scale, intr.params[3]/scale)
        )
        cam_infos.append(cam_info)
        poses[uid-1] = getPose(R,T,world_scale)
        if uid == 1:
            focal = [focal_length_x, focal_length_y]
            principle_point  = (intr.params[2]/scale, intr.params[3]/scale)

    poses = poses[poses.sum(axis=1).sum(axis=1)!=-12]
    sys.stdout.write('\n')
    near_fars = np.array([0.25, 50])
    video_poses = get_spiral_immersive(poses, near_fars, N_views=300, rads_scale=0.5)
    video_cameras = getImmersiveVideoCameras(
        video_poses,
        image_wh,
        focal,
        world_scale=1.0,
        width=width,
        height=height,
        principle_point=principle_point
    )
    return cam_infos, video_cameras

def readImmersiveSceneInfo(path, test_indices, world_scale = 0.0):

    scale = 1.0
    image = Image.open(os.path.join(path,'camera_0001','images_scaled_2','0000.png'))
    image_wh = (
        int(image.width / scale),
        int(image.height / scale),
    )  

    ply_path = os.path.join(path, "colmap", "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "colmap", "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "colmap", "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
        if world_scale <= 0.0:
            pt = torch.tensor(pcd.points).cuda()
            dist = torch.sqrt(torch.sum((pt.unsqueeze(0)-pt.unsqueeze(1))**2,dim=-1))
            world_scale = 1000/dist.max().item()
        pcd = pcd._replace(points=pcd.points*world_scale)
    except:
        pcd = None

    try:
        cameras_extrinsic_file = os.path.join(path, "colmap", "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap", "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap", "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap", "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos_unsorted, video_cam_infos = readImmersiveCameras(cam_extrinsics=cam_extrinsics, 
                                              cam_intrinsics=cam_intrinsics, 
                                              image_wh=image_wh,
                                              world_scale=world_scale)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.uid)
    video_cam_infos = sorted(video_cam_infos.copy(), key = lambda x : x.uid)

    # cam_info.uid is 1-indexed but test_indices is 0-indexed
    train_cam_infos = [cam_info for cam_info in cam_infos if cam_info.uid-1 not in test_indices]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.uid-1 in test_indices]
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def getImmersiveVideoCameras(poses, image_wh, focal, world_scale, width, height, principle_point):
    cameras = []

    for idx, p in tqdm(enumerate(poses)):
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(focal[0], image_wh[0])
        FovY = focal2fov(focal[1], image_wh[1])
        cameras.append(SequentialCameraInfo(uid=idx, R=R, T=T*world_scale, FovY=FovY, FovX=FovX,
                                        width=int(width), height=int(height),
                                        znear=.1, zfar=1000., 
                                        principle_point=principle_point))
    return cameras


def getVideoCameras(poses, image_wh, focal, znear, zfar):
    cameras = []

    for idx, p in tqdm(enumerate(poses)):
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(focal[0], image_wh[0])
        FovY = focal2fov(focal[1], image_wh[1])
        cameras.append(SequentialCameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                                            width=image_wh[0], height=image_wh[1],
                                            znear=znear, zfar=zfar, principle_point=None))
    return cameras

def readCamerasFromPoseBounds(poses_path=None, image_wh=[0,0], N_video_views=300,poses_arr=None,idx_offset=0):

    if poses_path != None:
        poses_arr = np.load(poses_path)
    elif poses_arr is not None:
        poses_arr=poses_arr
    else:
        raise NotImplementedError("No poses path or poses array provided")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    near_fars = poses_arr[:, -2:]

    
    # poses_bounds can have image at original resolution, which might not match the existing images
    # therefore we will scale it correctly according to `image_wh`
    H, W, focal = poses[0, :, -1]
    scale = W / image_wh[0]
    focal = focal / scale
    focal = [focal, focal]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

    cameras = []
    for idx in range(poses.shape[0]):
        pose = np.array(poses[idx])
        R = pose[:3,:3]
        R = -R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(focal[0], image_wh[0])
        FovY = focal2fov(focal[1], image_wh[1])
        # Load image after
        cameras.append(SequentialCameraInfo(uid=idx+idx_offset, R=R, T=T, FovY=FovY, FovX=FovX, 
                                             width=image_wh[0], height=image_wh[1],
                                             znear=near_fars[idx][0], 
                                             zfar=near_fars[idx][1],
                                             principle_point=None))
        
    video_poses = get_spiral(poses, near_fars, N_views=N_video_views)
    video_cameras = getVideoCameras(video_poses, image_wh, focal,
                                    znear=np.median(near_fars[:,0]),
                                    zfar=np.median(near_fars[:,1]))

    return cameras, video_cameras
        
def readDynerfInfo(datadir, test_indices, N_video_views=300, verbose=False):
    
    downsample = 1.0
    image = Image.open(os.path.join(datadir,'cam01','images','0000.png'))
    image_wh = (
        int(image.width / downsample),
        int(image.height / downsample),
    )  

    cam_infos, video_cam_infos = readCamerasFromPoseBounds(os.path.join(datadir, "poses_bounds.npy"), image_wh, N_video_views=N_video_views)
    cam_infos = sorted(cam_infos.copy(), key = lambda x : x.uid)
    video_cam_infos = sorted(video_cam_infos.copy(), key = lambda x : x.uid)

    train_cam_infos = [cam_info for cam_info in cam_infos if cam_info.uid not in test_indices]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.uid in test_indices]
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # loading all the data follow hexplane format
    if "dynerf" or "meetroom" or "immersive" in datadir:
        ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    else:
        ply_path = os.path.join(datadir, "colmap/dense/workspace/fused.ply")
    pcd = fetchPly(ply_path)
    if verbose:
        print("readDynerfInfo(): acquired point cloud of shape ", pcd.points.shape)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path
                           )

    return scene_info

def setup_camera(id, width, height, cam_intrinsics, w2c, near=0.01, far=100):
    fx, fy, cx, cy = cam_intrinsics[0][0], cam_intrinsics[1][1], cam_intrinsics[0][2], cam_intrinsics[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    FovX = focal2fov(fx, cx)
    FovY = focal2fov(fy, cy)
    R,T = getRT(w2c)
    cam_info = SequentialCameraInfo(id, R, T, FovY, FovX, width, height)
    return cam_info

def readPanopticInfo(datadir):
    train_meta = json.load(open(os.path.join(datadir,"train_meta.json"), 'r'))
    # Take first time step camera params, assumed to be the same throughout
    width, height, cam_intrinsics, w2c = train_meta["w"], train_meta["h"], train_meta["k"][0], train_meta["w2c"][0]
    train_infos = []
    for id in cam_intrinsics:
        train_info = setup_camera(width, height, cam_intrinsics[id], w2c)
        train_infos.append(train_info)
    
    # Return placeholder scene info - this function needs proper implementation
    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_infos,
                           test_cameras=[],
                           video_cameras=[],
                           nerf_normalization={"translate": np.array([0,0,0]), "radius": 1.0},
                           ply_path="")
    return scene_info

def readTechnicolorInfo(datadir, test_indices, N_video_views=300, verbose=False):
    
    print("Reading Training Transforms")    # 对于dynerf，训练集参考transforms_train.json文件
    train_cam_infos = readCamerasFromTransforms(datadir, "transforms_train.json", white_background = False, extension = '.png', time_duration=[0, 10], frame_ratio=1, dataloader=True)
    print("Reading Test Transforms")        # 对于dynerf，测试集参考transforms_test.json文件
    test_cam_infos = readCamerasFromTransforms(datadir, "transforms_test.json", white_background = False, extension = '.png', time_duration=[0, 10], frame_ratio=1, dataloader=True)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # loading all the data follow hexplane format
    # if "dynerf" in datadir:
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    # else:
    #     ply_path = os.path.join(datadir, "colmap/dense/workspace/fused.ply")
    pcd = fetchPly(ply_path)
    if verbose:
        print("readDynerfInfo(): acquired point cloud of shape ", pcd.points.shape)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path
                           )

    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", time_duration=None, frame_ratio=1, dataloader=False):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        
    frames = contents["cam_info"]
    tbar = tqdm(range(len(frames)))
    processed_cameras = set()
    processed_lock = Lock()

    def frame_read_fn(idx_frame):
        frame = idx_frame[1]     
        bounds = np.array(frame['bounds'])              # 添加一组参数bounds,为图像的最大最小深度值，用于计算相机的视锥体
        camera_id = int(frame["file_path"].split('_')[0].split('cam')[-1])

        cam_name = os.path.join(path, frame["file_path"] + extension)
        
        with processed_lock:
            if camera_id in processed_cameras:
                return None
            processed_cameras.add(camera_id)
        # # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image_path = os.path.join(path, cam_name) 
        
        if not dataloader:
            with Image.open(image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            if norm_data[:, :, 3:4].min() < 1:
                arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=2)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")
            else:
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            width, height = image.size[0], image.size[1]
        else:
            image = np.empty(0)
            width, height = imagesize.get(image_path)
        
        if 'depth_path' in frame:
            depth_name = frame["depth_path"]
            if not extension in frame["depth_path"]:
                depth_name = frame["depth_path"] + extension
            depth_path = os.path.join(path, depth_name)
            depth = Image.open(depth_path).copy()
        else:
            depth = None
        tbar.update(1)
        if 'fl_x' in frame and 'fl_y' in frame and 'cx' in frame and 'cy' in frame:
            FovX = FovY = -1.0
            fl_x = frame['fl_x']
            fl_y = frame['fl_y']
            cx = frame['cx']
            cy = frame['cy']

        return SequentialCameraInfo(uid=camera_id, R=R, T=T, FovY=FovY, FovX=FovX, width=width, height=height,
                        fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, znear=bounds[0], zfar=bounds[1], principle_point=None)

    
    with ThreadPool() as pool:      # 使用 map 方法将 frame_read_fn 函数应用到 frames 列表中的每个元素。
        cam_infos = pool.map(frame_read_fn, zip(list(range(len(frames))), frames)) 
        pool.close()
        pool.join()
    
    
    cam_infos = [cam_info for cam_info in cam_infos if cam_info is not None]
    return cam_infos


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Dynerf" : readDynerfInfo,
    "Immersive": readImmersiveSceneInfo,
    "Panoptic": readPanopticInfo,
    "Technicolor" : readTechnicolorInfo,
}