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


import torch
import math
import diff_gaussian_rasterization
import gaussian_rasterization_grad
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.cameras import SequentialCamera
import gsplat
from gsplat.cuda._wrapper import fully_fused_projection

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color, iteration, render_mode, ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # pc.set_anchor_mask(viewpoint_camera.camera_center, iteration, pc, resolution_scale = 1.0)
    # visible_mask = prefilter_voxel(viewpoint_camera, pc, pipe, bg_color).squeeze()  # 这个预过滤过程实现了两级筛选：LOD筛选，以及radii可见性筛选
    visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    xyz, color, opacity, scaling, rot, selection_mask = pc.generate_lod_gaussians(viewpoint_camera, visible_mask)
 
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if viewpoint_camera.fl_x > 0:                                           # Technicolor Dataset
        focal_length_x = viewpoint_camera.fl_x
        focal_length_y = viewpoint_camera.fl_y
    else:
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    
    if viewpoint_camera.fl_x > 0:                                           # Technicolor Dataset
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.cx],
                [0, focal_length_y, viewpoint_camera.cy],
                [0, 0, 1],
            ],
            device="cuda",
        )
    else:
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ],
            device="cuda",
        )
    
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    if viewpoint_camera.fl_x > 0:                                           # Technicolor Dataset
        
        render_colors, render_alphas, info = gsplat.rasterization(
        means=xyz,  # [N, 3]
        quats=rot,  # [N, 4]
        scales=scaling,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=color,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        near_plane=viewpoint_camera.znear,
        far_plane=viewpoint_camera.zfar,
        packed=False,
        sh_degree=pc.max_sh_degree,
        render_mode=render_mode,
        )
    else:
        
        render_colors, render_alphas, info = gsplat.rasterization(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=pc.max_sh_degree,
            render_mode=render_mode,
        )

    # [1, H, W, 3] -> [3, H, W]
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth,
    }
    
    return return_dict


def render_mask(viewpoint_camera: SequentialCamera, pc : GaussianModel, pipe, bg_color : torch.Tensor, iteration, 
                scaling_modifier = 1.0, override_color = None, image_shape = None, pixel_mask=None, 
                color_mask=None, cov_mask=None, render_depth=False, backward_alpha=False, render_flow=False,
                retain_grad=False, update_mask=None, gaussian_mask=None):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """

    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = gaussian_rasterization_grad.GaussianRasterizationSettings(
        image_height=image_shape[1],
        image_width=image_shape[2],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_depth=render_depth,
        backward_alpha=backward_alpha,
        render_flow=render_flow
    )
    rasterizer = gaussian_rasterization_grad.GaussianRasterizer(raster_settings=raster_settings)

    # pc.set_anchor_mask(viewpoint_camera.camera_center, iteration, pc, resolution_scale = 1.0)
    # visible_mask = prefilter_voxel(viewpoint_camera, pc, pipe, bg_color).squeeze()  # 这个预过滤过程实现了两级筛选：LOD筛选，以及radii可见性筛选
    visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    xyz, color, opacity, scaling, rot, selection_mask = pc.generate_lod_gaussians(viewpoint_camera, visible_mask)
    flow3D  = pc.get_flow[visible_mask] 

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_offset, dtype=pc.get_offset.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = screenspace_points[visible_mask]
    try:
        screenspace_points.retain_grad()
    except Exception:
        # Gradient retention failed, continue without it
        pass

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = scaling
        rotations = rot

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_offset - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = color
    else:
        colors_precomp = override_color

    if color_mask is not None:
        colors_precomp = viewpoint_camera.colors_precomp
        assert color_mask.shape[0] == colors_precomp.shape[0]
        
    if update_mask is None:
        update_mask = pc.mask_all[visible_mask]
    else:
        update_mask = pc.mask_all[visible_mask]*update_mask[visible_mask]
    assert update_mask.shape[0] == means3D.shape[0]
    assert update_mask.shape[1] == 7
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    
    if pc.frame_idx>1:
        assert means3D.shape[0] == flow3D.shape[0]
        assert means2D.shape[0] == flow3D.shape[0]
        assert means2D.shape[0] == shs.shape[0]
        assert opacity.shape[0] == shs.shape[0]
        assert opacity.shape[0] == rotations.shape[0]
        assert update_mask.shape[0] == rotations.shape[0]
        if pixel_mask is not None:
            assert pixel_mask.shape[-2] == image_shape[1] and pixel_mask.shape[-1] == image_shape[2]
    
    if gaussian_mask is not None:
        gaussian_mask = gaussian_mask[visible_mask]
        rendered_image, flow2D, infl, count_infl, depth, alpha, radii = rasterizer(
            means3D = means3D[gaussian_mask],
            flow3D = flow3D[gaussian_mask],
            means2D = means2D[gaussian_mask],
            shs = shs[gaussian_mask],
            colors_precomp = colors_precomp[gaussian_mask] if colors_precomp else colors_precomp,
            opacities = opacity[gaussian_mask],
            scales = scales[gaussian_mask],
            rotations = rotations[gaussian_mask],
            cov3D_precomp = cov3D_precomp[gaussian_mask] if cov3D_precomp else cov3D_precomp,
            pixel_mask = pixel_mask,
            color_mask = color_mask[gaussian_mask] if color_mask else color_mask, # 1 means read from colors_precomp, 0 means write to colors_precomp, None means neither
            cov_mask = cov_mask[gaussian_mask] if cov_mask else cov_mask,
            update_mask=update_mask[gaussian_mask],
            )
    else:
        rendered_image, flow2D, infl, count_infl, depth, alpha, radii = rasterizer(
            means3D = means3D,
            flow3D = flow3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            pixel_mask = pixel_mask,
            color_mask = color_mask, # 1 means read from colors_precomp, 0 means write to colors_precomp, None means neither
            cov_mask = cov_mask,
            update_mask=update_mask,
            )
    if retain_grad:
        rendered_image.retain_grad()
        alpha.retain_grad()
        infl.retain_grad()
        depth.retain_grad()
        flow2D.retain_grad()


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    render_pkg = {"render": rendered_image,
                  "flow": None,
                  "alpha": alpha,
                  "influence": infl,
                  "count_influence": count_infl,
                  "viewspace_points": screenspace_points,
                  "visibility_filter" : radii > 0,
                  "radii": radii,
                  "depth": None,
                  "opacity": opacity,
                  "visible_mask": visible_mask,
                  "selection_mask": selection_mask,}
    
    if raster_settings.render_depth:
        render_pkg["depth"] = depth[0]

    if raster_settings.render_flow:
        render_pkg["flow"] = flow2D

    return render_pkg


def prefilter_voxel(viewpoint_camera, pc, pipe, bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    means = pc.get_anchor[pc._anchor_mask]
    scales = pc.get_scaling[pc._anchor_mask][:, :3]
    quats = pc.get_rotation[pc._anchor_mask]
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor([
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],device="cuda",)[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        None,  # covars,
        quats,
        scales,
        viewmats,
        Ks,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        eps2d=0.3,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=False,
    )
    
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0
    
    # if pc.frame_idx > 1:
    visible_mask += True
    
    return visible_mask


def render_from_load(viewpoint_camera, pc: GaussianModel, pipe, bg_color, render_mode, ape_code=-1):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    xyz, color, opacity, scaling, rot, selection_mask = pc.generate_lod_gaussians_from_load(viewpoint_camera, None)
 
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    render_colors, render_alphas, info = gsplat.rasterization(
        means=xyz,  # [N, 3]
        quats=rot,  # [N, 4]
        scales=scaling,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=color,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=pc.max_sh_degree,
        render_mode=render_mode,
    )

    # [1, H, W, 3] -> [3, H, W]
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth
    }
    
    return return_dict

def render_technicolor(viewpoint_camera, pc: GaussianModel, pipe, bg_color, iteration, render_mode, ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # pc.set_anchor_mask(viewpoint_camera.camera_center, iteration, pc, resolution_scale = 1.0)
    # visible_mask = prefilter_voxel(viewpoint_camera, pc, pipe, bg_color).squeeze()  # 这个预过滤过程实现了两级筛选：LOD筛选，以及radii可见性筛选
    visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    xyz, color, opacity, scaling, rot, selection_mask = pc.generate_lod_gaussians(viewpoint_camera, visible_mask)
 
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.fl_x
    focal_length_y = viewpoint_camera.fl_y
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    render_colors, render_alphas, info = gsplat.rasterization(
        means=xyz,  # [N, 3]
        quats=rot,  # [N, 4]
        scales=scaling,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=color,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=pc.max_sh_degree,
        render_mode=render_mode,
    )

    # [1, H, W, 3] -> [3, H, W]
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth
    }
    
    return return_dict