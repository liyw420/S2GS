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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import math
import pickle
from einops import repeat
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal, getWorld2View2, getProjectionMatrix
from collections import OrderedDict
import torch.nn.functional as F
from utils.general_utils import strip_symmetric, build_scaling_rotation, warp_depth, knn, log_base
from utils.image_utils import coords_grid
from utils.graphics_utils import knn_gpu
from utils.compress_utils import CompressedLatents, init_latents
from arguments import QuantizeParams, ModelParams
from scene.decoders import LatentDecoder, DecoderIdentity, DecoderLayer, LatentDecoderRes, Gate
import time
from torch_scatter import scatter_max

class GaussianModel:
    """3D Gaussian Splatting model with quantized latent representations and temporal consistency."""

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, latent_args: QuantizeParams, model_args: ModelParams, opt_lod, frame_idx : int = 1, use_offset_legacy: bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        
        # Octree_LoD
        self.padding = opt_lod.padding
        self.n_offsets = opt_lod.n_offsets
        self.fork = opt_lod.fork
        self.visible_threshold = opt_lod.visible_threshold
        self.dist2level = opt_lod.dist2level
        self.base_layer = opt_lod.base_layer
        self.progressive = opt_lod.progressive
        self.extend = opt_lod.extend
        self.dist_ratio = opt_lod.dist_ratio
        self.levels = opt_lod.levels
        self.init_level = opt_lod.init_level
        self.log_base = opt_lod.log_base
        
        self.use_offset_legacy = use_offset_legacy
        print(f"Using offset_legacy mode: {self.use_offset_legacy}")

        # Initialize latent parameter storage
        self._latents = OrderedDict([(n,torch.empty(0)) for n in latent_args.param_names])

        # Gaussian tracking and optimization state
        self.max_radii2D = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.offset_denom = torch.empty(0)
        
        # Attribute masks for selective updates
        self.mask_anchor = torch.empty(0)
        self.mask_offset = torch.empty(0)
        self.mask_features_dc = torch.empty(0)
        self.mask_features_rest = torch.empty(0)
        self.mask_scaling = torch.empty(0)
        self.mask_rotation = torch.empty(0)
        self.mask_opacity = torch.empty(0)
        self.init_probs = None
        self.added_mask = None

        # Random number generator for splitting operations
        self.split_generator = torch.Generator(device="cuda")
        self.split_generator.manual_seed(latent_args.seed)

        self.param_names = latent_args.param_names
        self.mapping = None

        # Previous frame attributes
        self.prev_atts = OrderedDict({param_name:None for param_name in self.param_names})
        self.prev_latents = OrderedDict({param_name:None for param_name in self.param_names})
        self.prev_atts_initial = OrderedDict({param_name:None for param_name in self.param_names})

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.frame_idx = frame_idx

        # Freeze states for different attributes
        self.frz_anchor = opt_lod.frz_anchor
        self.frz_offset = opt_lod.frz_offset
        self.frz_features_dc =  opt_lod.frz_f_dc
        self.frz_features_rest =  opt_lod.frz_f_rest
        self.frz_scaling =  opt_lod.frz_sc
        self.frz_rotation =  opt_lod.frz_rot
        self.frz_opacity =  opt_lod.frz_op
        self.gate_atts = None
        self.latent_args = latent_args
        self.model_args = model_args
        self.setup_functions()
        self.setup_decoders(latent_args)

    def setup_decoders(self, latent_args: QuantizeParams, verbose=False):
        """Initialize latent decoders for each Gaussian attribute based on quantization settings."""
        self.feature_dims = OrderedDict([
            ("anchor", 3),
            ("offset", 3),
            ("f_dc", 3),
            ("f_rest", 3 * ((self.max_sh_degree + 1) ** 2 - 1)),
            ("sc", 6),
            ("rot", 4),
            ("op", 1)
        ])
        self.latent_decoders = OrderedDict()
        for i, param_name in enumerate(self.param_names):
            self.latent_decoders[param_name] = DecoderIdentity()
            if latent_args.quant_type[i] == 'sq':
                self.latent_decoders[param_name] = LatentDecoder(
                    latent_dim=latent_args.latent_dim[i],
                    feature_dim=self.feature_dims[param_name],
                    ldecode_matrix=latent_args.ldecode_matrix[i],
                    latent_norm=latent_args.latent_norm[i],
                    num_layers_dec=latent_args.num_layers_dec[i],
                    hidden_dim_dec=latent_args.hidden_dim_dec[i],
                    activation=latent_args.activation[i],
                    use_shift=latent_args.use_shift[i],
                    ldec_std=latent_args.ldec_std[i],
                    final_activation=latent_args.final_activation[i],
                ).cuda()
            if verbose:
                print(f"GaussianModel: Created {latent_args.quant_type[i]} decoder for {param_name}")

    @property
    def get_atts(self):
        return self._latents

    @property
    def get_anchor(self):
        # return self._anchor + torch.randn_like(self._anchor.detach()) * 0.0005 * self._anchor[:, -1].unsqueeze(1)
        return self._anchor 

    @property
    def get_decoded_atts(self):
        return OrderedDict({"anchor"  : self._anchor,
                            "offset"  : self._offset,
                            "f_dc"    : self._features_dc,
                            "f_rest"  : self._features_rest,
                            "sc"      : self._scaling,
                            "rot"     : self._rotation,
                            "op"      : self._opacity})
    
    @property
    def get_masks(self):
        return OrderedDict({"anchor"  : self.mask_anchor, 
                            "offset"  : self.mask_offset, 
                            "f_dc"    : self.mask_features_dc, 
                            "f_rest"  : self.mask_features_rest, 
                            "sc"      : self.mask_scaling, 
                            "rot"     : self.mask_rotation,
                            "op"      : self.mask_opacity})
    
    @property
    def get_frz(self):
        return OrderedDict({"anchor"  : self.frz_anchor,
                            "offset"  : self.frz_offset, 
                            "f_dc"    : self.frz_features_dc, 
                            "f_rest"  : self.frz_features_rest, 
                            "sc"      : self.frz_scaling, 
                            "rot"     : self.frz_rotation,
                            "op"      : self.frz_opacity})
    @property
    def mask_all(self):
        mask_all = torch.cat((self.mask_anchor, self.mask_offset, self.mask_features_dc, self.mask_features_rest,
                              self.mask_scaling, self.mask_rotation, self.mask_opacity),dim=1)
        return mask_all
    
    @property
    def _anchor(self):
        """Get anchor parameters with optional gating."""
        anchor = self.latent_decoders["anchor"](self._latents["anchor"])
        return anchor

    @property
    def _offset(self):
        """Get offset coordinates with optional gating and temporal consistency."""

        if self.use_offset_legacy:
            return self._offset_legacy()
        else:
            return self._offset_fixed()
    
    def _offset_legacy(self):
        # Decode latents to get offset attribute
        offset = self.latent_decoders["offset"](self._latents["offset"])
        
        # Apply gating if previous frame attributes exist and gating is enabled
        if self.prev_atts["offset"] is not None and self.gate_atts is not None and self.gate_params["offset"]:
            offset = self.gate_atts(offset-self.prev_atts["offset"])+self.prev_atts["offset"]
        return offset

    def _offset_fixed(self):
        # Decode latents to get offset attribute
        with torch.no_grad():
            self._latents["offset"][torch.isnan(self._latents["offset"])] = 0.0
        offset = self.latent_decoders["offset"](self._latents["offset"])
        
        # Apply gating if previous frame attributes exist and gating is enabled
        if self.prev_atts["offset"] is not None and self.gate_atts is not None and self.gate_params["offset"] and isinstance(self.latent_decoders['sc'], LatentDecoderRes):
            # Use gated residual from previous frame
            try:
                offset = self.gate_atts(offset-self.offset_before[self.mapping])+self.offset_before[self.mapping]
            except:
                # Handle mapping errors gracefully
                print(f"Warning: offset mapping error - offset shape: {offset.shape}, "
                      f"prev_atts shape: {self.prev_atts['offset'].shape}, "
                      f"offset_before shape: {self.offset_before.shape}, "
                      f"mapping max: {self.mapping.max()}")
                raise
            
            if (torch.isnan(offset).sum() > 0):
                raise ValueError("Nan")
            
        return offset

    @property
    def _ungated_offset_res(self):
        """Get ungated offset residual for regularization."""
        with torch.no_grad():
            self._latents["offset"][torch.isnan(self._latents["offset"])] = 0.0
        offset = self.latent_decoders["offset"](self._latents["offset"])
        return offset-self.prev_atts["offset"]
    
    @property
    def _features_dc(self):
        """Get DC (0th order) spherical harmonics features with optional gating."""
        if isinstance(self.latent_decoders["f_dc"], DecoderIdentity):
            features_dc = self._latents["f_dc"]
        else:
            features_dc = self.latent_decoders["f_dc"](self._latents["f_dc"])
            features_dc = features_dc.reshape(features_dc.shape[0], 1, 3)
        
        # Apply gating if enabled
        if self.prev_atts["f_dc"] is not None and self.gate_atts is not None and self.gate_params["f_dc"]:
            features_dc = self.gate_atts(features_dc-self.prev_atts["f_dc"])+self.prev_atts["f_dc"]
        return features_dc  # shape (N, 1, 3)
    
    @property
    def _features_rest(self):
        """Get higher-order spherical harmonics features with optional gating."""
        if isinstance(self.latent_decoders["f_rest"], DecoderIdentity):
            features_rest = self._latents["f_rest"]
        else:
            features_rest = self.latent_decoders["f_rest"](self._latents["f_rest"])
            features_rest = features_rest.reshape(features_rest.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)
        
        # Apply gating if enabled
        if self.prev_atts["f_rest"] is not None and self.gate_atts is not None and self.gate_params["f_rest"]:
            features_rest = self.gate_atts(features_rest-self.prev_atts["f_rest"])+self.prev_atts["f_rest"]
        return features_rest  # shape (N, C-1, 3)
    
    @property
    def _scaling(self):
        """Get scaling parameters with optional gating."""
        scaling = self.latent_decoders["sc"](self._latents["sc"])
        if self.prev_atts["sc"] is not None and self.gate_atts is not None and self.gate_params["sc"]:
            scaling = self.gate_atts(scaling-self.prev_atts["sc"])+self.prev_atts["sc"]
        
        if (torch.isnan(scaling).sum() > 0):
            raise ValueError("Nan")
        
        return scaling
    
    @property
    def _rotation(self):
        """Get rotation quaternions with optional gating."""
        rot = self.latent_decoders["rot"](self._latents["rot"])
        if self.prev_atts["rot"] is not None and self.gate_atts is not None and self.gate_params["rot"]:
            rot = self.gate_atts(rot-self.prev_atts["rot"])+self.prev_atts["rot"]
        return rot
    
    @property
    def _opacity(self):
        """Get opacity values with optional gating."""
        op = self.latent_decoders["op"](self._latents["op"])
        if self.prev_atts["op"] is not None and self.gate_atts is not None and self.gate_params["op"]:
            op = self.gate_atts(op-self.prev_atts["op"])+self.prev_atts["op"]
        return op
    
    @property
    def get_flow(self):
        return nn.Parameter(torch.zeros_like(self._opacity).requires_grad_(True))

    @property
    def mask_cov(self):
        return torch.logical_or(self.mask_scaling, self.mask_rotation)
    
    @property
    def mask_color(self):
        return torch.logical_or(self.mask_features_dc, self.mask_features_rest)
    
    @property
    def get_scaling(self):
        return torch.clamp(self.scaling_activation(self._scaling), min=1e-8)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_offset(self):
        return self._offset

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    
    def parameters(self):
        return list(self._latents.values()) + \
                [param for decoder in self.latent_decoders.values() for param in list(decoder.parameters())]
    
    def named_parameters(self):
        parameter_dict = self._latents
        for n, decoder in self.latent_decoders.items():
            parameter_dict.update(
                {n+'.'+param_name:param for param_name, param in dict(decoder.named_parameters()).items()}
                )
        return parameter_dict
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def norm_decoders(self, att_name=None):
        for param_name in self.param_names:
            if att_name is not None and param_name!=att_name:
                continue
            decoder = self.latent_decoders[param_name]
            if not isinstance(decoder, DecoderIdentity) and decoder.norm!="none":
                decoder.normalize(self._latents[param_name])
                
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale, cameras, scales):
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(pcd.points, dtype=torch.float, device="cuda")
        colors = torch.tensor(pcd.colors, dtype=torch.float, device="cuda")
        self.set_level(points, cameras, scales)
        box_min = torch.min(points)*self.extend
        box_max = torch.max(points)*self.extend
        box_d = box_max - box_min
        print(box_d)
        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(points, colors)  

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _, self.root_indices = self.weed_out(self.positions, self._level, self.root_indices)
        self.positions, self._level, _, weed_mask, self.root_indices= self.weed_out(self.positions, self._level, self.root_indices)
        self.colors = self.colors[weed_mask]

        PINK = '\033[1;95m'      # 亮洋红色（通常显示为粉色）
        RESET = '\033[0m'

        print(f'{PINK}Branches of Tree: {self.fork}{RESET}')
        print(f'{PINK}Base Layer of Tree: {self.base_layer}{RESET}')
        print(f'{PINK}Visible Threshold: {self.visible_threshold}{RESET}')
        print(f'{PINK}LOD Levels: {self.levels}{RESET}')
        print(f'{PINK}Initial Levels: {self.init_level}{RESET}')
        print(f'{PINK}Initial Voxel Number: {self.positions.shape[0]}{RESET}')
        print(f'{PINK}Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}{RESET}')
        print(f'{PINK}Max Voxel Size: {self.voxel_size}{RESET}')
        print(f'{PINK}Unique Base Voxels: {len(torch.unique(self.root_indices))}{RESET}')

        fused_point_cloud, fused_color = self.positions, RGB2SH(self.colors)
        offsets = torch.zeros((fused_point_cloud.shape[0], 3)).float().cuda()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(fused_point_cloud)).float().cuda()), 0.0000001)
        dist2 = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        scales = torch.clamp(scales, -10, 4)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        anchor = fused_point_cloud
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(anchor.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.ones(anchor.shape[0], dtype=torch.bool, device="cuda")
        
        ###################################### Init latents ##########################################
        self._latents = OrderedDict([(n,None) for n in self.param_names])

        init = self.latent_decoders["anchor"].invert(anchor)
        self._latents["anchor"] = nn.Parameter(init.requires_grad_(True))
        
        init = self.latent_decoders["offset"].invert(offsets)
        self._latents["offset"] = nn.Parameter(init.requires_grad_(True))

        init = features[:,:,0:1].transpose(1, 2).contiguous()
        if self.latent_args.f_dc_invert_type == "autoenc" and not isinstance(self.latent_decoders["f_dc"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, fused_color.flatten(1).cuda(),"f_dc", lambda_distortion=0.0)
            self.latent_decoders["f_dc"].load_state_dict(decoder_state_dict)
        elif not isinstance(self.latent_decoders["f_dc"], DecoderIdentity):
            init = self.latent_decoders["f_dc"].invert(fused_color.flatten(start_dim=1).contiguous().cuda())
        self._latents["f_dc"] = nn.Parameter(init.requires_grad_(True))

        init = features[:,:,1:].transpose(1, 2).contiguous()
        if isinstance(self.latent_decoders["f_rest"], LatentDecoder):
            init = torch.zeros((features.size(0),self.latent_decoders["f_rest"].latent_dim)).to(init).contiguous()
        self._latents["f_rest"] = nn.Parameter(init.requires_grad_(True))

        if self.latent_args.sc_invert_type == "autoenc" and not isinstance(self.latent_decoders["sc"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, scales,"sc", lambda_distortion=0.0)
            self.latent_decoders["sc"].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders["sc"].invert(scales)
        self._latents["sc"] = nn.Parameter(init.requires_grad_(True))

        if self.latent_args.rot_invert_type == "autoenc" and not isinstance(self.latent_decoders["rot"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, rots,"rot", lambda_distortion=0.0)
            self.latent_decoders["rot"].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders["rot"].invert(rots)
        self._latents["rot"] = nn.Parameter(init.requires_grad_(True))

        if self.latent_args.op_invert_type == "autoenc" and not isinstance(self.latent_decoders["op"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, opacities,"op", lambda_distortion=0.0)
            self.latent_decoders["op"].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders["op"].invert(opacities)
        self._latents["op"] = nn.Parameter(init.requires_grad_(True))

        ##########################################################################################

        self.max_radii2D = torch.zeros((self.get_offset.shape[0]), device="cuda")

        self.mask_anchor.data = torch.ones_like(self._opacity).bool()
        self.mask_offset.data = torch.ones_like(self._opacity).bool()
        self.mask_features_dc.data = torch.ones_like(self._opacity).bool()
        self.mask_features_rest.data= torch.ones_like(self._opacity).bool()
        self.mask_scaling.data = torch.ones_like(self._opacity).bool()
        self.mask_rotation.data = torch.ones_like(self._opacity).bool()
        self.mask_opacity.data = torch.ones_like(self._opacity).bool()

    def create_from_depth_immersive(self, cameras, spatial_lr_scale, downsample_scale=2, alpha_thresh=0.1, renderFunc=None):
        self.spatial_lr_scale = spatial_lr_scale

        _,orig_H,orig_W = cameras[0].original_image.shape
        downsample_size = (int(orig_H/downsample_scale),int(orig_W/downsample_scale))
        H,W = downsample_size
        xyz = coords_grid(1,H,W, device='cuda')[0].permute(1,2,0).view(-1,2)
        xyz = torch.cat((xyz,torch.zeros_like(xyz[:,0:1]),torch.ones_like(xyz[:,0:1])),dim=-1) # N x 4
        xyz[:,0] = xyz[:,0]/(0.5*W)+(1/W-1)
        xyz[:,1] = xyz[:,1]/(0.5*H)+(1/H-1)
        fused_point_cloud, fused_color = None, None
        world_xyz_colmap = torch.cat((self._offset,torch.ones_like(self._offset[:,0:1])),dim=1)
        net_points = 0
        for idx in range(len(cameras)):
            camera = cameras[idx]

            #################################################### Colmap coords ####################################################
            
            with torch.no_grad():
                world_xyz_colmap = torch.cat((self._offset,torch.ones_like(self._offset[:,0:1])),dim=1)
                cam_xyz_colmap = torch.matmul(camera.world_view_transform.T.unsqueeze(0),world_xyz_colmap.unsqueeze(-1)).squeeze(-1)
                cam_hom_colmap = torch.matmul(camera.projection_matrix.T.unsqueeze(0),cam_xyz_colmap.unsqueeze(-1)).squeeze(-1)
                cam_proj_colmap = cam_hom_colmap[:,:3]/cam_hom_colmap[:,3:]

                in_frustum = torch.logical_and(torch.all(cam_proj_colmap<1,dim=1),torch.all(cam_proj_colmap>-1,dim=1))
                in_frustum_depth = cam_proj_colmap[:,2]>0

                cam_proj_colmap_filtered = cam_proj_colmap[in_frustum*in_frustum_depth]
                cam_xyz_colmap_filtered = cam_xyz_colmap[in_frustum*in_frustum_depth]
                world_xyz_colmap_filtered = world_xyz_colmap[in_frustum*in_frustum_depth]

            # X, Y, Z = cam_xyz_colmap_filtered[:,0], cam_xyz_colmap_filtered[:,1], cam_xyz_colmap_filtered[:,2]
            # xp, yp, zp, wp = 1/math.tan((camera.FoVx/2))*X/Z, 1/math.tan((camera.FoVx/2))*Y/Z, camera.zfar/(camera.zfar-camera.znear)*(1-camera.znear/Z), Z

            ######################################### Scaling values for inverse depth #############################################

            with torch.no_grad():
                image, depth = camera.original_image, camera.gt_depth # Actually inverse_depth for midas

                # Grid sample expects a  N x H x W x 2 grid for some reason. Create an array with first N values populated and reshape
                grid = torch.zeros(1,H,W,2).reshape(-1,2)
                grid[:cam_proj_colmap_filtered.shape[0]] = cam_proj_colmap_filtered[:,:2]

                # Sample from network produced inverse depth at those coordinates
                img = F.grid_sample(depth.unsqueeze(0).unsqueeze(0), grid.to(depth).reshape(1,H,W,2), mode='bilinear', padding_mode='zeros', align_corners=True)
                # Use only the coordinates at populated grid locations
                inverse_depths = img.reshape(-1,1)[:cam_proj_colmap_filtered.shape[0]]

                orig_depths = cam_xyz_colmap_filtered[:,2]
                inverse_gt_depths = 1/(orig_depths+camera.znear)
                out = torch.linalg.lstsq(torch.cat((inverse_depths,torch.ones_like(inverse_depths)),dim=1), inverse_gt_depths)

            ###################################### Subsample and index locations for depth #############################################

            with torch.no_grad():
                Zc = 1/(depth*out.solution[0]+out.solution[1]).view(-1,1)
                Xc = Zc*math.tan((camera.FoVx/2))*xyz[:,0:1]
                Yc = Zc*math.tan((camera.FoVy/2))*xyz[:,1:2]
                cam_xyz = torch.cat((Xc, Yc, Zc, torch.ones_like(Xc)),dim=1)
                world_xyz = torch.matmul(torch.linalg.inv(camera.world_view_transform.T).unsqueeze(0),cam_xyz.unsqueeze(-1)).squeeze(-1)[:,:3]

                cam_hom= torch.matmul(camera.projection_matrix.T.unsqueeze(0),cam_xyz.unsqueeze(-1)).squeeze(-1)
                depth = cam_hom[:,2]/cam_hom[:,3]
                camera.gt_depth = depth.reshape(orig_H, orig_W)

                render_pkg = renderFunc(viewpoint_camera=camera,pc=self)
                _, alpha = render_pkg["depth"], render_pkg["alpha"]
                alpha = F.interpolate(alpha.unsqueeze(0), size=downsample_size, mode='bilinear',
                                        align_corners=True)[0]
                alpha_mask = (alpha<alpha_thresh).reshape(-1) # 1 implies we add points from our GT depth, 0 implies neighborhood points already exist from colmap
                if alpha_mask.sum()==0:
                    continue
                assert alpha_mask.sum()!=alpha_mask.numel()

            # New points to add in empty regions
            num_new = alpha_mask.sum()/(~alpha_mask).numel()*((in_frustum*in_frustum_depth).sum())
            if num_new == 0:
                continue

            world_xyz_new = world_xyz[alpha_mask] # Only points corresponding to new areas without colmap init

            indices = torch.randperm(world_xyz_new.shape[0])[:num_new.long()]
            world_xyz_subsampled = world_xyz_new[indices]
            net_points += world_xyz_subsampled.shape[0]

            ################################################### Color #########################################################

            image_new = image.permute(1,2,0).view(-1,3)[alpha_mask,:] # Get corresponding pixels at the new areas
            image_subsampled = image_new[indices]

            color = RGB2SH(image_subsampled)
            features = torch.zeros((color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = color
            features[:, 3:, 1:] = 0.0

            if fused_point_cloud is None :
                fused_point_cloud = world_xyz_subsampled
                fused_color = features
            else:
                fused_point_cloud = torch.cat((fused_point_cloud,world_xyz_subsampled))
                fused_color = torch.cat((fused_color,features))

        if fused_color is None:
            return
            
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        new_xyz = self.latent_decoders["xyz"].invert(fused_point_cloud)
        if isinstance(self.latent_decoders["f_dc"],DecoderIdentity):
            new_features_dc = fused_color[:,:,0:1].transpose(1, 2).contiguous()
        else:
            new_features_dc = self.latent_decoders["f_dc"].invert(fused_color[:,:,0].contiguous().cuda())
        if isinstance(self.latent_decoders["f_rest"],DecoderIdentity):
            new_features_rest = fused_color[:,:,1:].transpose(1, 2).contiguous()
        else:
            new_features_rest = torch.zeros((fused_color.size(0),self.latent_decoders["f_rest"].latent_dim)).to(fused_color).contiguous()

        new_scaling = self.latent_decoders["sc"].invert(scales)
        new_rotation = self.latent_decoders["rot"].invert(rots)
        new_opacity = self.latent_decoders["op"].invert(opacities)
        new_flow = torch.zeros_like(fused_point_cloud)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_flow)


        print("\nAdded {} points!".format(net_points))

    @torch.no_grad()
    def update_points_flow(self):
        if type(self.latent_decoders["offset"]) == LatentDecoderRes:
            self._latents["offset"].data = self.latent_decoders["offset"].invert(self._flow)*self.mask_offset
        elif isinstance(self.latent_decoders["offset"], DecoderIdentity) \
            or isinstance(self.latent_decoders["offset"], LatentDecoder):
            if self.gate_atts is not None:
                new_offset = self._offset + self._flow*self.mask_offset*self.gate_atts.gate.unsqueeze(-1)
            else:
                new_offset = self._offset+self._flow*self.mask_offset
            self._latents["offset"].data = self.latent_decoders["offset"].invert(new_offset)

        self._latents["flow"] *= 0

    def size(self):
        """Calculate compressed model size in bits for storage estimation."""
        with torch.no_grad():
            latents_size = ldec_size = 0
            frz, masks = self.get_frz, self.get_masks
            
            for param_name in self.param_names:
                if param_name == "anchor":
                    continue
                mask = masks[param_name].flatten()
                    
                # Add decoder size
                ldec_size += self.latent_decoders[param_name].size()
                decoder = self.latent_decoders[param_name]
                
                if self.gate_atts is not None:
                    gate = self.gate_atts.get_gates()
                    mask = (gate!=0.0).sum().item()/gate.numel()

                # Calculate latent storage size based on decoder type
                if isinstance(decoder, DecoderIdentity) \
                    or (type(decoder)==LatentDecoderRes and decoder.identity):
                    p = self._latents[param_name]
                    mask_frac = 1.0
                    
                    # Apply masking based on freeze state
                    if frz[param_name] == "st":
                        mask_frac = mask.sum().item()/mask.numel()
                    elif frz[param_name] == "all":
                        mask_frac = 0.0
                        
                    # Apply gating compression if enabled
                    if self.gate_atts is not None and self.gate_params[param_name]:
                        assert frz[param_name] == "none"
                        mask_frac = mask
                    latents_size += p.numel()*torch.finfo(p.dtype).bits*mask_frac
                else:
                    if frz[param_name] == "all":
                        continue
                        
                    # Determine previous attribute for compression
                    if type(self.latent_decoders[param_name])==LatentDecoder:
                        if self.frame_idx == 1:
                            prev_att = None
                        else:
                            prev_att = torch.round(self.prev_latents[param_name])
                    elif type(self.latent_decoders[param_name])==LatentDecoderRes:
                        prev_att = None
                    else:
                        raise Exception(f"Unknown {param_name} decoder {type(self.latent_decoders[param_name])}")
                        
                    # Calculate entropy-based compression size
                    for dim in range(self._latents[param_name].size(1)):
                        if prev_att is not None:
                            weight = (torch.round(self._latents[param_name][:,dim])-prev_att[:,dim]).long()
                        else:
                            weight = torch.round(self._latents[param_name][:,dim]).long() 
                        if frz[param_name] == "st":
                            weight = weight[mask]
                        
                        unique_vals, counts = torch.unique(weight, return_counts = True)
                        probs = counts/torch.sum(counts)

                        information_bits = torch.clamp(-1.0 * torch.log(probs + 1e-10) / np.log(2.0), 0, 1000)
                        size_bits = torch.sum(information_bits*counts).item()
                        latents_size += size_bits
                        
            # Add gate size if present
            if self.gate_atts is not None:
                latents_size += self.gate_atts.size()
                
        return ldec_size+latents_size
    
    def training_setup(self, training_args, opt_lod):
        # self.percent_dense = training_args.percent_dense
        self.offset_gradient_accum = torch.zeros((self.get_offset.shape[0], 1), device="cuda")
        self.infl_accum = torch.zeros((self.get_offset.shape[0]), device="cuda")
        self.infl_denom = torch.zeros((self.get_offset.shape[0]), device="cuda")
        self.added_mask = None

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.lr_scaling = OrderedDict()
        self.gate_params = OrderedDict()
        for i,param in enumerate(self.param_names):
            decoder = self.latent_decoders[param]
            if type(decoder) == DecoderIdentity or (type(decoder)== LatentDecoderRes and decoder.identity):
                self.lr_scaling[param] = 1.0
            else:
                self.lr_scaling[param] = training_args.latents_lr_scaling[i]
            self.gate_params[param] = self.latent_args.gate_params[i]!="none"

        lr = {  'anchor':opt_lod.position_lr_init * self.spatial_lr_scale,
                'offset':opt_lod.offset_lr_init * self.spatial_lr_scale*self.lr_scaling["offset"],
                'f_dc':opt_lod.feature_lr*self.lr_scaling["f_dc"],
                'f_rest':opt_lod.feature_lr*self.lr_scaling["f_rest"],
                'sc':opt_lod.scaling_lr*self.lr_scaling["sc"],
                'rot':opt_lod.rotation_lr*self.lr_scaling["rot"],
                'op':opt_lod.opacity_lr*self.lr_scaling["op"],
            }
        
        l = []
        for i,param in enumerate(self.param_names):
            l += [{'params': [self._latents[param]], 'lr': lr[param], "name": param}]
            if not isinstance(self.latent_decoders[param], DecoderIdentity):
                l += [{'params': self.latent_decoders[param].parameters(), 'lr': training_args.ldecs_lr[i], "name":f"ldec_{param}"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=opt_lod.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=opt_lod.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=opt_lod.position_lr_delay_mult,
                                                    max_steps=opt_lod.position_lr_max_steps)

        self.offset_scheduler_args = get_expon_lr_func(lr_init=opt_lod.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=opt_lod.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=opt_lod.offset_lr_delay_mult,
                                                    max_steps=opt_lod.offset_lr_max_steps)

    def update_masks(self, args, dynamic_mask):
        frz_args = {"offset":args.frz_offset, "f_dc":args.frz_f_dc, "f_rest":args.frz_f_rest, 
                    "op":args.frz_op, "rot":args.frz_rot, "sc":args.frz_sc}
        # frz_args = self.get_frz
        masks = self.get_masks
        for param_name in frz_args:
            if frz_args[param_name] not in ["none","all"]:
                assert dynamic_mask is not None
                masks[param_name].data = dynamic_mask
                # lod_mask = (self.get_level == 1) | (self.get_level == 2) | (self.get_level == 3)
                # lod_mask = (self.get_level == 4)
                # masks[param_name].data = lod_mask
                # masks[param_name].data *= False
            elif frz_args[param_name] == "all":
                masks[param_name].data *= False
            elif frz_args[param_name] == "none":
                masks[param_name].data += True

    def freeze_atts(self, args):
        self.frz_anchor = args.frz_anchor
        self.frz_offset = args.frz_offset
        self.frz_features_dc = args.frz_f_dc
        self.frz_features_rest = args.frz_f_rest
        self.frz_scaling = args.frz_sc
        self.frz_rotation = args.frz_rot
        self.frz_opacity = args.frz_op

        # 0 is static content 1 is dynamic content
        frz, atts, masks = self.get_frz, self.get_atts, self.get_masks
        for att_name in atts:
            if torch.any(~masks[att_name]) and frz[att_name]!="none":
                if "f_" in att_name:
                    mask = masks[att_name][...,None]
                else:
                    mask = masks[att_name]
                if frz[att_name] == "all":
                    atts[att_name].requires_grad_(False)
                    for param in self.latent_decoders[att_name].parameters():
                        param.requires_grad_(False)
                elif frz[att_name] == "st":
                    self.latent_decoders[att_name].freeze_partial(masks[att_name].flatten())
                    continue
                    if "f_" in att_name:
                        atts[att_name].register_post_accumulate_grad_hook(lambda grad: grad.mul_(mask.unsqueeze(-1)))
                    else:
                        atts[att_name].register_hook(lambda grad: grad*(mask))
                else:
                    raise Exception('Undefined mode ', frz[att_name])

    def std_reg(self):
        net_std = 0.0
        for att_name in self.get_atts:
            decoder = self.latent_decoders[att_name]
            if "offset" not in att_name and "flow" not in att_name and \
                type(decoder)!=DecoderIdentity:
                net_std += self._latents[att_name].std(dim=0).mean()
        return net_std
    
    def update_residuals(self,dataset):
        """Initialize residual encoders for temporal compression in subsequent frames."""
        atts = self.get_atts # All the latent variables
        
        # Initialize parent mapping for tracking Gaussian relationships
        self.offset_before = self.get_offset.clone()
        if self.gate_atts is not None:
            self.gate_before = self.gate_atts.get_gates().clone()
        self.mapping =  torch.arange(self.offset_before.shape[0], device="cuda")
        
        for i, att_name in enumerate(atts):
            if self.latent_args.quant_type[i] == 'sq_res':
                decoded_att = self.latent_decoders[att_name](self._latents[att_name])
                
                if self.frame_idx == 2:
                    # Switch from identity to residual decoder after frame 1
                    assert isinstance(self.latent_decoders[att_name], DecoderIdentity)
                    decoder = LatentDecoderRes(
                        latent_dim=self.latent_args.latent_dim[i],
                        feature_dim=self.feature_dims[att_name],
                        ldecode_matrix=self.latent_args.ldecode_matrix[i],
                        latent_norm=self.latent_args.latent_norm[i],
                        num_layers_dec=self.latent_args.num_layers_dec[i],
                        hidden_dim_dec=self.latent_args.hidden_dim_dec[i],
                        activation=self.latent_args.activation[i],
                        use_shift=self.latent_args.use_shift[i],
                        ldec_std=self.latent_args.ldec_std[i],
                        final_activation=self.latent_args.final_activation[i],
                        gates = self.gate_before if self.gate_atts is not None else None,
                    ).cuda()
                    self.latent_decoders[att_name] = decoder
                else:
                    # Verify residual decoder type for subsequent frames
                    decoder = self.latent_decoders[att_name]
                    assert isinstance(self.latent_decoders[att_name], LatentDecoderRes)
                    decoder.gates = self.gate_before if self.gate_atts is not None else None

                self.latent_decoders[att_name].frame_idx = self.frame_idx

                # Initialize decoder with previous frame's decoded attributes
                if "f_" in att_name and self.frame_idx == 2:
                    decoder.init_decoded(decoded_att.reshape(decoded_att.shape[0], -1))  # SH coefficients in (N, X)
                else:
                    decoder.init_decoded(decoded_att)

                # Set up new latent parameters for current frame
                if type(self.latent_decoders[att_name])== LatentDecoderRes \
                    and self.latent_args.quant_after[i]>0.0:
                    # Start in identity mode, switch to quantized later
                    self.latent_decoders[att_name].identity = True
                    decoder = self.latent_decoders[att_name]
                    latent = torch.zeros_like(decoder.decoded_att)
                    self._latents[att_name] = nn.Parameter(latent.requires_grad_(True))
                else:
                    # Initialize with zero residuals
                    self._latents[att_name] = nn.Parameter(
                                                torch.zeros((self._latents[att_name].shape[0],
                                                                self.latent_args.latent_dim[i]), 
                                                            dtype=torch.float, 
                                                            device="cuda").requires_grad_(True)
                                                )

    def update_grads(self):
        frz, atts, masks = self.get_frz, self.get_atts, self.get_masks
        for att_name in atts:
            if frz[att_name] == 'st':
                mask = masks[att_name] if frz[att_name] == 'st' else ~masks[att_name]
                if "f_" in att_name:
                    atts[att_name].grad *= mask.unsqueeze(-1)
                else:
                    atts[att_name].grad *= mask

    def update_learning_rate(self, iteration, latent_args: QuantizeParams):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr            
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr*self.lr_scaling["offset"]
            elif param_group["name"] in self.param_names:
                idx = self.param_names.index(param_group["name"])

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, mask):
        mkdir_p(os.path.dirname(path))

        anchor = self.get_anchor
        grid_offsets = self.get_offset
        scaling = self.get_scaling
        offsets = grid_offsets.view([-1, 3]) * scaling[:,:3]
        scale = scaling[:,3:].detach().cpu().numpy()
        # scale = scaling.detach().cpu().numpy()
        xyz = anchor + offsets
        # xyz = anchor
        xyz = xyz.detach().cpu().numpy()
        xyz = xyz.astype(np.float32)

        normals = np.zeros_like(xyz)
        normals = normals.astype(np.float32)
        
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        vertex_ids = np.arange(xyz.shape[0])
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
        # dtype_full.extend([(attribute, 'f4') for attribute in 
        #                 [f'f_dc_{i}' for i in range(f_dc.shape[1])]])
        # dtype_full.extend([(attribute, 'f4') for attribute in 
        #                 [f'f_rest_{i}' for i in range(f_rest.shape[1])]])
        # dtype_full.extend([('opacity', 'f4')])
        # dtype_full.extend([(attribute, 'f4') for attribute in 
        #                 [f'scale_{i}' for i in range(scale.shape[1])]])
        # dtype_full.extend([(attribute, 'f4') for attribute in 
        #                 [f'rot_{i}' for i in range(rotation.shape[1])]])
        # dtype_full.append(('vertex_id', 'i4'))  # Add the vertex_id field

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        # for i in range(f_dc.shape[1]):
        #     elements[f'f_dc_{i}'] = f_dc[:, i]
        # for i in range(f_rest.shape[1]):
        #     elements[f'f_rest_{i}'] = f_rest[:, i]
        # elements['opacity'] = opacities[:, 0]  # Fix the shape here
        # for i in range(scale.shape[1]):
        #     elements[f'scale_{i}'] = scale[:, i]
        # for i in range(rotation.shape[1]):
        #     elements[f'rot_{i}'] = rotation[:, i]
        # elements['vertex_id'] = vertex_ids

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def save_compressed_pkl(self, path, latent_args):
    #     mkdir_p(os.path.dirname(path))

    #     latents = OrderedDict()
    #     decoder_state_dict = OrderedDict()
    #     decoder_args = OrderedDict()
                
    #     for i,attribute in enumerate(self.param_names):
    #         if isinstance(self.latent_decoders[attribute], DecoderIdentity):
    #             if self.prev_atts[attribute] is not None and self.gate_atts is not None and self.gate_params[attribute]:
    #                 xyz = self.latent_decoders["offset"](self._latents["offset"])
    #                 prev = self.offset_before[self.mapping]
    #                 residual_xyz = xyz - prev
                    
    #                 # Get ungated indices and their residual values
    #                 ungated_xyz_indices = self.gate_atts.sample_gate(stochastic=False).nonzero(as_tuple=True)[0]
    #                 # compute the number of bits needed to store the max value in ungated_xyz_indices
    #                 max_value = ungated_xyz_indices.max()
    #                 num_bits = max_value.item().bit_length()
    #                 if num_bits < 8:
    #                     ungated_xyz_indices = ungated_xyz_indices.type(torch.int8)
    #                 elif num_bits < 16:
    #                     ungated_xyz_indices = ungated_xyz_indices.type(torch.short)
    #                 elif num_bits < 32:
    #                     ungated_xyz_indices = ungated_xyz_indices.type(torch.int)
    #                 else:
    #                     ungated_xyz_indices = ungated_xyz_indices.type(torch.long)
    #                 # Store the gated residuals, not raw residuals
    #                 ungated_residuals = self.gate_atts(residual_xyz)[ungated_xyz_indices]
    #                 compressed_residuals = CompressedLatents()
    #                 compressed_residuals.compress(ungated_residuals, scale=10000.0)

    #                 # Store minimal data needed for reconstruction
    #                 latents[attribute] = {
    #                     'mapping': self.mapping,
    #                     'ungated_indices': ungated_xyz_indices,
    #                     'ungated_residuals_compressed': compressed_residuals
    #                     # '_xyz_debug': self._xyz,
    #                     # 'xyz_before_debug': self.xyz_before,
    #                     # 'prev_att_debug': self.prev_atts[attribute]
    #                 }
                    
    #                 # Verify reconstruction matches _xyz
    #                 reconstructed_xyz = prev.clone()
    #                 reconstructed_xyz[ungated_xyz_indices] += ungated_residuals
    #                 # assert torch.allclose(self._xyz, reconstructed_xyz, rtol=1e-5, atol=1e-5), "Reconstruction verification failed!"
    #             else:
    #                 # For non-gated DecoderIdentity attributes, just store the latents directly
    #                 latents[attribute] = self._latents[attribute].detach().cpu()
    #         else:
    #             latent = self._latents[attribute].detach().cpu()
    #             compressed_obj = CompressedLatents()
    #             compressed_obj.compress(latent)
    #             latents[attribute] = compressed_obj
    #             decoder_args[attribute] = {
    #                 'latent_dim': latent_args.latent_dim[i],
    #                 'feature_dim': self.feature_dims[attribute],
    #                 'ldecode_matrix': latent_args.ldecode_matrix[i],
    #                 'latent_norm': latent_args.latent_norm[i],
    #                 'num_layers_dec': latent_args.num_layers_dec[i],
    #                 'hidden_dim_dec': latent_args.hidden_dim_dec[i],
    #                 'activation': latent_args.activation[i],
    #                 'use_shift': latent_args.use_shift[i],
    #                 'ldec_std': latent_args.ldec_std[i]
    #             }
    #             decoder_state_dict[attribute] = self.latent_decoders[attribute].state_dict().copy()

    #             # manually add the decoded_att to the state_dict
    #             if hasattr(self.latent_decoders[attribute], 'decoded_att'):
    #                 decoder_state_dict[attribute]['decoded_att'] = self.latent_decoders[attribute].decoded_att.detach().cpu()

    #     save_state = {
    #                      'latents': latents,
    #                      'decoder_state_dict': decoder_state_dict,
    #                      'decoder_args': decoder_args,
    #                      'latent_decoders_dict': {attr: type(self.latent_decoders[attr]).__name__ for attr in self.param_names}
    #         }

    #     with open(path,'wb') as f:
    #         pickle.dump(save_state, f)

    # def load_compressed_pkl(self, path):
    #     with open(path,'rb') as f:
    #         data = pickle.load(f)
    #         latents = data['latents']
    #         decoder_state_dict = data['decoder_state_dict']
    #         decoder_args = data['decoder_args']
    #         latent_decoders_dict = data['latent_decoders_dict']
                        
    #     # First verify the number of gaussians matches
    #     num_gaussians = None
    #     for attribute in latents:
    #         if isinstance(latents[attribute], dict) and 'num_gaussians' in latents[attribute]:
    #             if num_gaussians is None:
    #                 num_gaussians = latents[attribute]['num_gaussians']
    #             elif num_gaussians != latents[attribute]['num_gaussians']:
    #                 raise ValueError(f"Inconsistent number of gaussians in compressed data: {num_gaussians} vs {latents[attribute]['num_gaussians']}")
        
    #     # Initialize gate if needed
    #     if num_gaussians is not None:
    #         if self.gate_atts is None:
    #             self.gate_atts = Gate(num_gaussians, 
    #                                 gamma=self.model_args.gate_gamma,
    #                                  eta=self.model_args.gate_eta,
    #                                  lr=self.model_args.gate_lr,
    #                                  temp=self.model_args.gate_temp).cuda()
        
    #     # Then proceed with loading attributes
    #     for i, attribute in enumerate(latents):
    #         if self.latent_args.gate_params[i] == 'on':
    #             # if the gate_params is on, that means we are loading gated attributes
    #             # Reconstruct prev_atts from sparse difference
    #             if self.prev_atts[attribute] is not None:
    #                 prev_att = self.prev_atts[attribute]
    #             else:
    #                 prev_att = torch.zeros([num_gaussians, 3], device="cuda")

    #             prev_att_mapping = latents[attribute]["mapping"].cuda()

    #             reconstructed = prev_att[prev_att_mapping].clone() 
    #             ungated_xyz_indices = latents[attribute]['ungated_indices']
    #             ungated_residuals = latents[attribute]['ungated_residuals_compressed'].uncompress(scale=10000.0).cuda()
    #             reconstructed[ungated_xyz_indices] += ungated_residuals
                
    #             # Verify reconstruction matches _xyz
    #             # original_xyz = latents[attribute]['_xyz_debug']
    #             # assert torch.allclose(original_xyz, reconstructed, rtol=1e-5, atol=1e-5), "Reconstruction verification failed!"
                
    #             self.mapping = prev_att_mapping
                
    #             self._latents[attribute] = nn.Parameter(reconstructed.requires_grad_(False))
    #         else:
    #             if self.prev_atts[attribute] is not None:
    #                 prev_att = self.prev_atts[attribute]
    #             else:
    #                 prev_att = torch.zeros(latents[attribute]["shape"], device="cuda")
    #             remapped_prev_att = prev_att[prev_att_mapping]  

    #             # then we define it based on the decoder type
    #             if latent_decoders_dict[attribute] == "LatentDecoder":
    #                 self.latent_decoders[attribute] = LatentDecoder(**decoder_args[attribute]).cuda()
    #                 self.latent_decoders[attribute].load_state_dict(decoder_state_dict[attribute])
    #                 self._latents[attribute] = nn.Parameter(latents[attribute].uncompress().cuda().requires_grad_(False))
                    
    #             elif latent_decoders_dict[attribute] == "DecoderIdentity":
    #                 # Identity decoder (no compression)
    #                 self.latent_decoders[attribute] = DecoderIdentity().cuda()
    #                 self._latents[attribute] = nn.Parameter(latents[attribute].cuda().requires_grad_(False))
                    
    #             elif latent_decoders_dict[attribute] == "LatentDecoderRes":
    #                 # Residual decoder with previous frame reference
    #                 self.latent_decoders[attribute] = LatentDecoderRes(**decoder_args[attribute]).cuda()
                    
    #                 # Extract and initialize with previous frame's decoded attributes
    #                 decoded_att = decoder_state_dict[attribute]['decoded_att'].clone().cuda()
    #                 del decoder_state_dict[attribute]['decoded_att']  # Remove from state dict before loading
                    
    #                 self.latent_decoders[attribute].load_state_dict(decoder_state_dict[attribute])
    #                 self.latent_decoders[attribute].init_decoded(decoded_att)
    #                 self.latent_decoders[attribute].identity = False  # Enable quantized mode
    #                 self.latent_decoders[attribute].frame_idx = self.frame_idx
    #                 self._latents[attribute] = nn.Parameter(latents[attribute].uncompress().cuda().requires_grad_(False))
                    
    #                 # Verify decoder functionality
    #                 test_output = self.latent_decoders[attribute](self._latents[attribute])
    #                 expected_dim = decoder_args[attribute]['feature_dim']
    #                 if test_output.shape[-1] != expected_dim:
    #                     print(f"Warning: {attribute} decoder output shape {test_output.shape} != expected {expected_dim}")
        
    #     self.active_sh_degree = self.max_sh_degree

    def decode_latents(self):
        with torch.no_grad():
            for param in self.param_names:
                if not isinstance(self.latent_decoders[param], DecoderIdentity):
                    decoded = self.latent_decoders[param](self._latents[param])
                    if param == "f_rest":
                        self._latents[param].data = decoded.reshape(self._latents[param].shape[0], 
                                                                                 (self.max_sh_degree + 1) ** 2 - 1, 3)
                    elif param == "f_dc":
                        self._latents[param].data = decoded.reshape(self._latents[param].shape[0], 1, 3)
                    else:
                        self._latents[param].data = decoded
                    self.latent_decoders[param] = DecoderIdentity()

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        if type(self.latent_decoders["op"]) == LatentDecoderRes:
            opacities_new = self.latent_decoders["op"].invert(opacities_new-self.latent_decoders["op"].decoded_att)
        else:
            opacities_new = self.latent_decoders["op"].invert(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "op")
        self._latents["op"] = optimizable_tensors["op"]

    def load_ply(self, path, verbose=False):
        plydata = PlyData.read(path)
        if verbose:
            print(f"GaussianModel::load_ply(): loaded gaussian from ply file at: {path}")

        # positions and opacities
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # SH dc (0-freq) values
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # SH "ac" values
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        expected_extra_f_dim = 3 * (self.max_sh_degree + 1) ** 2 - 3
        if len(extra_f_names) == expected_extra_f_dim:
            if verbose:
                print(f"GaussianModel::load_ply(): parsed SH extra f dim matches expectation: {expected_extra_f_dim}")
        elif len(extra_f_names) > expected_extra_f_dim:
            if verbose:
                print(f"GaussianModel::load_ply(): parsed SH extra f dim ({len(extra_f_names)}) exceeds expectation ({expected_extra_f_dim}).")
        else:
            raise RuntimeError(f"GaussianModel::load_ply(): parsed SH extra f dim ({len(extra_f_names)}) does not reach expectation ({expected_extra_f_dim})")

        features_extra = np.zeros((xyz.shape[0], expected_extra_f_dim))
        for idx, attr_name in enumerate(extra_f_names):
            # if provided features (extra_f_names) have higher dim than what we need (expected_extra_f_dim), ignore them.
            if idx < expected_extra_f_dim:
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P, F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # scales
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # rotations
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # initialize the attributes
        self._latents["anchor"] = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._latents["offset"] = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._latents["f_dc"] = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._latents["f_rest"] = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._latents["rot"] = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._latents["sc"] = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._latents["op"] = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))

        for param in self.param_names:
            self.latent_decoders[param] = DecoderIdentity()  # by doing so, we initialize the Gaussian with no decoding.
            if hasattr(self,"gate_params"):
                self.gate_params[param] = False
        self.gate_atts = None

        self.active_sh_degree = self.max_sh_degree

        # extra stuff
        self.max_radii2D = torch.zeros((self.get_offset.shape[0]), device="cuda")
        self.mask_anchor.data = torch.ones_like(self._opacity).bool()
        self.mask_offset.data = torch.ones_like(self._opacity).bool()
        self.mask_features_dc.data = torch.ones_like(self._opacity).bool()
        self.mask_features_rest.data= torch.ones_like(self._opacity).bool()
        self.mask_scaling.data = torch.ones_like(self._opacity).bool()
        self.mask_rotation.data = torch.ones_like(self._opacity).bool()
        self.mask_opacity.data = torch.ones_like(self._opacity).bool()
        if verbose:
            print(f"GaussianModel::load_ply(): initialized Gaussian attributes from: {path}")

    def replace_tensor_to_optimizer(self, tensor, name, lr=None):
        optimizable_tensors = {}
        assert "ldec" not in name, "Latent decoder params cannot be replaced!"
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                if lr is not None:
                    group['lr'] = lr
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
                    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        masks = self.get_masks
        for group in self.optimizer.param_groups:
            if "ldec" in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

            
            masks[group['name']].data = torch.cat((masks[group['name']],
                                              torch.ones(extension_tensor.shape[0],1).bool().to(masks[group['name']].device)), dim=0)

        return optimizable_tensors
        
    def densify_dynamic(self, iteration, opt):
        # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1], 计算每个offset的梯度范数，识别训练充分的点
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
        
        self.anchor_growing(iteration, grads_norm, opt.densify_grad_threshold, opt.update_ratio, opt.extra_ratio, opt.extra_up, offset_mask, opt.overlap)
        
        # update offset_denom, 重置已生长点的训练统计量，让它们重新开始积累
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # prune anchors, 锚点修剪
        prune_mask = (self.opacity_accum < opt.min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N]
        # scale_mask = torch.logical_or(torch.prod(self.get_scaling[:,:3], dim=1) < 1e-8, torch.prod(self.get_scaling[:,:3], dim=1) > 5)
        scale_mask = torch.prod(self.get_scaling[:,:3], dim=1) < 1e-8
        prune_mask = torch.logical_or(prune_mask, scale_mask)

        # update offset_denom, 从所有数据结构中移除被修剪的锚点, 所有统计量保持相同的锚点索引
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)   # 最终从模型中物理删除被标记的锚点。

    def add_densification_stats(self, render_pkg, width, height):
        viewspace_point_tensor = render_pkg["viewspace_points"]
        update_filter = render_pkg["visibility_filter"]
        anchor_visible_mask = render_pkg["visible_mask"]
        offset_selection_mask = render_pkg["selection_mask"]
        opacity = render_pkg["opacity"]
        # update opacity stats
        
        temp_opacity = torch.zeros(offset_selection_mask.shape[0], dtype=torch.float32, device="cuda")
        temp_opacity[offset_selection_mask] = opacity.clone().view(-1).detach()
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        grad_norm = torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        # self.offset_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # self.denom[update_filter] += 1

    def add_influence_stats(self, infl_tensor):
        self.infl_accum += infl_tensor
        self.infl_denom += 1

    @torch.no_grad()
    def influence_prune(self, infl_threshold):
        out = self.infl_accum/self.infl_denom
        out[out.isnan()] = 0.0
        prune_mask = out<=infl_threshold
            
        self.prune_points(prune_mask)
        new_mapping = self.mapping[~prune_mask]
        if new_mapping.max() >= self.offset_before.shape[0]:
            new_mapping[new_mapping >= self.offset_before.shape[0]] = new_mapping[new_mapping[new_mapping >= self.offset_before.shape[0]]]
        self.mapping = new_mapping
        self.infl_accum *= 0
        self.infl_denom *= 0

    # def copy(self):
    #     """Create a deep copy of the GaussianModel instance.
        
    #     Returns:
    #         GaussianModel: A new instance with copied attributes and parameters.
    #     """
    #     # Create new instance with same initialization parameters
    #     new_model = GaussianModel(self.max_sh_degree, self.latent_args, self.model_args, self.frame_idx, self.use_offset_legacy)
        
    #     # Copy basic attributes
    #     new_model.active_sh_degree = self.active_sh_degree
    #     new_model.spatial_lr_scale = self.spatial_lr_scale
    #     new_model.percent_dense = self.percent_dense
        
    #     # Copy latents
    #     for param_name in self.param_names:
    #         new_model._latents[param_name] = nn.Parameter(self._latents[param_name].data.clone().requires_grad_(True))
        
    #     # Copy decoder state dicts
    #     atts = self.get_atts # All the latent variables
    #     for i, att_name in enumerate(atts):
    #         if not isinstance(self.latent_decoders[param_name], DecoderIdentity):
    #             # first check if the decoder is a DecoderIdentity
    #             if isinstance(new_model.latent_decoders[param_name], DecoderIdentity):
    #                 decoder = LatentDecoderRes(
    #                     latent_dim=self.latent_args.latent_dim[i],
    #                     feature_dim=self.feature_dims[att_name],
    #                     ldecode_matrix=self.latent_args.ldecode_matrix[i],
    #                     latent_norm=self.latent_args.latent_norm[i],
    #                     num_layers_dec=self.latent_args.num_layers_dec[i],
    #                     hidden_dim_dec=self.latent_args.hidden_dim_dec[i],
    #                     activation=self.latent_args.activation[i],
    #                     use_shift=self.latent_args.use_shift[i],
    #                     ldec_std=self.latent_args.ldec_std[i],
    #                     final_activation=self.latent_args.final_activation[i],
    #                 ).cuda()
    #                 new_model.latent_decoders[att_name] = decoder
    #             else: 
    #                 new_model.latent_decoders[param_name].load_state_dict(
    #                     self.latent_decoders[param_name].state_dict()
    #                 )
        
    #     # Copy masks
    #     new_model.mask_offset.data = self.mask_offset.data.clone()
    #     new_model.mask_features_dc.data = self.mask_features_dc.data.clone()
    #     new_model.mask_features_rest.data = self.mask_features_rest.data.clone()
    #     new_model.mask_scaling.data = self.mask_scaling.data.clone()
    #     new_model.mask_rotation.data = self.mask_rotation.data.clone()
    #     new_model.mask_opacity.data = self.mask_opacity.data.clone()
    #     new_model.mask_flow.data = self.mask_flow.data.clone()
        
    #     # Copy freeze states
    #     new_model.frz_offset = self.frz_offset
    #     new_model.frz_features_dc = self.frz_features_dc
    #     new_model.frz_features_rest = self.frz_features_rest
    #     new_model.frz_scaling = self.frz_scaling
    #     new_model.frz_rotation = self.frz_rotation
    #     new_model.frz_opacity = self.frz_opacity
    #     new_model.frz_flow = self.frz_flow
        
    #     # Copy previous attributes and latents
    #     for param_name in self.param_names:
    #         if self.prev_atts[param_name] is not None:
    #             new_model.prev_atts[param_name] = self.prev_atts[param_name].clone()
    #         if self.prev_latents[param_name] is not None:
    #             new_model.prev_latents[param_name] = self.prev_latents[param_name].clone()
        
    #     # Copy gate attributes if they exist
    #     if self.gate_atts is not None:
    #         new_model.gate_atts = self.gate_atts.copy()
    #         new_model.gate_params = self.gate_params.copy()
        
    #     # Copy other tensors
    #     new_model.max_radii2D = self.max_radii2D.clone()
    #     new_model.offset_gradient_accum = self.offset_gradient_accum.clone()
    #     new_model.infl_accum = self.infl_accum.clone()
    #     new_model.denom = self.denom.clone()
    #     new_model.infl_denom = self.infl_denom.clone()
        
    #     # Copy mapping and xyz_before if they exist and are not None
    #     if hasattr(self, 'mapping') and self.mapping is not None:
    #         new_model.mapping = self.mapping.clone()
    #     if hasattr(self, 'xyz_before') and self.offset_before is not None:
    #         new_model.offset_before = self.offset_before.clone()
        
    #     # Copy added_mask if it exists and is not None
    #     if hasattr(self, 'added_mask') and self.added_mask is not None:
    #         new_model.added_mask = self.added_mask.clone()
        
    #     # Copy init_probs if it exists and is not None
    #     if hasattr(self, 'init_probs') and self.init_probs is not None:
    #         new_model.init_probs = self.init_probs.clone()
        
    #     return new_model

# OCTREE_LoD---------------------------------------------------------------------------------------------------------------------
    @property
    def get_level(self):
        return self._level    

    @property
    def get_root_indices(self):
        return self.root_indices    
    
    def set_level(self, points, cameras, scales):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, self.dist_ratio)
                dist_min = torch.quantile(dist, 1 - self.dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, self.dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - self.dist_ratio)
        self.standard_dist = dist_max
        if self.levels == -1:
            # self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
            self.levels = torch.round(log_base(dist_max/dist_min, self.log_base)).int().item() + 1
        if self.init_level == -1:
            self.init_level = int(self.levels/2)

    def octree_sample(self, points, colors):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self.colors = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda()
        self.root_indices = torch.empty(0).long().cuda()    # 用于存储每个体素对应的第一层体素索引
        self.root_coord_to_idx = {}                         # 创建快速查找字典（只在CPU，用于建立映射）

        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)                  # 计算当前层级的体素大小
            new_candidates = torch.floor((points - self.init_pos) / cur_size).int()     # 将点坐标归一化到以init_pos为原点的坐标系，并将坐标转换为体素索引。 
            new_candidates_unique, inverse_indices = torch.unique(new_candidates, return_inverse=True, dim=0)   # 找到所有唯一的体素索引，并记录每个原始点对应到唯一体素的映射关系
            # 关键优化：在第一层建立坐标到索引的映射
            if cur_level == 0:
                # 第一层直接使用体素坐标作为唯一标识
                root_voxel_coords = new_candidates_unique
                # 为每个第一层体素分配唯一索引 (0, 1, 2, ...)
                root_indices = torch.arange(root_voxel_coords.shape[0], device=new_candidates_unique.device)
                for idx, coord in enumerate(root_voxel_coords.cpu()):
                    self.root_coord_to_idx[tuple(coord.numpy())] = idx
                # 当前层级的根层索引就是自身
                current_root_indices = root_indices
            else:
                # 对于更深层级：将当前坐标转换到第一层坐标系
                root_scale_factor = float(self.fork) ** cur_level  # 当前层到根层的缩放因子
                root_coords = torch.floor(new_candidates_unique / root_scale_factor).int()
                # 批量查找根层索引（避免循环）
                current_root_indices = torch.zeros(new_candidates_unique.shape[0], 
                                                dtype=torch.long, device=new_candidates_unique.device)
                # 使用向量化操作（比循环快得多）
                root_coords_cpu = root_coords.cpu()
                for i, coord in enumerate(root_coords_cpu):
                    coord_tuple = tuple(coord.numpy())
                    current_root_indices[i] = self.root_coord_to_idx.get(coord_tuple, -1)            
            
            new_positions = new_candidates_unique * cur_size + self.init_pos    # 将体素索引转换回实际坐标
            new_positions += self.padding * cur_size                            # 添加padding偏移  
            new_levels = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level         # 为每个新体素标记当前层级
            new_colors = scatter_max(colors, inverse_indices.unsqueeze(1).expand(-1, colors.size(1)), dim=0)[0] # 将落入同一体素的所有点的颜色进行聚合
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self.colors = torch.concat((self.colors, new_colors), dim=0)
            self._level = torch.concat((self._level, new_levels), dim=0)
            self.root_indices = torch.cat((self.root_indices, current_root_indices), dim=0)                     # 保存根层索引到对象中 

        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def weed_out(self, gaussian_positions, gaussian_levels, root_indices):
        visible_count = torch.zeros(gaussian_positions.shape[0], dtype=torch.int, device="cuda")    # 为每个高斯元素创建一个计数器，记录它在多少个相机视角下"需要被渲染"。
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((gaussian_positions - cam_center)**2, dim=1)) * scale       # 计算每个高斯元素到相机中心的欧氏距离, 乘以scale进行距离缩放
            # pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)                   # 根据距离计算理论上应该使用的细节层级 
            pred_level = log_base(self.standard_dist/dist, self.log_base)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)                          # 将连续的预测层级映射到离散的整数层级
            visible_count += (gaussian_levels <= int_level).int()           # 判断可见性, 如果高斯元素的实际层级 <= 相机视角预测的理想层级, 说明该高斯元素在这个视角下需要被渲染
        visible_count = visible_count/len(self.cam_infos)                   # 将计数转换为比例（在多少比例的视角下需要渲染）
        weed_mask = (visible_count > self.visible_threshold)                # 创建筛选掩码：只有可见比例超过阈值的高斯元素被保留
        mean_visible = torch.mean(visible_count)
        return gaussian_positions[weed_mask], gaussian_levels[weed_mask], mean_visible, weed_mask, root_indices[weed_mask]

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    def set_coarse_interval(self, opt_lod):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/opt_lod.coarse_factor
            a1 = opt_lod.coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_anchor_mask(self, cam_center, iteration, gaussians, resolution_scale):
        dist = torch.sqrt(torch.sum((self._anchor - cam_center)**2, dim=1)) * resolution_scale      # 计算每个锚点到相机的欧几里得距离
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level   # 预测锚点LoD级别
        
        if self.progressive:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level  # 渐进式模式: 根据训练迭代次数动态增加细节级别
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        int_level = torch.ones(dist.shape, dtype=torch.int32, device=pred_level.device) * 4         # 渲染特定LoD的图像时添加这一行
        # if gaussians.frame_idx == 1:
        #     self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)     
        # else:
        self._anchor_mask = self._level.squeeze(dim=1).bool() + True      

    def generate_lod_gaussians(self, viewpoint_camera, visible_mask=None):
        ## view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]
        grid_offsets = self.get_offset[visible_mask]
        scaling = self.get_scaling[visible_mask]
        opacity = self.get_opacity[visible_mask]
        rotation = self.get_rotation[visible_mask]
        color = self.get_features[visible_mask]

        if self.dist2level=="progressive":
            prog = self._prog_ratio[visible_mask]
            transition_mask = self.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            opacity = opacity * prog

        # offsets
        offsets = grid_offsets.view([-1, 3]) * scaling[:,:3]
        scaling = scaling[:,3:] 
        
        xyz = anchor + offsets 
        mask = torch.ones(xyz.shape[0], dtype=torch.bool, device="cuda")

        return xyz, color, opacity, scaling, rotation, mask

    def generate_lod_gaussians_from_load(self, viewpoint_camera, visible_mask=None):
        ## view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device = self.get_anchor.device)

        anchor = self.get_anchor[visible_mask]
        offsets = self.get_offset[visible_mask]
        scaling = self.get_scaling[visible_mask]
        opacity = self.get_opacity[visible_mask]
        rotation = self.get_rotation[visible_mask]
        color = self.get_features[visible_mask]

        if self.dist2level=="progressive":
            prog = self._prog_ratio[visible_mask]
            transition_mask = self.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            opacity = opacity * prog

        scaling = scaling[:,3:] 
        
        xyz = offsets 
        mask = torch.ones(xyz.shape[0], dtype=torch.bool, device="cuda")

        return xyz, color, opacity, scaling, rotation, mask
    
    def run_densify(self, iteration, opt):
        # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1], 计算每个offset的梯度范数，识别训练充分的点
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > opt.update_interval * opt.success_threshold * 0.5).squeeze(dim=1)
        
        self.anchor_growing(iteration, grads_norm, opt.densify_grad_threshold, opt.update_ratio, opt.extra_ratio, opt.extra_up, offset_mask, opt.overlap)
        
        # update offset_denom, 重置已生长点的训练统计量，让它们重新开始积累
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # prune anchors, 锚点修剪
        prune_mask = (self.opacity_accum < opt.min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > opt.update_interval * opt.success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N]
        scale_mask = torch.prod(self.get_scaling[:,:3], dim=1) < 1e-8
        prune_mask = torch.logical_or(prune_mask, scale_mask)

        # update offset_denom, 从所有数据结构中移除被修剪的锚点, 所有统计量保持相同的锚点索引
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)   # 最终从模型中物理删除被标记的锚点。


    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask, overlap):
        init_length = self.get_anchor.shape[0]                                      # 记录初始锚点数: 用于后续处理新增锚点
        grads[~offset_mask] = 0.0                                                   # 掩码无效梯度: 只考虑有效训练点的梯度
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):                                        # 遍历所有LoD层级: 从粗糙到精细
            update_value = self.fork ** update_ratio                                # 计算更新系数: 基于层级深度
            level_mask = (self.get_level == cur_level).squeeze(dim=1)               # 创建层级掩码: 选择当前层级和下一层级的锚点
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue                                                            # 阈值计算与候选点识别
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)            # 计算体素大小: 随层级细化而减小
            ds_size = cur_size / self.fork                                          
            # update threshold, 动态阈值: 高层级使用更严格的阈值 
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold, 三级候选策略: candidate_mask在当前层级添加, candidate_ds_mask在下一层级添加（更高分辨率), candidate_extra_mask额外添加点
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)
            
            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            # 应用层级过滤; 限制在当前层级: 确保只处理属于当前层级的点
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            candidate_extra_mask = torch.logical_and(candidate_extra_mask, level_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:      # 渐进式训练控制: 只在特定迭代后启用额外细化
                self._extra_level += extra_up * candidate_extra_mask.float()    
            # 坐标转换与体素化: 计算所有点的世界坐标; 体素坐标转换, 将连续坐标离散化为网格坐标; 选择候选点坐标
            all_xyz = self.get_anchor + self.get_offset * self.get_scaling[:,:3]
            grid_coords = torch.floor((self.get_anchor[level_mask]-self.init_pos)/cur_size - self.padding).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.floor((selected_xyz-self.init_pos)/cur_size - self.padding).int()
            # 去重处理: 查找唯一网格位置, 避免在同一位置重复添加点; 重叠检测, 检查新点是否与现有点冲突; 杂草剔除, 移除不合适的新点
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if overlap:
                remove_duplicates = torch.ones(selected_grid_coords_unique.shape[0], dtype=torch.bool, device="cuda")
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                new_indices = scatter_max(self.get_root_indices[candidate_mask], inverse_indices, dim=0)[0][remove_duplicates]
                candidate_anchor, new_level, _, weed_mask, new_indices = self.weed_out(candidate_anchor, new_level, new_indices)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            elif selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size + self.init_pos + self.padding * cur_size
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                new_indices = scatter_max(self.get_root_indices[candidate_mask], inverse_indices, dim=0)[0][remove_duplicates]
                candidate_anchor, new_level, _, weed_mask, new_indices = self.weed_out(candidate_anchor, new_level, new_indices)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')
                new_indices = torch.zeros([0], dtype=torch.long, device='cuda')
            # 选择下一层级候选点坐标，并做去重处理
            grid_coords_ds = torch.floor((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size-self.padding).int()
            selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
            selected_grid_coords_ds = torch.floor((selected_xyz_ds-self.init_pos)/ds_size-self.padding).int()
            selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:    # 只在渐进训练后期且非最细层级时启用
                if overlap:
                    remove_duplicates_ds = torch.ones(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device="cuda")
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    new_indices_ds = scatter_max(self.get_root_indices[candidate_ds_mask], inverse_indices_ds, dim=0)[0][remove_duplicates_ds]
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask, new_indices_ds = self.weed_out(candidate_anchor_ds, new_level_ds, new_indices_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                elif selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos+self.padding*ds_size
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    new_indices_ds = scatter_max(self.get_root_indices[candidate_ds_mask], inverse_indices_ds, dim=0)[0][remove_duplicates_ds]
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask, new_indices_ds = self.weed_out(candidate_anchor_ds, new_level_ds, new_indices_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
                    new_indices_ds = torch.zeros([0], dtype=torch.long, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.zeros(selected_grid_coords_unique_ds.shape[0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
                new_indices_ds = torch.zeros([0], dtype=torch.long, device='cuda')
            # 参数初始化
            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()
                new_indices = torch.cat([new_indices, new_indices_ds]).long().cuda()
                
                new_features_dc_1 = self._latents["f_dc"][candidate_mask]
                new_features_dc_1 = scatter_max(new_features_dc_1, inverse_indices.unsqueeze(1).expand(-1, new_features_dc_1.size(1)), dim=0)[0][remove_duplicates]
                new_features_dc_2 = self._latents["f_dc"][candidate_ds_mask]
                new_features_dc_2 = scatter_max(new_features_dc_2, inverse_indices_ds.unsqueeze(1).expand(-1, new_features_dc_2.size(1)), dim=0)[0][remove_duplicates_ds]
                new_features_dc = torch.cat([new_features_dc_1, new_features_dc_2], dim=0)
                new_features_rest_1 = self._latents["f_rest"][candidate_mask]
                new_features_rest_1 = scatter_max(new_features_rest_1, inverse_indices.unsqueeze(1).expand(-1, new_features_rest_1.size(1)), dim=0)[0][remove_duplicates]
                new_features_rest_2 = self._latents["f_rest"][candidate_ds_mask]
                new_features_rest_2 = scatter_max(new_features_rest_2, inverse_indices_ds.unsqueeze(1).expand(-1, new_features_rest_2.size(1)), dim=0)[0][remove_duplicates_ds]
                new_features_rest = torch.cat([new_features_rest_1, new_features_rest_2], dim=0)
                
                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities_ds = inverse_sigmoid(0.1 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda"))                
                new_opacities = torch.cat([new_opacities, new_opacities_ds], dim=0)
                
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size # *0.05
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)
                
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_offsets = torch.zeros_like(candidate_anchor).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])
                
                d = {
                    "anchor": new_anchor,
                    "sc": new_scaling,
                    "rot": new_rotation,
                    "f_dc": new_features_dc,
                    "f_rest": new_features_rest,
                    "offset": new_offsets,
                    "op": new_opacities
                }   

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._latents["anchor"] = optimizable_tensors["anchor"]
                self._latents["sc"] = optimizable_tensors["sc"]
                self._latents["rot"] = optimizable_tensors["rot"]
                self._latents["f_dc"] = optimizable_tensors["f_dc"]
                self._latents["f_rest"] = optimizable_tensors["f_rest"]
                self._latents["offset"] = optimizable_tensors["offset"]
                self._latents["op"] = optimizable_tensors["op"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
                self.root_indices = torch.cat([self.root_indices, new_indices], dim=0)

    def prune_anchor(self, mask):
        valid_points_mask = ~mask
        
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._latents["anchor"] = optimizable_tensors["anchor"]
        self._latents["offset"] = optimizable_tensors["offset"]
        self._latents["f_dc"] = optimizable_tensors["f_dc"]
        self._latents["f_rest"] = optimizable_tensors["f_rest"]
        self._latents["op"] = optimizable_tensors["op"]
        self._latents["sc"] = optimizable_tensors["sc"]
        self._latents["rot"] = optimizable_tensors["rot"]
        self._level = self._level[valid_points_mask]
        self._extra_level = self._extra_level[valid_points_mask]
        self.root_indices = self.root_indices[valid_points_mask]

        for mask_name, mask in self.get_masks.items():
            mask.data = mask[valid_points_mask]

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "ldec" in group["name"]:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, num_overlap=1, use_chunk=True):
        counts = torch.zeros(selected_grid_coords_unique.shape[0], dtype=torch.int, device=selected_grid_coords_unique.device)

        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            for i in range(max_iters):
                chunk = grid_coords[i * chunk_size:(i + 1) * chunk_size]
                matches = (selected_grid_coords_unique.unsqueeze(1) == chunk.unsqueeze(0)).all(-1)
                counts += matches.sum(dim=1)
        else:
            matches = (selected_grid_coords_unique.unsqueeze(1) == grid_coords.unsqueeze(0)).all(-1)
            counts = matches.sum(dim=1)

        remove_duplicates = counts >= num_overlap

        return remove_duplicates