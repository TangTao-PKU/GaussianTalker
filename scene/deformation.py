import functools
import math
import os
import time
from tkinter import W

import numpy as np
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from scene.networks import AudioNet, AudioAttNet, MLP
from scene.canonical_tri_plane import canonical_tri_plane
from scene.transformer.transformer import Spatial_Audio_Attention_Module

# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.grid_pe = grid_pe
        
        # audio network
        self.audio_in_dim = 29    
        self.audio_dim = 32
        self.eye_dim = 1
        
        self.no_grid = args.no_grid
        self.tri_plane = canonical_tri_plane(args,self.D, self.W)
        # D layers MLP with W hidden dimention
        
        self.args = args
        self.only_infer = args.only_infer
        self.enc_x = None
        self.aud_ch_att = None
        
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.transformer = Spatial_Audio_Attention_Module(args)
        self.ratio=0
        self.create_net()
        
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)        # audio determine
        self.audio_mlp = MLP(32, args.d_model, 64, 2)
        
        self.in_dim = 32 
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 64, 2)
        self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
        
        
        self.eye_mlp = MLP(32, args.d_model, 64, 2)
        self.cam_mlp = MLP(12, args.d_model, 64, 2)
        self.null_vector = nn.Parameter(torch.randn(1, 1, args.d_model))
        self.enc_x_mlp = MLP(32, args.d_model, 64, 2)
        
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.tri_plane.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))
        
    
    def mlp_init_zeros(self):
        nn.init.xavier_uniform_(self.pos_deform[-1].weight,gain=0.1)
        nn.init.zeros_(self.pos_deform[-1].bias)
        
        nn.init.xavier_uniform_(self.scales_deform[-1].weight,gain=0.1) 
        nn.init.zeros_(self.scales_deform[-1].bias)
        
        nn.init.xavier_uniform_(self.rotations_deform[-1].weight,gain=0.1)
        nn.init.zeros_(self.rotations_deform[-1].bias)
        
        nn.init.xavier_uniform_(self.opacity_deform[-1].weight,gain=0.1)
        nn.init.zeros_(self.opacity_deform[-1].bias)
        
        nn.init.xavier_uniform_(self.shs_deform[-1].weight,gain=0.1)
        nn.init.zeros_(self.shs_deform[-1].bias)
        
        #triplane
        self.tri_plane.mlp_init_zeros()
    def mlp2cpu(self):
        self.tri_plane.mlp2cpu()
        
    def eye_encoding(self,value, d_model=32):    # sin cos position encoding
        batch_size, _ = value.shape  
        value = value.to('cuda')

        positions = torch.arange(d_model, device='cuda').float()
        div_term = torch.pow(10000, (2 * (positions // 2)) / d_model)

        encoded_vec = torch.zeros(batch_size, d_model, device='cuda')
        encoded_vec[:, 0::2] = torch.sin(value * div_term[::2])  
        encoded_vec[:, 1::2] = torch.cos(value * div_term[1::2])

        return encoded_vec
    
    
    def query_audio(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features):
        # audio_features [1, 8, 29, 16]) 
        audio_features = audio_features.squeeze(0)
        enc_a = self.audio_net(audio_features)
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        
        if self.only_infer:
            if self.enc_x == None:
                self.enc_x = self.grid(rays_pts_emb[:,:3]) 
                self.aud_ch_att = self.aud_ch_att_net(self.enc_x)
            enc_x = self.enc_x
            aud_ch_att = self.aud_ch_att
            
        else:
            enc_x = self.grid(rays_pts_emb[:,:3])   # shape ; # , 32
            aud_ch_att = self.aud_ch_att_net(enc_x)

        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        
        enc_w = enc_a * aud_ch_att
        eye_att = self.eye_att_net(enc_x)
        eye_att = torch.sigmoid(eye_att)
        eye_features = eye_features * eye_att
        hidden = torch.cat([enc_x, enc_w, eye_features], dim=-1)            # # , 65(32+32+1)
    
        hidden = self.feature_out(hidden)   
  
        return hidden
    # audio attention
    def attention_query_audio_batch(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features, cam_features):
        # audio_features [B, 8, 29, 16]) 
        B, _, _, _= audio_features.shape
        
        if self.only_infer: #cashing
            if self.enc_x == None:
                self.enc_x = self.tri_plane(rays_pts_emb,only_feature = True, train_tri_plane = self.args.train_tri_plane)
            enc_x = self.enc_x[:B]
        
        else:
            # enc_x: torch.Size([16, 34650, 64])
            enc_x = self.tri_plane(rays_pts_emb,only_feature = True, train_tri_plane = self.args.train_tri_plane)
            
        enc_a_list= []
        for i in range(B):
            enc_a = self.audio_net(audio_features[i])       # 8 32
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # 1 32 
            enc_a_list.append(enc_a.unsqueeze(0))
        
        enc_a = torch.cat(enc_a_list,dim=0)                     # torch.Size([16, 1, 64])
        enc_eye = self.eye_encoding(eye_features).unsqueeze(1)  # torch.Size([16, 1, 64])
        enc_cam = self.cam_mlp(cam_features).unsqueeze(1)       # torch.Size([16, 1, 64])
        
        if self.args.d_model != 32:
            enc_a = self.audio_mlp(enc_a) # B, 1, dim
            enc_eye = self.eye_mlp(enc_eye) # B, 1, dim
        
        enc_source = torch.cat([enc_a,enc_eye, enc_cam, self.null_vector.repeat(B,1,1)],dim = 1)    # torch.Size([16, 4, 64])
        # cross-attention
        # q: enc_x torch.Size([16, 34650, 64])
        # k & v: enc_source torch.Size([16, 4, 64])
        # x : output v
        # attention: output (q @ k_t) / math.sqrt(d_tensor) 
        x, attention = self.transformer(enc_x, enc_source)
        # x: torch.Size([16, 34650, 64])
        # attention: torch.Size([16, 2, 34650, 4])
        return x, attention
    
    def attention_query_audio(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features):
        # audio_features [1, 8, 29, 16])
        if self.only_infer:
            if self.enc_x == None:
                enc_x = self.tri_plane(rays_pts_emb.unsqueeze(0),only_feature = True, train_tri_plane = self.args.train_tri_plane).unsqueeze(0)
            enc_x = self.enc_x
            
        else:
            enc_x = self.tri_plane(rays_pts_emb.unsqueeze(0),only_feature = True, train_tri_plane = self.args.train_tri_plane).unsqueeze(0)
        
        audio_features = audio_features.squeeze(0) # 8, 29, 16
        enc_a = self.audio_net(audio_features)  # 8, 32
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)).unsqueeze(0) # 1, 1, 32
        
        enc_eye = self.eye_encoding(eye_features).unsqueeze(1) # 1, 1, 32
        
        
        if self.args.d_model != 32:
            enc_a = self.audio_mlp(enc_a)
            enc_eye = self.eye_mlp(enc_eye)
            

        enc_source = torch.cat([enc_a,enc_eye,self.null_vector],dim = 1) # 1, 3, dim
        
        x, attention = self.transformer(enc_x, enc_source) # 1, N, dim
        x = x.squeeze()
        
        return x, attention

    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, audio_features=None, eye_features=None,cam_features=None):
        if audio_features is None:
            # coarse stage, only input position
            return self.forward_static(rays_pts_emb)        
        elif len(rays_pts_emb.shape)==3:                    
            ### fine stage, input other featrues
            # rays_pts_emb      torch.Size([16, 34650, 63])
            # scales_emb        torch.Size([16, 34650, 15])
            # rotations_emb     torch.Size([16, 34650, 20])
            # opacity           torch.Size([16, 34650, 1])
            # shs_emb           torch.Size([16, 34650, 16, 3])
            # audio_features    torch.Size([16, 8, 29, 16])
            # eye_features      torch.Size([16, 1])
            # cam_features      torch.Size([16, 12])
            return self.forward_dynamic_batch(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, audio_features, eye_features, cam_features)
        elif len(rays_pts_emb.shape)==2:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, audio_features, eye_features)

    def forward_static(self, rays_pts_emb):
        grid_feature, scale, rotation, opacity, sh = self.tri_plane(rays_pts_emb)
        # dx = self.static_mlp(grid_feature)
        return rays_pts_emb, scale, rotation, opacity, sh.reshape([sh.shape[0],sh.shape[1],16,3])
    
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, audio_features, eye_features):
        hidden, attention = self.attention_query_audio(rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features)
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
            
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            shs = shs_emb*mask.unsqueeze(-1) + dshs
            
        return pts, scales, rotations, opacity, shs, attention
    
    def forward_dynamic_batch(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, audio_features, eye_features, cam_features):
        hidden, attention = self.attention_query_audio_batch(rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features, cam_features)
        # hidden: torch.Size([16, 34650, 64])       score @ v
        # attention: torch.Size([16, 2, 34650, 4])  q @ k_t
        B, _, _, _ = audio_features.shape
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,:,0]).unsqueeze(-1)  # torch.Size([16, 34650, 1])
        # deformation prediction
        if self.args.no_dx:
            pts = rays_pts_emb[:,:,:3]
        else:
            dx = self.pos_deform(hidden)                        # torch.Size([16, 34650, 3])
            pts = torch.zeros_like(rays_pts_emb[:,:,:3])
            pts = rays_pts_emb[:B,:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:,:3]
        else:
            ds = self.scales_deform(hidden)                     # torch.Size([16, 34650, 3])

            scales = torch.zeros_like(scales_emb[:,:,:3])
            scales = scales_emb[:B,:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:,:4]
        else:
            dr = self.rotations_deform(hidden)                  # torch.Size([16, 34650, 4])

            rotations = torch.zeros_like(rotations_emb[:,:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:B,:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:,:1] 
        else:
            do = self.opacity_deform(hidden)                    # torch.Size([16, 34650, 1])
          
            opacity = torch.zeros_like(opacity_emb[:,:,:1])
            opacity = opacity_emb[:B,:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],shs_emb.shape[1],16,3])    
            # torch.Size([16, 34650, 16, 3])
            shs = torch.zeros_like(shs_emb)
            shs = shs_emb[:B]*mask.unsqueeze(-1) + dshs
        
        return pts, scales, rotations, opacity, shs, attention
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        grid_pe = args.grid_pe
        # self.posebase_pe = 10
        # self.scale_rotation_pe = 2
        # self.opacity_pe = 2
        
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, args=args)
        self.only_infer = args.only_infer
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        self.deformation_net.mlp_init_zeros()
        
        self.point_emb = None
        self.scales_emb = None
        self.rotations_emb = None
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, audio_features = None, eye_features=None, cam_features =None):
        if audio_features is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, shs, audio_features, eye_features, cam_features)
        else:
            return self.forward_static(point, scales, rotations, opacity, shs)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points, scales, rotations, opacity, shs):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, audio_features=None, eye_features=None, cam_features=None):
        if self.only_infer:
            if self.point_emb == None:
                self.point_emb = poc_fre(point,self.pos_poc)         # #, 3 -> #, 63
                self.scales_emb = poc_fre(scales,self.rotation_scaling_poc)
                self.rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
                
            point_emb = self.point_emb
            scales_emb = self.scales_emb
            rotations_emb = self.rotations_emb

            means3D, scales, rotations, opacity, shs, attention = self.deformation_net(point_emb,
                                                    scales_emb,
                                                    rotations_emb,
                                                    opacity,
                                                    shs,
                                                    audio_features, eye_features, cam_features)
            return means3D, scales, rotations, opacity, shs, attention

        else:
            point_emb = poc_fre(point,self.pos_poc)                         # B, N, 3 -> B, N, 63 (3*10 + 3*10 + 3)
            scales_emb = poc_fre(scales,self.rotation_scaling_poc)          # B, N, 3 -> B, N, 15 (3*2 + 3*2 + 3)
            rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)    # B, N, 4 -> B, N, 20 (4*2 + 4*2 + 4)
            means3D, scales, rotations, opacity, shs, attention = self.deformation_net(point_emb,
                                                    scales_emb,
                                                    rotations_emb,
                                                    opacity,
                                                    shs,
                                                    audio_features, eye_features, cam_features)
            return means3D, scales, rotations, opacity, shs, attention
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    def mlp2cpu(self):
        self.deformation_net.mlp2cpu()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
def poc_fre(input_data,poc_buf):
    # poc_buf: torch.Size([10]) tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.], device='cuda:0')
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)       # torch.Size([16, 34650, 30])
    input_data_sin = input_data_emb.sin()                                   # torch.Size([16, 34650, 30])
    input_data_cos = input_data_emb.cos()                                   # torch.Size([16, 34650, 30])
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)     # torch.Size([16, 34650, 63])
    return input_data_emb