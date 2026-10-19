# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from typing import List, Dict, Any, Optional

from vggt.models.aggregator_vis import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead

from vggt.utils.attention_cache import AttentionCacheManager


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        
        # 初始化attention缓存管理器
        self.attention_cache = AttentionCacheManager()
        
        # 设置aggregator的缓存管理器引用
        self.aggregator.set_attention_cache_manager(self.attention_cache)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions
    
    # ==================== Attention缓存管理接口 ====================
    
    def enable_attention_caching(self, target_layers: List[str] = None,
                                 max_heads: Optional[int] = None):
        """
        启用attention缓存用于可视化
        
        Args:
            target_layers: 要缓存的层名称列表，如['global_block_20', 'global_block_21']
                          默认仅缓存global_block_23
            max_heads: None averages all heads; a positive integer averages
                       the first K heads. The cache stores this mean on CPU.
        """
        self.attention_cache.enable_caching(target_layers, max_heads)
    
    def disable_attention_caching(self):
        """禁用attention缓存"""
        self.attention_cache.disable_caching()
    
    def get_cached_attention(self) -> Dict[str, Any]:
        """
        获取缓存的attention数据
        
        Returns:
            包含所有缓存层attention数据的字典
        """
        return self.attention_cache.get_all_cached_attention()
    
    def extract_patch_attention(self, source_img_idx: int, patch_y: int, patch_x: int,
                              layer_name: str = 'global_block_23',
                              batch_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        从缓存中提取指定patch的attention数据（替代原hook方式）
        
        Args:
            source_img_idx: 源图像索引
            patch_y, patch_x: 源patch坐标
            layer_name: 目标层名称
            
        Returns:
            包含patch attention数据的字典，与原attn_util.extract_single_patch_attention_simple兼容
        """
        return self.attention_cache.extract_patch_attention(
            source_img_idx, patch_y, patch_x, layer_name, batch_idx=batch_idx
        )
    
    def clear_attention_cache(self):
        """清空attention缓存"""
        self.attention_cache.clear_cache()
    
    def get_attention_cache_info(self) -> Dict[str, Any]:
        """获取attention缓存状态信息"""
        return self.attention_cache.get_cache_info()
    
    def start_new_attention_run(self, run_id: str = None):
        """
        开始新的attention缓存运行
        通常在处理新的图像序列前调用
        
        Args:
            run_id: 可选的运行ID，用于标识这次运行
        """
        self.attention_cache.start_new_run(run_id)
