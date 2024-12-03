#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from itertools import chain
from typing import Callable, List, Tuple, Dict
from copy import deepcopy
import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from typing import Optional, Union
from lerobot.common.policies.transformer_diffusion.configuration import TransformerDiffusionConfig as TDConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
import lerobot.common.policies.transformer_diffusion.tinydiffp as dfp
from lerobot.common.policies.utils import populate_queues



class TransformerDiffusionPolicy(nn.Module, PyTorchModelHubMixin):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    name = "act"

    def __init__(
        self,
        config: Optional[TDConfig] = None,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = TDConfig()
        self.config: TDConfig = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.model = TransformerDiffusion(config)
        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        _, loss_dict = self.model(batch)
        return loss_dict


class TransformerDiffusion(nn.Module):
    def __init__(self, config: TDConfig):
        super().__init__()
        self.config = config
        self.use_input_state = "observation.state" in config.input_shapes
        num_images = len([k for k in config.input_shapes if k.startswith("observation.image")])
        if num_images > 0:
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                cond_dim = encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                cond_dim = self.rgb_encoder.feature_dim * num_images

        self.noise_scheduler = dfp.DDPMScheduler(
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            beta_start=0.0001,
            clip_sample=True,
            num_train_timesteps=config.num_train_steps,
            prediction_type='epsilon', #'epsilon',
            variance_type='fixed_small'
        )
        self.num_inference_steps = config.num_inference_steps

        self.diff_model = dfp.TransformerForDiffusion(
            input_dim=config.output_shapes["action"][0],
            output_dim=config.output_shapes["action"][0],
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
            cond_dim=cond_dim,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            n_emb=config.n_emb,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            causal_attn=config.causal_attn,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=0
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.rgb_encoder.parameters(), self.diff_model.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tuple[None, None]]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:

        {
            "observation.state": (B, state_dim) batch of robot states.
            "observation.images": (B, n_cameras, C, H, W) batch of images.
            "action" (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.training:
            assert (
                "action" in batch
            ), "actions must be provided when using the variational objective in training mode."

        batch_size, n_obs_steps = batch["observation.images"].shape[:2]
        device = batch["observation.images"].device
        dtype = batch["observation.images"].dtype
        if self.config.use_separate_rgb_encoder_per_camera:
            # Combine batch and sequence dims while rearranging to make the camera index dimension first.
            images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
            img_features_list = torch.cat(
                [
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
            )
            # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )# output [B, To, encoder_dim]
        else:
            # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
            img_features = self.rgb_encoder(
                einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
            )
            # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )# output [B, To, encoder_dim]
        encoder_out = img_features
        actions, loss_dict = None, {}

        if self.training:
            trajectory = batch["action"]
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            bsz = trajectory.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device
            ).long()
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
            pred = self.diff_model(noisy_trajectory, timesteps, encoder_out)
            pred_type = self.noise_scheduler.config.prediction_type # epsilon
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = trajectory
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}") 
            loss_dict["loss"] = F.mse_loss(pred, target, reduction='mean')
        else:
            trajectory = torch.randn(size=(batch_size, self.config.horizon, self.config.output_shapes["action"][0]), dtype=dtype, device=device)
            trajectory = dfp.conditional_sample(self.diff_model, self.noise_scheduler, trajectory, encoder_out, num_inference_steps=self.num_inference_steps)
            actions = trajectory
        
        return actions, loss_dict

class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: TDConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.input_shapes` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.input_shapes`.
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: we have a check in the config class to make sure all images have the same shape.
        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
        )
        dummy_input = torch.zeros(size=(1, config.input_shapes[image_key][0], *dummy_input_h_w))
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)
        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module

class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb