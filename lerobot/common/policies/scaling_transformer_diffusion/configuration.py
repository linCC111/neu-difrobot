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
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class ScalingTransformerDiffusionConfig:
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - At least one key starting with "observation.image is required as an input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initalize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        temporal_ensemble_momentum: Exponential moving average (EMA) momentum parameter (α) for ensembling
            actions for a given time step over multiple policy invocations. Updates are calculated as:
            x⁻ₙ = αx⁻ₙ₋₁ + (1-α)xₙ. Note that the ACT paper and original ACT code describes a different
            parameter here: they refer to a weighting scheme wᵢ = exp(-m⋅i) and set m = 0.01. With our
            formulation, this is equivalent to α = exp(-0.01) ≈ 0.99. When this parameter is provided, we
            require `n_action_steps == 1` (since we need to query the policy every step anyway).
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 2
    horizon: int = 32
    n_action_steps: int = 16

    input_shapes: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "observation.images.top": [3, 480, 640],
            "observation.state": [14],
        }
    )
    output_shapes: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "action": [14],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: Dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.top": "mean_std",
            "observation.state": "mean_std",
        }
    )
    output_normalization_modes: Dict[str, str] = field(
        default_factory=lambda: {
            "action": "mean_std",
        }
    )

    # Architecture.
    #DDPM.
    num_train_steps: int= 100
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Transformer layers.
    n_heads: int = 8
    n_layers: int = 4
    n_emb: int= 256
    causal_attn: bool= True

    # Inference.
    num_inference_steps: int= 100

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        
        image_keys = {k for k in self.input_shapes if k.startswith("observation.image")}

        if len(image_keys) == 0 and "observation.environment_state" not in self.input_shapes:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if len(image_keys) > 0:
            if self.crop_shape is not None:
                for image_key in image_keys:
                    if (
                        self.crop_shape[0] > self.input_shapes[image_key][1]
                        or self.crop_shape[1] > self.input_shapes[image_key][2]
                    ):
                        raise ValueError(
                            f"`crop_shape` should fit within `input_shapes[{image_key}]`. Got {self.crop_shape} "
                            f"for `crop_shape` and {self.input_shapes[image_key]} for "
                            "`input_shapes[{image_key}]`."
                        )
            # Check that all input images have the same shape.
            first_image_key = next(iter(image_keys))
            for image_key in image_keys:
                if self.input_shapes[image_key] != self.input_shapes[first_image_key]:
                    raise ValueError(
                        f"`input_shapes[{image_key}]` does not match `input_shapes[{first_image_key}]`, but we "
                        "expect all image shapes to match."
                    )