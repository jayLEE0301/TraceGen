import logging
import math
import os
import sys
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    DDIMScheduler,
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from einops import rearrange

logger = logging.getLogger(__name__)

video_diffusion_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'video_diffusion')
if video_diffusion_path not in sys.path:
    sys.path.append(video_diffusion_path)         


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    scale_factor_spatial: int,
    num_frames: int,
    transformer_config,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare rotary positional embeddings matching diffusers implementation."""

    grid_height = height // (scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (scale_factor_spatial * transformer_config.patch_size)

    p = transformer_config.patch_size
    p_t = getattr(transformer_config, 'patch_size_t', None)

    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p

    if p_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + p_t - 1) // p_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
            device=device,
        )

    return freqs_cos, freqs_sin
    

class CogVideoXDecoder_flow(nn.Module):
    """CogVideoX-based video decoder.
    
    Integrates CogVideoX transformer for video generation with gradient flow through
    encoder_hidden_states conditioning from the transformer trunk.
    
    Supports both training from scratch and using pretrained CogVideoX models.
    """
    
    def __init__(
        self,
        latent_dim: int = 768,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 30,
        num_frames: int = 13,
        frame_size: int = 32,  # Size of latent frames
        patch_size: int = 2,
        patch_size_t: int = 4,
        max_text_seq_length: int = 226,
        text_embed_dim: int = 4096,
        use_rotary_positional_embeddings: bool = True,
        enable_encoder_hidden_states_grad: bool = True,
        torch_dtype: str = "float16",
        scale_factor: float = 0.7,
        scale_factor_spatial: int = 8,
        scale_factor_temporal: int = 4,
        put_frames_in_channels: int = 1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.frame_size = frame_size
        self.scale_factor = scale_factor
        self.scale_factor_spatial = scale_factor_spatial
        self.scale_factor_temporal = scale_factor_temporal
        assert num_frames % put_frames_in_channels == 0, "num_frames must be divisible by put_frames_in_channels"
        self.put_frames_in_channels = put_frames_in_channels
        self.input_channels = in_channels * put_frames_in_channels
        self.output_channels = out_channels * put_frames_in_channels
        self.num_frames = num_frames // put_frames_in_channels
        # breakpoint()
        # self.device = device

        # Convert string dtype to torch dtype
        if torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.float16
        
        # Initialize custom CogVideoX transformer from scratch
        self._setup_custom_model(
            num_attention_heads, attention_head_dim, self.input_channels, self.output_channels,
            num_layers, frame_size, patch_size, patch_size_t, max_text_seq_length,
            text_embed_dim, use_rotary_positional_embeddings,
            enable_encoder_hidden_states_grad
        )
        
        # Project trunk output to CogVideoX text embedding dimension
        # Use direct projection to avoid bottlenecking and better initialization
        if latent_dim != text_embed_dim:
            self.trunk_to_text_proj = nn.Linear(latent_dim, text_embed_dim)
        
            # Initialize with small weights to preserve input scale
            with torch.no_grad():
                # Initialize to approximate identity (preserve input variance)
                nn.init.normal_(self.trunk_to_text_proj.weight, mean=0.0, std=0.01)
                if self.trunk_to_text_proj.bias is not None:
                    nn.init.zeros_(self.trunk_to_text_proj.bias)
        else:
            self.trunk_to_text_proj = nn.Identity()
        
        # Create learnable padding tokens instead of using zeros
        # This prevents the conditioning from being overwhelmed by zeros
        self.learnable_padding_tokens = nn.Parameter(
            torch.randn(1, max_text_seq_length, text_embed_dim) * 0.01
        )
        
        # Output dimension for compatibility
        self.out_dim = out_channels

    def _setup_custom_model(
        self, num_attention_heads, attention_head_dim, in_channels,
        out_channels, num_layers, frame_size, patch_size, patch_size_t,
        max_text_seq_length, text_embed_dim, use_rotary_positional_embeddings,
        enable_encoder_hidden_states_grad
    ):
        """Setup custom CogVideoX model from scratch."""
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Get rank for logging
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        logger.info(f"Rank {rank}: Starting CogVideoX initialization...")
        
        import sys
        import os
        video_diffusion_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'video_diffusion')
        if video_diffusion_path not in sys.path:
            sys.path.append(video_diffusion_path)
        
        logger.info(f"Rank {rank}: Importing CogVideoXTransformer3DModel...")
        from models.decoder.cogvideox_transformer_3d import CogVideoXTransformer3DModel
        
        logger.info(f"Rank {rank}: Creating transformer instance...")
        
        # DON'T pass device parameter - let DDP handle it
        total_token_num = frame_size / patch_size * frame_size / patch_size * self.num_frames / patch_size_t
        print(f"Total token number: {total_token_num}")
        self.cogvideox = CogVideoXTransformer3DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            sample_height=frame_size,
            sample_width=frame_size,
            sample_frames=self.num_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            max_text_seq_length=max_text_seq_length,
            text_embed_dim=text_embed_dim,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            enable_encoder_hidden_states_grad=enable_encoder_hidden_states_grad,
            # DON'T PASS DEVICE HERE
        )
        
        logger.info(f"Rank {rank}: CogVideoX transformer created successfully")

    def normalize_act_data(self, data):
        data = rearrange(data, 'b t c h w -> b h w t c')
        # Use data's device directly
        device = data.device
        data = (data - self.data_act_bias.to(device)) / self.data_act_scale.to(device)
        data = torch.clamp(data, -1, 1)
        return rearrange(data, 'b h w t c -> b t c h w')

    def unnormalize_act_data(self, data):
        data = rearrange(data, 'b t c h w -> b h w t c')
        # Use data's device directly
        device = data.device
        data = data * self.data_act_scale.to(device) + self.data_act_bias.to(device)
        return rearrange(data, 'b h w t c -> b t c h w')

    def set_data_act_statistics(self, max, min):
        self.data_act_scale = (max - min) / 2
        self.data_act_bias = (max + min) / 2

    def project_latents_to_cogvideox_format(self, latents: torch.Tensor):
        encoder_hidden_states = self.trunk_to_text_proj(latents).to(self.cogvideox.dtype)

        # Ensure encoder_hidden_states has the right sequence length
        if encoder_hidden_states.shape[1] != self.cogvideox.config.max_text_seq_length:
            # Pad or truncate to match expected sequence length
            current_seq_len = encoder_hidden_states.shape[1]
            target_seq_len = self.cogvideox.config.max_text_seq_length
            
            if current_seq_len < target_seq_len:
                # first, repeat the encoder_hidden_states to match the target_seq_len, and then, pad the rest with learnable padding tokens
                # Use learnable padding tokens instead of zeros
                batch_size = encoder_hidden_states.shape[0]
                repeat_factor = target_seq_len // current_seq_len
                encoder_hidden_states = encoder_hidden_states.repeat(1, repeat_factor, 1)
                num_padding = target_seq_len - encoder_hidden_states.shape[1]
                # Expand learnable padding tokens for batch size
                padding_tokens = self.learnable_padding_tokens[:, :num_padding, :].expand(
                    batch_size, num_padding, -1
                ).to(encoder_hidden_states.device, encoder_hidden_states.dtype)
                
                encoder_hidden_states = torch.cat([encoder_hidden_states, padding_tokens], dim=1)
            else:
                # Truncate
                encoder_hidden_states = encoder_hidden_states[:, :target_seq_len]

        return encoder_hidden_states

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        trunk_conditioning: torch.Tensor,
        inference_with_negative_prompt: bool = False,
        **kwargs
    ):
        """Forward pass through CogVideoX decoder.
        
        Args:
            noisy_latents: Noisy video latents [B, C, T, H, W]
            timesteps: Diffusion timesteps [B]
            trunk_conditioning: Conditioning from transformer trunk [B, seq_len, latent_dim]
            
        Returns:
            Dictionary containing predicted noise/velocity
        """
        # breakpoint()
        batch_size, channels, num_frames, height, width = noisy_latents.shape
        device = noisy_latents.device
        
        # Handle put_frames_in_channels conversion
        if hasattr(self, 'put_frames_in_channels') and self.put_frames_in_channels > 1:
            # Check if input is already converted (from forward_diffusion_training)
            # Original format: [B, C, T, H, W] where C=3, T=16
            # Converted format: [B, C*put_frames_in_channels, T//put_frames_in_channels, H, W] where C*put_frames_in_channels=12, T//put_frames_in_channels=4
            expected_converted_channels = 3 * self.put_frames_in_channels  # 3 * 4 = 12

            # breakpoint()
            
            if channels == expected_converted_channels:
                # Input is already converted, use as-is
                hidden_states = noisy_latents
            else:
                logger.warning(f"forward function expect the {expected_converted_channels} channels as default. You may useing this function without converting the input channels.")
                # Input is in original format, need to convert
                assert num_frames % self.put_frames_in_channels == 0, f"num_frames ({num_frames}) must be divisible by put_frames_in_channels ({self.put_frames_in_channels})"
                
                # Reshape to group adjacent frames: [B, C, T, H, W] -> [B, C, T//put_frames_in_channels, put_frames_in_channels, H, W]
                grouped_frames = noisy_latents.view(batch_size, channels, num_frames // self.put_frames_in_channels, self.put_frames_in_channels, height, width)
                
                # Move put_frames_in_channels to channel dimension: [B, C*put_frames_in_channels, T//put_frames_in_channels, H, W]
                hidden_states = grouped_frames.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                    batch_size, channels * self.put_frames_in_channels, num_frames // self.put_frames_in_channels, height, width
                )
        else:
            # No frame-to-channel conversion, use original format
            hidden_states = noisy_latents
        
        # Convert to CogVideoX format: [B, T, C, H, W]
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        
        # Project trunk conditioning to CogVideoX text embedding dimension
        encoder_hidden_states = self.project_latents_to_cogvideox_format(trunk_conditioning)
        
        # Prepare rotary embeddings if needed
        image_rotary_emb = None
        if hasattr(self.cogvideox.config, 'use_rotary_positional_embeddings') and self.cogvideox.config.use_rotary_positional_embeddings:
            # Handle patch_size_t parameter correctly
            if hasattr(self.cogvideox.config, 'patch_size_t') and self.cogvideox.config.patch_size_t is not None and self.cogvideox.config.patch_size_t > 1:
                # CogVideoX 1.5: Use compressed temporal dimension
                image_rotary_emb = prepare_rotary_positional_embeddings(
                    height=height * self.scale_factor_spatial,
                    width=width * self.scale_factor_spatial,
                    scale_factor_spatial=self.scale_factor_spatial,
                    num_frames=num_frames,  # Use compressed temporal dimension
                    device=device,
                    transformer_config=self.cogvideox.config,
                )
            else:
                # CogVideoX 1.0: Use full temporal dimension
                print(f"CogVideoX 1.0: num_frames={num_frames}")
                image_rotary_emb = prepare_rotary_positional_embeddings(
                    height=height * self.scale_factor_spatial,
                    width=width * self.scale_factor_spatial,
                    scale_factor_spatial=self.scale_factor_spatial,
                    num_frames=num_frames,
                    device=device,
                    transformer_config=self.cogvideox.config,
                )

        model_output = self.cogvideox(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,  # Gradients flow through this!
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        
        # Convert output back to [B, C, T, H, W] format to match expected output
        model_output = model_output.permute(0, 2, 1, 3, 4)
        # Handle reverse put_frames_in_channels conversion
        if hasattr(self, 'put_frames_in_channels') and self.put_frames_in_channels > 1:
            # Convert [B, C*put_frames_in_channels, T//put_frames_in_channels, H, W] back to [B, C, T, H, W]
            batch_size_out, channels_out, num_frames_out, height_out, width_out = model_output.shape
            
            # Check if we need to reverse the conversion
            # If channels_out == 3 * put_frames_in_channels, then we need to reverse
            if channels_out == 3 * self.put_frames_in_channels:
                # Reshape channels back to frames: [B, C*put_frames_in_channels, T//put_frames_in_channels, H, W] -> [B, C, put_frames_in_channels, T//put_frames_in_channels, H, W]
                model_output = model_output.view(batch_size_out, channels_out // self.put_frames_in_channels, self.put_frames_in_channels, num_frames_out, height_out, width_out)
                
                # Move put_frames_in_channels back to temporal dimension: [B, C, T, H, W]
                model_output = model_output.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                    batch_size_out, channels_out // self.put_frames_in_channels, num_frames_out * self.put_frames_in_channels, height_out, width_out
                )
        
        return {'video': model_output}

    def forward_diffusion_training(
        self,
        trunk_conditioning: torch.Tensor,
        target_video: torch.Tensor,
        diffusion_loss,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Handle diffusion training with CogVideoX-specific preprocessing.
        
        Args:
            trunk_conditioning: Conditioning tokens from trunk [B, seq_len (704), latent_dim (768)]
            target_video: Target video frames [B, 400 (H,W), 17 (T), 2 (C)] or [B, V, T, C, H, W] 
            
        Returns:
            Dictionary containing decoder outputs, noise, and latent targets
        """
        # Accept any incoming seq length; it will be adjusted downstream
        expected_seq_len = getattr(self.cogvideox.config, "max_text_seq_length", trunk_conditioning.shape[-2])
        if trunk_conditioning.shape[-2] != expected_seq_len:
            # WARNING: log the trunk_conditioning and expected_seq_len
            logger.warning(f"Expected sequence length {expected_seq_len} does not match actual sequence length {trunk_conditioning.shape[-2]}")
        # breakpoint()
        target_video = target_video[:, :, 1:, :]  # we don't need first keypoint, since it's always same across all the dataset

        # reshape target_video to [B, 20, 20, 16, 3]
        target_video = target_video.reshape(target_video.shape[0], int(math.sqrt(target_video.shape[1])), int(math.sqrt(target_video.shape[1])), target_video.shape[2], 3)
        target_video = rearrange(target_video, 'b h w t c -> b t c h w')
        target_video = self.normalize_act_data(target_video)
        device = target_video.device
        
        B, T, C, H, W = target_video.shape
        effective_batch_size = B

        # latent_video = self.get_latents_from_video(target_video, reshape_height=reshape_height, reshape_width=reshape_width).to(self.cogvideox.dtype)
        
        # just use target video as latent video for now
        latent_video = rearrange(target_video, 'b t c h w -> b c t h w')
        
        # Handle put_frames_in_channels conversion for training
        if hasattr(self, 'put_frames_in_channels') and self.put_frames_in_channels > 1:
            # Convert [B, C, T, H, W] to [B, C*put_frames_in_channels, T//put_frames_in_channels, H, W]
            B, C, T, H, W = latent_video.shape
            assert T % self.put_frames_in_channels == 0, f"T ({T}) must be divisible by put_frames_in_channels ({self.put_frames_in_channels})"
            
            # Reshape to group frames: [B, C, T, H, W] -> [B, C, T//put_frames_in_channels, put_frames_in_channels, H, W]
            grouped_frames = latent_video.view(B, C, T // self.put_frames_in_channels, self.put_frames_in_channels, H, W)
            
            # Move put_frames_in_channels to channel dimension: [B, C*put_frames_in_channels, T//put_frames_in_channels, H, W]
            latent_video = grouped_frames.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                B, C * self.put_frames_in_channels, T // self.put_frames_in_channels, H, W
            )
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, diffusion_loss.noise_scheduler.config.num_train_timesteps,
            (effective_batch_size,), device=device
        ).long()
        
        # Generate noise only for future frames
        noise_future = torch.randn_like(latent_video)

        # Add noise only to future frames
        t_float = timesteps / diffusion_loss.noise_scheduler.config.num_train_timesteps
        # t_float=t_float.repeat(1, latent_video.shape[1], 1)
        t_float = t_float.view(-1, 1, 1, 1, 1)  # Reshape to [B, 1, 1, 1, 1] for broadcasting

        xt = (1 - t_float) * latent_video + t_float * noise_future
        dx_dt_true = -latent_video + noise_future

        # Forward through CogVideoX decoder
        dx_dt_pred = self(xt, timesteps, trunk_conditioning)['video']

        # revert the shape of dx_dt_true to match dx_dt_pred
        if hasattr(self, 'put_frames_in_channels') and self.put_frames_in_channels > 1:
            # dx_dt_true is in converted format [B, C*put_frames_in_channels, T//put_frames_in_channels, H, W]
            # Need to convert it back to original format [B, C, T, H, W] to match dx_dt_pred
            B_true, C_true, T_true, H_true, W_true = dx_dt_true.shape
            
            # Reshape channels back to frames: [B, C*put_frames_in_channels, T//put_frames_in_channels, H, W] -> [B, C, put_frames_in_channels, T//put_frames_in_channels, H, W]
            dx_dt_true = dx_dt_true.view(B_true, C_true // self.put_frames_in_channels, self.put_frames_in_channels, T_true, H_true, W_true)
            
            # Move put_frames_in_channels back to temporal dimension: [B, C, T, H, W]
            dx_dt_true = dx_dt_true.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                B_true, C_true // self.put_frames_in_channels, T_true * self.put_frames_in_channels, H_true, W_true
            )
        
        return {
            'noise_pred': dx_dt_pred,
            'noise_target': dx_dt_true,
            'actions_norm': latent_video
        }

    def predict_trajectory(
        self,
        trunk_conditioning: torch.Tensor,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, DDPMScheduler, DDIMScheduler, CogVideoXDDIMScheduler, CogVideoXDPMScheduler] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate a video from text embedding or prompt with CFG support.
        
        Args:
            text_embedding: Input text embedding [1, seq_len, embed_dim] (optional if prompt provided)
            prompt: Text prompt string (optional if text_embedding provided)
            negative_prompt: Negative prompt for CFG
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            generator: Random generator for reproducibility
        
        Returns:
            Generated video tensor
        """
        expected_seq_len = getattr(self.cogvideox.config, "max_text_seq_length", trunk_conditioning.shape[-2])
        if trunk_conditioning.shape[-2] != expected_seq_len:
            # WARNING: log the trunk_conditioning and expected_seq_len
            logger.warning(f"Expected sequence length {expected_seq_len} does not match actual sequence length {trunk_conditioning.shape[-2]}")
        # Handle prompt encoding with CFG support
        do_classifier_free_guidance = guidance_scale > 1.0
        print(f"Using classifier-free guidance: {do_classifier_free_guidance} (guidance_scale={guidance_scale})")
        
        prompt_embeds = self.project_latents_to_cogvideox_format(trunk_conditioning).to(self.cogvideox.dtype)

        # Prepare encoder hidden states for CFG
        batch_size, seq_len, embed_dim = prompt_embeds.shape
        negative_prompt_embeds = torch.randn(
            batch_size, seq_len, embed_dim, 
            device=prompt_embeds.device, 
            dtype=prompt_embeds.dtype
        ) * 0.1  # Scale down to reasonable range

        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            print(f"CFG: Concatenated embeddings shape: {encoder_hidden_states.shape}")
        else:
            encoder_hidden_states = prompt_embeds
            print(f"No CFG: Using prompt embeddings shape: {encoder_hidden_states.shape}")
        
        # Get transformer's dtype to avoid dtype mismatch
        transformer_dtype = next(self.cogvideox.parameters()).dtype
        
        # Ensure encoder_hidden_states has the correct dtype
        encoder_hidden_states = encoder_hidden_states.to(transformer_dtype)
        
        # Initialize latents with random noise in [B, T, C, H, W] format
        # Note: self.num_frames is already adjusted for put_frames_in_channels in __init__
        latents_shape = (batch_size, self.num_frames, self.input_channels, self.frame_size, self.frame_size)  # divide by 2 because we have first frame latents and future frames latents
        latents = torch.randn(latents_shape, generator=generator, device=trunk_conditioning.device, dtype=transformer_dtype)

        # Scale initial noise by scheduler's sigma
        if hasattr(scheduler, 'init_noise_sigma'):
            latents = latents * scheduler.init_noise_sigma
            # print(f"After noise scaling by {scheduler.init_noise_sigma:.4f}: [{latents.min():.4f}, {latents.max():.4f}]")

        # Set scheduler timesteps
        scheduler.set_timesteps(num_inference_steps, device=trunk_conditioning.device)
        timesteps = scheduler.timesteps
        
        # Prepare rotary embeddings if needed
        image_rotary_emb = None
        
        old_pred_original_sample = None  # Initialize for DPM scheduler

        # breakpoint()
        for i, t in enumerate(timesteps):
            # Prepare latent model input for CFG
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
                
            # Scale model input
            scale_factor = getattr(scheduler, 'sigmas', None)

            # Scale model input (if method is available on scheduler)
            if hasattr(scheduler, 'scale_model_input'):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            else:
                # Flow matching schedulers don't use scale_model_input
                print(f"Scheduler {type(scheduler).__name__} doesn't use scale_model_input, using input as-is")

            # Expand timestep for batch
            timestep = t.expand(latent_model_input.shape[0]).to(transformer_dtype)
            
            # CogVideoX expects [B, T, C, H, W] format directly
            # Predict noise/velocity
            with torch.no_grad():

                noise_pred = self.cogvideox(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0].to(transformer_dtype)
            
            # Perform classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step (matching diffusers CogVideoX implementation)
            if hasattr(scheduler, 'step') and 'old_pred_original_sample' in str(scheduler.step.__code__.co_varnames):
                # DPM scheduler needs special handling for old_pred_original_sample
                latents, old_pred_original_sample = scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    return_dict=False,
                )
            else:
                # Standard scheduler step for DDIM and others
                # latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                dt = 1 / scheduler.config.num_train_timesteps
                latents = latents - noise_pred * dt
            # Ensure latents maintain correct dtype
            latents = latents.to(transformer_dtype)

        # Handle reverse put_frames_in_channels conversion for prediction
        if hasattr(self, 'put_frames_in_channels') and self.put_frames_in_channels > 1:
            # Convert [B, T, C, H, W] back to original frame format
            B, T, C, H, W = latents.shape
            
            # Reshape channels back to frames: [B, T, C, H, W] -> [B, T*put_frames_in_channels, C//put_frames_in_channels, H, W]
            latents = latents.view(B, T, C // self.put_frames_in_channels, self.put_frames_in_channels, H, W)
            latents = latents.permute(0, 1, 3, 2, 4, 5).contiguous().view(
                B, T * self.put_frames_in_channels, C // self.put_frames_in_channels, H, W
            )
        
        latents = self.unnormalize_act_data(latents)
        videos = rearrange(latents, 'b t c h w -> b (h w) t c')
        return videos
    