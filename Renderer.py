"""FasterGS/Renderer.py"""

import numpy as np

import torch
import time

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGS.Model import FasterGSModel
from Methods.FasterGS.FasterGSCudaBackend import diff_rasterize, rasterize, RasterizerSettings


def extract_settings(
    view: View,
    active_sh_bases: int,
    bg_color: torch.Tensor,
    proper_antialiasing: bool,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGS renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    return RasterizerSettings(
        view.w2c,
        view.position,
        bg_color,
        active_sh_bases,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
        proper_antialiasing,
    )


@Framework.Configurable.configure(
    SCALE_MODIFIER=1.0,
    PROPER_ANTIALIASING=False,
    FORCE_OPTIMIZED_INFERENCE=False,
)
class FasterGSRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGSModel])
        self.start_time = time.time()
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGS renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGS renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.render_image_benchmark(view, to_chw=to_chw or benchmark)
        elif self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(self, view: View, update_densification_info: bool, bg_color: torch.Tensor) -> torch.Tensor:

        # """Renders an image for a given view."""
        image = diff_rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, bg_color, self.PROPER_ANTIALIASING),
        )
        return image

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view with SV or SG Visibility Filtering."""
    
        # --- OPTIE B: SPHERICAL GAUSSIANS (SG) ---
        if hasattr(self.model.gaussians, 'sg_axis') and self.model.gaussians.sg_axis is not None and False:
            num_sg = min(total_count, self.model.gaussians.sg_axis.shape[0])
            curr_axis = self.model.gaussians.sg_axis[:num_sg]
            curr_sharpness = self.model.gaussians.sg_sharpness[:num_sg]
            curr_amplitude = self.model.gaussians.sg_amplitude[:num_sg]

            # Dot product tussen camera-kijkrichting en SG-lobben
            dot = torch.einsum('d,nld->nl', cam_pos, curr_axis)
            activations = curr_amplitude * torch.exp(curr_sharpness * (dot - 1.0))
            visibility_scores = torch.sum(activations, dim=1)
            
            mask = torch.ones(total_count, dtype=torch.bool, device=device)
            mask[:num_sg] = visibility_scores > 0.001 # SG drempel is vaak lager

        # --- OPTIE C: GEEN FILTERING ---
        elif False:
            mask = torch.ones(total_count, dtype=torch.bool, device=device)

        # --- Culling Statistieken ---
        # num_kept = mask.sum().item()
        # num_culled = total_count - num_kept
        # percent_culled = (num_culled / total_count) * 100

        #print(f"Geculled: {num_culled} ({percent_culled:.2f}%)")
        #print(f"Overgebleven: {num_kept}")


        #self.model.gaussians.num_sites.fill_(0)
        image, contribution = rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.raw_scales,
            rotations=self.model.gaussians.raw_rotations ,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color, self.PROPER_ANTIALIASING),
            to_chw=to_chw,
            sites = self.model.gaussians.sv_sites,
            values = self.model.gaussians.sv_values,
            num_sites = self.model.gaussians.num_sites  
        )

        if not to_chw and image.shape[0] == 3:
            image = image.permute(1, 2, 0).contiguous()
                                    
        return {'rgb': image, 'contribution': contribution}
    
    @torch.inference_mode()
    def render_image_benchmark(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        return self.render_image_inference(view, to_chw=to_chw)

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb']}
