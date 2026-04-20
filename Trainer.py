"""FasterGS/Trainer.py"""

import json
from copy import deepcopy

import torch
import numpy as np
import torchvision
from PIL import Image

import Framework
from Datasets.Base import BaseDataset, View
from Datasets.utils import BasicPointCloud, apply_background_color,look_at
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FasterGS.Loss import FasterGSLoss
from Methods.FasterGS.utils import enable_expandable_segments, carve
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    DENSIFICATION_START_ITERATION=600,  # while official code states 500, densification actually starts at 600 there
    DENSIFICATION_END_ITERATION=14_900,  # should be set to 24900 when using MCMC; while official code states 15000, densification actually stops at 14900 there
    DENSIFICATION_INTERVAL=100,
    DENSIFICATION_GRAD_THRESHOLD=0.0002,  # only used when USE_MCMC=False
    DENSIFICATION_PERCENT_DENSE=0.01,  # only used when USE_MCMC=False
    USE_MCMC=False,
    MAX_PRIMITIVES=1_000_000,  # only used when USE_MCMC=True
    OPACITY_RESET_INTERVAL=3_000,  # will be skipped when USE_MCMC=True
    EXTRA_OPACITY_RESET_ITERATION=500,  # will be skipped when USE_MCMC=True
    MORTON_ORDERING_INTERVAL=5000,  # lowering to 2500 or 1000 may improve performance when number of Gaussians is high
    MORTON_ORDERING_END_ITERATION=15000,  # should be set to 25000 when using MCMC
    FILTER_3D=Framework.ConfigParameterList(
        USE=False,
        ORIGINAL_FORMULATION=False,  # if True, the original formulation from the Mip-Splatting paper is used
        FILTER_VARIANCE=0.2,
    ),
    USE_RANDOM_BACKGROUND_COLOR=False,  # prevents the model from overfitting to the background color
    MIN_OPACITY_AFTER_TRAINING=1 / 255,
    RANDOM_INITIALIZATION=Framework.ConfigParameterList(
        FORCE=False,  # if True, the point cloud from the dataset will be ignored
        N_POINTS=100_000,  # number of random points to be sampled within the scene bounding box
        ENABLE_CARVING=True,  # removes points that are never in-frustum in any training view
        CARVING_IN_ALL_FRUSTUMS=False,  # removes points not in-frustum in all views
        CARVING_ENFORCE_ALPHA=False,  # removes points that project to a pixel with alpha=0 in any view where the point is in-frustum
    ),
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,  # weight for the per-pixel L1 loss on the rgb image
        LAMBDA_DSSIM=0.2,  # weight for the DSSIM loss on the rgb image
        LAMBDA_OPACITY_REGULARIZATION=0.0,  # should be set to 0.01 when using MCMC
        LAMBDA_SCALE_REGULARIZATION=0.0,  # should be set to 0.01 when using MCMC
    ),
    OPTIMIZER=Framework.ConfigParameterList(
        LEARNING_RATE_MEANS_INIT=0.00016,
        LEARNING_RATE_MEANS_FINAL=0.0000016,
        LEARNING_RATE_MEANS_MAX_STEPS=30_000,
        LEARNING_RATE_SH_COEFFICIENTS_0=0.0025,
        LEARNING_RATE_SH_COEFFICIENTS_REST=0.000125,  # 0.0025 / 20
        LEARNING_RATE_OPACITIES=0.025,  # should be set to 0.05 (old default in official code) when using MCMC
        LEARNING_RATE_SCALES=0.005,
        LEARNING_RATE_ROTATIONS=0.001,
    ),
)

class FasterGSTrainer(GuiTrainer):
    """Defines the trainer for the FasterGS variant."""

    def __init__(self, **kwargs) -> None:
        self.requires_empty_cache = True
        if not Framework.config.TRAINING.GUI.ACTIVATE:
            if enable_expandable_segments():
                self.requires_empty_cache = False
                Logger.log_info('using "expandable_segments:True" with the torch cuda memory allocator')
        super().__init__(**kwargs)
        self.train_sampler = None
        self.eval_sphere_views = []
        self.loss = FasterGSLoss(loss_config=self.LOSS, gaussians=self.model.gaussians)

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)


###
    @pre_training_callback(priority=50)
    @torch.no_grad()
    def add_cameras_sphere(self, _, dataset: 'BaseDataset') -> None:

        n_points = 2000
        target_fill_ratio = 0.9
        base_cam = dataset.default_camera

        # Derive camera radius from the AABB diagonal so the scene spans ~90% of the screen diagonal.
        bbox = dataset.bounding_box
        bbox_diag = torch.linalg.norm(bbox.size).item()
        scene_radius = max(0.5 * bbox_diag, 1.0e-6)

        half_width = 0.5 * float(base_cam.width)
        half_height = 0.5 * float(base_cam.height)
        tan_half_fov_diag = np.sqrt((half_width / float(base_cam.focal_x)) ** 2 + (half_height / float(base_cam.focal_y)) ** 2)
        sphere_radius = scene_radius / max(target_fill_ratio * tan_half_fov_diag, 1.0e-6)
        world_origin = np.zeros(3, dtype=np.float64)

        Logger.log_info(
            f'sphere camera radius set to {sphere_radius:.3f} (bbox diagonal={bbox_diag:.3f}, target_fill={target_fill_ratio:.2f})'
        )
      
        # 1. Vectorized Fibonacci Sphere Generation
        i = np.arange(n_points)
        phi = np.pi * (3. - np.sqrt(5.))
      
        y = 1 - (i / float(n_points - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
      
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
      
        # Scale unit points to the desired sphere radius
        unit_coordinates = np.stack([x, y, z], axis=1)
        camera_positions = unit_coordinates * sphere_radius

        # 2. Camera Setup
        num_train_views = len(dataset.train())

        for i, eye in enumerate(camera_positions):
            # Maak een nieuwe camera aan (lokaal)
            new_cam = deepcopy(base_cam)
            new_cam.shared_settings.far_plane = sphere_radius + scene_radius + 10.0
            new_cam.shared_settings.near_plane = 0.1
            
            # Bereken c2w zoals je al deed
            up = np.array([0, -1, 0], dtype=np.float64)
            if abs(eye[1] / sphere_radius) > 0.9: 
                up = np.array([0, 0, 1], dtype=np.float64)

            c2w = look_at(eye, world_origin, up)

            # Maak de View aan
            view = View(
                camera=new_cam,
                camera_index=-1,
                frame_idx=-1,
                # Offset the index so filenames don't collide
                global_frame_idx=num_train_views + i, 
                c2w=c2w,
                timestamp=0.0
            )
            
            self.eval_sphere_views.append(view)


    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        dataset.train()
        camera_centers = torch.stack([view.position for view in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        Logger.log_info(f'training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None and not self.RANDOM_INITIALIZATION.FORCE:
            point_cloud = dataset.point_cloud
        else:
            samples = torch.rand((self.RANDOM_INITIALIZATION.N_POINTS, 3), dtype=torch.float32, device=Framework.config.GLOBAL.DEFAULT_DEVICE)
            positions = samples * dataset.bounding_box.size + dataset.bounding_box.min
            if self.RANDOM_INITIALIZATION.ENABLE_CARVING:
                positions = carve(positions, dataset, self.RANDOM_INITIALIZATION.CARVING_IN_ALL_FRUSTUMS, self.RANDOM_INITIALIZATION.CARVING_ENFORCE_ALPHA)
            point_cloud = BasicPointCloud(positions)
        self.model.gaussians.initialize_from_point_cloud(point_cloud, self.USE_MCMC)
        self.model.gaussians.training_setup(self, radius)
        if not self.USE_MCMC:
            self.model.gaussians.reset_densification_info()
        if self.FILTER_3D.USE:
            self.model.gaussians.setup_3d_filter(self.FILTER_3D, dataset)

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(priority=100, start_iteration='DENSIFICATION_START_ITERATION', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        if self.USE_MCMC:
            self.model.gaussians.mcmc_densification(min_opacity=0.005, cap_max=self.MAX_PRIMITIVES)
        else:
            self.model.gaussians.adaptive_density_control(self.DENSIFICATION_GRAD_THRESHOLD, 0.005, iteration > self.OPACITY_RESET_INTERVAL)
            if iteration < self.DENSIFICATION_END_ITERATION:
                self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()
        if self.FILTER_3D.USE:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(priority=99, end_iteration='MORTON_ORDERING_END_ITERATION', iteration_stride='MORTON_ORDERING_INTERVAL')
    @torch.no_grad()
    def morton_ordering(self, *_) -> None:
        """Apply morton ordering to all Gaussian parameters and their optimizer states."""
        self.model.gaussians.apply_morton_ordering()

    @training_callback(active='FILTER_3D.USE', priority=95, start_iteration='DENSIFICATION_END_ITERATION', iteration_stride=100)
    @torch.no_grad()
    def recompute_3d_filter(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Recompute 3D filter."""
        if self.DENSIFICATION_END_ITERATION < iteration < self.NUM_ITERATIONS - 100:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(priority=90, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        if not self.USE_MCMC:
            self.model.gaussians.reset_opacities()

    @training_callback(priority=90, start_iteration='EXTRA_OPACITY_RESET_ITERATION', end_iteration='EXTRA_OPACITY_RESET_ITERATION')
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: 'BaseDataset') -> None:
        """Reset opacities one additional time when using a white background."""
        # original implementation only supports black or white background, this is an attempt to make it work with any color
        if not self.USE_MCMC and dataset.default_camera.background_color.sum() != 0.0:
            Logger.log_info('resetting opacities one additional time because using non-black background')
            self.model.gaussians.reset_opacities()

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random view
        view = self.train_sampler.get(dataset=dataset)['view']
        # render
                
        bg_color = torch.rand_like(view.camera.background_color) if self.USE_RANDOM_BACKGROUND_COLOR else view.camera.background_color
        image = self.renderer.render_image_training(
            view=view,
            update_densification_info=not self.USE_MCMC and iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
        )

        
        # calculate loss
        # compose gt with background color if needed  # FIXME: integrate into data model

        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg_color)
        loss = self.loss(image, rgb_gt)
        # backward
        loss.backward()
        # optimizer step
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()
        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds Gaussian count to default Weights & Biases logging."""
        Framework.wandb.log({
            '#Gaussians': self.model.gaussians.means.shape[0]
        }, step=iteration)
        # default logging
        super().log_wandb(iteration, dataset)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, *_) -> None:
        """Clean up after training."""
        n_gaussians = self.model.gaussians.training_cleanup(min_opacity=self.MIN_OPACITY_AFTER_TRAINING)
        Logger.log_info(f'final number of Gaussians: {n_gaussians:,}')
        with open(str(self.output_directory / 'n_gaussians.txt'), 'w') as n_gaussians_file:
            n_gaussians_file.write(
                f'Final number of Gaussians: {n_gaussians:,}\n'
                f'\n'
                f'N_Gaussians:{n_gaussians}'
            )

    @post_training_callback(priority=900)
    @torch.no_grad()
    def save_contribution_per_view(self, _, dataset: 'BaseDataset') -> None:
        self.model.eval()
        dataset.train()
        output_dir = self.output_directory / 'view_contributions'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map voor de renders aanmaken
        image_dir = output_dir / 'renders'
        image_dir.mkdir(exist_ok=True)

        gaussians = self.model.gaussians
        all_views = list(dataset.train()) + self.eval_sphere_views
        
        # --- 1. GAUSSIAN ATLAS ---
        np.savez_compressed(
            str(output_dir / 'gaussians_atlas.npz'),
            means=gaussians.means.detach().float().cpu().numpy(),
            rotations=gaussians.rotations.detach().float().cpu().numpy(),
            scales=gaussians.scales.detach().float().cpu().numpy(),
            opacities=gaussians.opacities.detach().float().cpu().numpy(),
            sh_coefficients_0=gaussians.sh_coefficients_0.detach().float().cpu().numpy(),
            sh_coefficients_rest=gaussians.sh_coefficients_rest.detach().float().cpu().numpy(),
            active_sh_degree=np.array([gaussians.active_sh_degree], dtype=np.int32),
            max_sh_degree=np.array([gaussians.max_sh_degree], dtype=np.int32),
        )

        all_contributions = []
        all_camera_types = []
        all_camera_widths = []
        all_camera_heights = []
        all_camera_focal_x = []
        all_camera_focal_y = []
        all_camera_center_x = []
        all_camera_center_y = []
        all_camera_near_plane = []
        all_camera_far_plane = []
        all_camera_background_colors = []
        all_camera_c2w = []
        all_camera_w2c = []
        all_camera_positions = []
        all_camera_forwards = []
        all_camera_rights = []
        all_camera_ups = []
        all_camera_indices = []
        all_frame_indices = []
        all_global_frame_indices = []
        all_timestamps = []
        all_camera_exif = []
        all_distortion_types = []
        all_distortion_coefficients = []
        all_distortion_undistortion_eps = []
        all_distortion_undistortion_iterations = []

        # --- 2. CAMERA- EN CONTRIBUTION-DATA ---
        for view in all_views:
            outputs = self.renderer.render_image_inference(view=view)
            
            # --- OPSLAAN ALS PNG ---
            rgb_tensor = outputs['rgb']
            if rgb_tensor.shape[0] == 3:
                rgb_tensor = rgb_tensor.permute(1, 2, 0)
            
            rgb_np = (rgb_tensor.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(rgb_np)
            img.save(image_dir / f"render_{view.global_frame_idx:05d}.png")
            # ----------------------

            all_contributions.append(outputs['contribution'].detach().half().cpu().numpy())

            camera = view.camera
            distortion = getattr(camera, 'distortion', None)

            all_camera_types.append(type(camera).__name__)
            all_camera_widths.append(camera.width)
            all_camera_heights.append(camera.height)
            all_camera_focal_x.append(getattr(camera, 'focal_x', np.nan))
            all_camera_focal_y.append(getattr(camera, 'focal_y', np.nan))
            all_camera_center_x.append(getattr(camera, 'center_x', np.nan))
            all_camera_center_y.append(getattr(camera, 'center_y', np.nan))
            all_camera_near_plane.append(camera.near_plane)
            all_camera_far_plane.append(camera.far_plane)
            all_camera_background_colors.append(camera.background_color.detach().float().cpu().numpy())
            all_camera_c2w.append(view.c2w_numpy)
            all_camera_w2c.append(view.w2c_numpy)
            all_camera_positions.append(view.position_numpy)
            all_camera_forwards.append(view.forward_numpy)
            all_camera_rights.append(view.right_numpy)
            all_camera_ups.append(view.up_numpy)
            all_camera_indices.append(view.camera_index)
            all_frame_indices.append(view.frame_idx)
            all_global_frame_indices.append(view.global_frame_idx)
            all_timestamps.append(view.timestamp)
            all_camera_exif.append(json.dumps(view.exif, default=str, ensure_ascii=True))

            if distortion is None:
                all_distortion_types.append('')
                all_distortion_coefficients.append(np.full(8, np.nan, dtype=np.float32))
                all_distortion_undistortion_eps.append(np.nan)
                all_distortion_undistortion_iterations.append(-1)
            else:
                all_distortion_types.append(type(distortion).__name__)
                all_distortion_coefficients.append(np.array([
                    distortion.k1,
                    distortion.k2,
                    distortion.k3,
                    distortion.k4,
                    distortion.k5,
                    distortion.k6,
                    distortion.p1,
                    distortion.p2,
                ], dtype=np.float32))
                all_distortion_undistortion_eps.append(distortion.undistortion_eps)
                all_distortion_undistortion_iterations.append(distortion.undistortion_iterations)

        np.savez_compressed(
            str(output_dir / 'camera_data.npz'),
            contributions=np.stack(all_contributions),
            camera_types=np.array(all_camera_types, dtype='<U64'),
            camera_widths=np.asarray(all_camera_widths, dtype=np.int32),
            camera_heights=np.asarray(all_camera_heights, dtype=np.int32),
            camera_focal_x=np.asarray(all_camera_focal_x, dtype=np.float32),
            camera_focal_y=np.asarray(all_camera_focal_y, dtype=np.float32),
            camera_center_x=np.asarray(all_camera_center_x, dtype=np.float32),
            camera_center_y=np.asarray(all_camera_center_y, dtype=np.float32),
            camera_near_plane=np.asarray(all_camera_near_plane, dtype=np.float32),
            camera_far_plane=np.asarray(all_camera_far_plane, dtype=np.float32),
            camera_background_colors=np.stack(all_camera_background_colors),
            camera_c2w=np.stack(all_camera_c2w),
            camera_w2c=np.stack(all_camera_w2c),
            camera_positions=np.stack(all_camera_positions),
            camera_forwards=np.stack(all_camera_forwards),
            camera_rights=np.stack(all_camera_rights),
            camera_ups=np.stack(all_camera_ups),
            camera_indices=np.asarray(all_camera_indices, dtype=np.int32),
            frame_indices=np.asarray(all_frame_indices, dtype=np.int32),
            global_frame_indices=np.asarray(all_global_frame_indices, dtype=np.int32),
            timestamps=np.asarray(all_timestamps, dtype=np.float32),
            exif_json=np.array(all_camera_exif, dtype=object),
            distortion_types=np.array(all_distortion_types, dtype='<U64'),
            distortion_coefficients=np.stack(all_distortion_coefficients),
            distortion_undistortion_eps=np.asarray(all_distortion_undistortion_eps, dtype=np.float32),
            distortion_undistortion_iterations=np.asarray(all_distortion_undistortion_iterations, dtype=np.int32),
        )

        Logger.log_info(f"Export voltooid naar {output_dir}")