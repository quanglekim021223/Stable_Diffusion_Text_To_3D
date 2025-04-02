import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union, Dict
import time
import trimesh
import nvdiffrast.torch as dr
from tqdm import tqdm
import json
import torchvision.transforms as transforms
from dataclasses import dataclass
from transformers import CLIPProcessor, CLIPModel
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from skimage import measure
import pymeshlab
import torch_scatter


class NeRFMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4):
        super(NeRFMLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch

        # Create MLP layers
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] +
                                         [nn.Linear(W, W) for _ in range(D-1)])

        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W//2)])

        # Feature vector
        self.pts_linears.append(nn.Linear(W, W))

        # Output layers
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # Process points through MLP
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i == self.D-2:
                alpha = self.alpha_linear(h)
                feature = h

        # Process views
        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        # Output RGB
        rgb = self.rgb_linear(h)

        # Combine outputs
        outputs = torch.cat([rgb, alpha], -1)
        return outputs


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    focal_length: float = 1000.0
    principal_point: Tuple[float, float] = (256.0, 256.0)
    near: float = 0.1
    far: float = 100.0
    resolution: Tuple[int, int] = (512, 512)


class VoxelGrid(nn.Module):
    """Optimized voxel grid using Instant-NGP encoding."""

    def __init__(
        self,
        resolution: int = 128,
        feature_dim: int = 32,
        num_levels: int = 16,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        desired_resolution: Optional[int] = None
    ):
        super().__init__()
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.desired_resolution = desired_resolution or resolution

        # Initialize feature grids
        self.feature_grids = nn.ParameterList([
            nn.Parameter(torch.randn(2**log2_hashmap_size, feature_dim) * 0.1)
            for _ in range(num_levels)
        ])

        # Initialize density grid
        self.density_grid = nn.Parameter(torch.randn(
            resolution, resolution, resolution) * 0.1)

        # Initialize color MLP
        self.color_mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def get_features(self, xyz: torch.Tensor) -> torch.Tensor:
        """Get features for given 3D coordinates using multi-resolution hash encoding."""
        features = []
        for level in range(self.num_levels):
            # Calculate grid resolution for this level
            grid_res = self.base_resolution * (2 ** level)

            # Get grid indices
            xyz_scaled = xyz * grid_res
            indices = xyz_scaled.long()

            # Get feature indices using spatial hashing
            hash_indices = self._spatial_hash(indices, level)

            # Get features
            features.append(self.feature_grids[level][hash_indices])

        # Combine features from all levels
        return torch.cat(features, dim=-1)

    def _spatial_hash(self, indices: torch.Tensor, level: int) -> torch.Tensor:
        """Compute spatial hash for given indices."""
        primes = [1, 2654435761, 805459861]
        hash_indices = torch.zeros_like(indices[..., 0], dtype=torch.long)
        for i, prime in enumerate(primes):
            hash_indices = hash_indices ^ (indices[..., i] * prime)
        return hash_indices % (2 ** self.log2_hashmap_size)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get density and color."""
        # Get features
        features = self.get_features(xyz)

        # Get density
        density = F.sigmoid(self.density_grid)

        # Get color
        color = self.color_mlp(features)

        return density, color


class DreamFusionGenerator:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        resolution: int = 128,
        feature_dim: int = 32,
        num_levels: int = 16,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        camera_params: Optional[CameraParams] = None,
        sd_model_name: str = "runwayml/stable-diffusion-v1-5"
    ):
        """Initialize DreamFusion generator with improved components."""
        self.device = device
        self.camera_params = camera_params or CameraParams()

        # Initialize voxel grid
        self.voxel_grid = VoxelGrid(
            resolution=resolution,
            feature_dim=feature_dim,
            num_levels=num_levels,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size
        ).to(device)

        # Initialize renderer
        self.glctx = dr.RasterizeCudaContext()

        # Initialize CLIP for better loss
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32").to(device)

        # Initialize Stable Diffusion components
        self._init_sd_model(sd_model_name)

        # Initialize optimizers
        self.optimizer = torch.optim.Adam(
            self.voxel_grid.parameters(), lr=1e-4)

        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def _init_sd_model(self, model_name: str):
        """Initialize Stable Diffusion components."""
        # Load components
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet").to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae").to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler")

        # Freeze models
        for model in [self.text_encoder, self.unet, self.vae]:
            for param in model.parameters():
                param.requires_grad = False

    def _render_views(
        self,
        camera_poses: List[torch.Tensor],
        num_samples: int = 64
    ) -> torch.Tensor:
        """Render views using direct voxel grid rendering."""
        images = []
        for pose in camera_poses:
            # Generate rays
            rays_o, rays_d = self._generate_rays(pose)

            # Sample points along rays
            t = torch.linspace(0, 1, num_samples, device=self.device)
            points = rays_o.unsqueeze(-1) + rays_d.unsqueeze(-1) * \
                t.unsqueeze(0).unsqueeze(0)

            # Get density and color
            density, color = self.voxel_grid(points)

            # Compute alpha values
            alpha = 1 - torch.exp(-density * 0.01)

            # Compute weights for volume rendering
            weights = alpha * torch.cumprod(1 - alpha + 1e-10, dim=-1)

            # Render image
            image = torch.sum(weights.unsqueeze(-1) * color, dim=-2)
            images.append(image)

        return torch.stack(images)

    def _generate_rays(self, camera_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for rendering."""
        # Get camera parameters
        focal = self.camera_params.focal_length
        pp = torch.tensor(self.camera_params.principal_point,
                          device=self.device)
        H, W = self.camera_params.resolution

        # Generate pixel coordinates
        y, x = torch.meshgrid(
            torch.linspace(0, H-1, H, device=self.device),
            torch.linspace(0, W-1, W, device=self.device),
            indexing='ij'
        )

        # Convert to camera space
        x = (x - pp[0]) / focal
        y = (y - pp[1]) / focal
        z = torch.ones_like(x)

        # Create ray directions
        rays_d = torch.stack([x, y, z], dim=-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        # Transform rays to world space
        rays_d = torch.matmul(
            camera_pose[:3, :3], rays_d.unsqueeze(-1)
        ).squeeze(-1)
        rays_o = camera_pose[:3, 3].expand_as(rays_d)

        return rays_o, rays_d

    def _compute_sds_loss(self, images: torch.Tensor, prompt: str) -> torch.Tensor:
        """Compute Score Distillation Sampling loss using Stable Diffusion."""
        # Encode prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]

        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images.permute(
                0, 3, 1, 2)).latent_dist.sample() * 0.18215

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get model prediction
        with torch.no_grad():
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def _optimize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Optimize mesh quality using various techniques."""
        # Remove duplicate vertices
        mesh.remove_duplicate_vertices()

        # Remove degenerate faces
        mesh.remove_degenerate_faces()

        # Remove infinite values
        mesh.remove_infinite_values()

        # Remove unused vertices
        mesh.remove_unreferenced_vertices()

        # Fix winding
        mesh.fix_normals()

        # Smooth mesh
        mesh = self._smooth_mesh(mesh)

        # Optimize topology
        mesh = self._optimize_topology(mesh)

        return mesh

    def _smooth_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply Laplacian smoothing to the mesh with color preservation."""
        # Get adjacency matrix
        adjacency = mesh.vertex_adjacency_graph

        # Compute Laplacian matrix
        laplacian = adjacency - \
            torch.eye(len(mesh.vertices), device=self.device)

        # Apply smoothing to vertices
        vertices = torch.from_numpy(mesh.vertices).float().to(self.device)
        smoothed_vertices = torch.matmul(laplacian, vertices)

        # Update mesh vertices
        mesh.vertices = smoothed_vertices.cpu().numpy()

        # Update vertex colors to match new positions
        mesh.vertex_colors = self._get_vertex_colors(mesh.vertices)

        return mesh

    def _optimize_topology(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Optimize mesh topology using PyMeshLab."""
        # Create PyMeshLab mesh
        ms = pymeshlab.MeshSet()
        ms.add_new_mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces,
            v_normals_matrix=mesh.vertex_normals,
            f_normals_matrix=mesh.face_normals
        )

        # Apply topology optimization
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_unreferenced_vertices()

        # Get optimized mesh
        optimized_mesh = ms.current_mesh()

        # Convert back to trimesh
        return trimesh.Trimesh(
            vertices=optimized_mesh.vertex_matrix(),
            faces=optimized_mesh.face_matrix(),
            vertex_normals=optimized_mesh.vertex_normal_matrix(),
            face_normals=optimized_mesh.face_normal_matrix()
        )

    def _get_vertex_colors(self, vertices: np.ndarray) -> np.ndarray:
        """Get colors for mesh vertices using feature interpolation."""
        # Convert vertices to tensor
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device)

        # Get features and colors
        features = self.voxel_grid.get_features(vertices_tensor)
        colors = self.voxel_grid.color_mlp(features)

        return colors.detach().cpu().numpy()

    def _compute_loss(
        self,
        rendered_images: torch.Tensor,
        prompt: str,
        camera_poses: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute combined loss using CLIP, SDS, and regularization."""
        # CLIP loss for better detail control
        clip_loss = self._compute_clip_loss(rendered_images, prompt)

        # SDS loss with classifier-free guidance
        sds_loss = self._compute_sds_loss(rendered_images, prompt)

        # Regularization loss for smooth geometry
        reg_loss = self._compute_regularization_loss()

        # Combine losses with weights
        total_loss = (
            clip_loss +  # CLIP loss for detail
            0.1 * sds_loss +  # SDS loss for overall shape
            0.01 * reg_loss  # Regularization for smoothness
        )

        return total_loss

    def _compute_clip_loss(self, images: torch.Tensor, prompt: str) -> torch.Tensor:
        """Compute CLIP loss between images and prompt."""
        # Process images
        processed_images = []
        for img in images:
            img_pil = Image.fromarray(
                (img.cpu().numpy() * 255).astype(np.uint8))
            processed_images.append(self.transform(img_pil))
        processed_images = torch.stack(processed_images).to(self.device)

        # Get image and text embeddings
        image_features = self.clip_model.get_image_features(processed_images)
        text_features = self.clip_model.get_text_features(
            self.clip_processor(
                text=prompt, return_tensors="pt", padding=True).to(self.device)
        )

        # Normalize features
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        # Compute loss
        loss = 1 - (image_features @ text_features.T).mean()

        return loss

    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for smooth geometry."""
        # L1 regularization on density
        density_reg = torch.abs(self.voxel_grid.density_grid).mean()

        # Total variation regularization
        tv_reg = 0
        for i in range(3):
            tv_reg += torch.abs(
                self.voxel_grid.density_grid[..., 1:] -
                self.voxel_grid.density_grid[..., :-1]
            ).mean()
            tv_reg += torch.abs(
                self.voxel_grid.density_grid[..., 1:, :] -
                self.voxel_grid.density_grid[..., :-1, :]
            ).mean()
            tv_reg += torch.abs(
                self.voxel_grid.density_grid[..., 1:, :, :] -
                self.voxel_grid.density_grid[..., :-1, :, :]
            ).mean()

        return density_reg + 0.1 * tv_reg

    def generate_3d_model(
        self,
        prompt: str,
        num_iterations: int = 1000,
        num_views: int = 8,
        output_dir: Optional[str] = None,
        save_interval: int = 100
    ) -> trimesh.Trimesh:
        """Generate 3D model using DreamFusion."""
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Generate camera poses
        camera_poses = self._generate_camera_poses(num_views)

        # Training loop
        for i in tqdm(range(num_iterations), desc="Training"):
            # Render views
            rendered_images = self._render_views(camera_poses)

            # Compute loss
            loss = self._compute_loss(rendered_images, prompt, camera_poses)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Save intermediate results
            if output_dir and (i + 1) % save_interval == 0:
                # Save rendered images
                for j, img in enumerate(rendered_images):
                    img_pil = Image.fromarray(
                        (img.detach().cpu().numpy() * 255).astype(np.uint8))
                    img_pil.save(os.path.join(
                        output_dir, f"view_{j:03d}_iter_{i+1:04d}.png"))

                # Save mesh
                mesh = self._extract_mesh()
                mesh.export(os.path.join(
                    output_dir, f"mesh_iter_{i+1:04d}.ply"))

                # Save training info
                info = {
                    "iteration": i + 1,
                    "loss": loss.item(),
                    "camera_poses": [pose.tolist() for pose in camera_poses]
                }
                with open(os.path.join(output_dir, f"info_iter_{i+1:04d}.json"), "w") as f:
                    json.dump(info, f)

        # Extract final mesh
        final_mesh = self._extract_mesh()

        # Save final results
        if output_dir:
            final_mesh.export(os.path.join(output_dir, "final_mesh.ply"))

        return final_mesh

    def _generate_camera_poses(self, num_views: int) -> List[torch.Tensor]:
        """Generate camera poses around the object."""
        poses = []
        for i in range(num_views):
            # Generate rotation matrix
            angle = 2 * np.pi * i / num_views
            rotation = torch.tensor([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], device=self.device)

            # Generate translation
            translation = torch.tensor([0, 0, 2], device=self.device)

            # Create pose matrix
            pose = torch.eye(4, device=self.device)
            pose[:3, :3] = rotation
            pose[:3, 3] = translation

            poses.append(pose)

        return poses

    def _extract_mesh(self, threshold: float = 0.5) -> trimesh.Trimesh:
        """Extract and optimize mesh from voxel grid."""
        # Get density values
        density = self.voxel_grid.density_grid.detach().cpu().numpy()

        # Create mesh using marching cubes
        vertices, faces = self._marching_cubes(density, threshold)

        # Get vertex colors
        vertex_colors = self._get_vertex_colors(vertices)

        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors
        )

        # Optimize mesh
        mesh = self._optimize_mesh(mesh)

        return mesh

    def _marching_cubes(self, density: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Implement marching cubes algorithm using scikit-image."""
        # Normalize density to [0, 1] range
        density = (density - density.min()) / (density.max() - density.min())

        # Apply marching cubes
        vertices, faces, normals, values = measure.marching_cubes(
            density, threshold, spacing=(1, 1, 1)
        )

        # Center and scale vertices
        vertices = vertices / (self.voxel_grid.resolution - 1) * 2 - 1

        return vertices, faces


class DreamFusion:
    """
    Implementation of DreamFusion-like approach for 3D model generation from text.
    This is a simplified version of the original implementation, focusing on the
    core concepts.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "output/3d_models",
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    ):
        """
        Initialize the DreamFusion model.

        Args:
            device (str): The device to run the model on
            output_dir (str): Directory to save generated 3D models
            pretrained_model_name (str): Pretrained Stable Diffusion model name
        """
        self.device = device
        self.output_dir = output_dir
        self.pretrained_model_name = pretrained_model_name

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load the diffusion pipeline components
        # We only need specific components from Stable Diffusion
        self._init_sd_model()

        # NeRF-related parameters
        # In a full implementation, this would be a more complex 3D representation
        self.voxel_resolution = 128
        self.voxel_grid = None  # Will be initialized during optimization

    def _init_sd_model(self):
        """Initialize the Stable Diffusion components needed for optimization."""
        # Text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name,
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name,
            subfolder="text_encoder"
        ).to(self.device)

        # Diffusion model
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name,
            subfolder="unet"
        ).to(self.device)

        # Scheduler for diffusion process
        self.scheduler = DDIMScheduler.from_pretrained(
            self.pretrained_model_name,
            subfolder="scheduler"
        )

        # VAE for latent space
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name,
            subfolder="vae"
        ).to(self.device)

        # Freeze all models - we only optimize the 3D representation
        for model in [self.text_encoder, self.unet, self.vae]:
            for param in model.parameters():
                param.requires_grad = False

    def _encode_prompt(self, prompt: str):
        """Encode the prompt into text embeddings."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        text_input_ids = text_inputs.input_ids.to(self.device)

        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]

        # For classifier-free guidance, we need uncond embeddings (empty text)
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_input_id = uncond_input.input_ids.to(self.device)

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input_id)[0]

        # Concat the unconditional and text embeddings for guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def _init_voxel_grid(self):
        """Initialize the 3D voxel grid representation."""
        # In a full implementation, we might use a more sophisticated 3D representation
        # like a NeRF MLP or a differentiable volumetric renderer
        # Here we use a simple voxel grid for simplicity
        self.voxel_grid = torch.zeros(
            (1, 4, self.voxel_resolution,
             self.voxel_resolution, self.voxel_resolution),
            device=self.device,
            requires_grad=True
        )
        # Initialize with a simple shape (sphere) for better convergence
        x = torch.linspace(-1, 1, self.voxel_resolution)
        y = torch.linspace(-1, 1, self.voxel_resolution)
        z = torch.linspace(-1, 1, self.voxel_resolution)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        distance = torch.sqrt(xx**2 + yy**2 + zz**2)

        # Create a soft sphere
        sphere = 1.0 - torch.clamp(distance / 0.5, 0, 1)

        # Set initial density to the sphere shape
        self.voxel_grid[0, 3, :, :, :] = sphere
        # Random RGB values
        self.voxel_grid[0, 0:3, :, :, :] = torch.rand(
            3, 1, 1, 1).expand(-1, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution)

        return self.voxel_grid

    def _render_views(self, num_views: int = 4):
        """
        Render multiple views of the 3D model.

        This is a simplified version - in a full implementation 
        we would use a proper differentiable renderer.

        Args:
            num_views (int): Number of views to render

        Returns:
            torch.Tensor: Rendered views as images
        """
        # In a complete implementation, we'd use a proper differentiable renderer
        # Here we'll just create a simple projection as an example
        views = []
        angles = torch.linspace(0, 2 * torch.pi, num_views + 1)[:-1]

        # Simple rotations around y-axis
        for angle in angles:
            # Create rotation matrix
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            rot_matrix = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], device=self.device)

            # Rotate voxel grid (simplified)
            # In a real implementation, we would render through the rotated grid
            # Here we'll just take a simple projection

            # Simple orthographic projection (for demo purposes)
            # This is highly simplified - real implementation would use proper rendering
            density = self.voxel_grid[0, 3:4, :, :, :]
            colors = self.voxel_grid[0, 0:3, :, :, :]

            # Project down a dimension (simplified for demonstration)
            proj_density = torch.max(density, dim=3)[0]  # project along z
            proj_colors = torch.max(
                colors * density.expand_as(colors), dim=3)[0]
            proj_colors = proj_colors / \
                (proj_density.expand_as(proj_colors) + 1e-5)

            # Combine into RGBA image (simplified)
            image = torch.cat([proj_colors, proj_density], dim=0)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(512, 512),
                mode='bilinear',
                align_corners=False
            )[0]

            views.append(image)

        return torch.stack(views)

    def _diffusion_loss(self, rendered_views, text_embeddings, guidance_scale=7.5):
        """
        Calculate the diffusion model loss for the rendered views.

        Args:
            rendered_views (torch.Tensor): Rendered views of the 3D model
            text_embeddings (torch.Tensor): Encoded text embeddings
            guidance_scale (float): Classifier-free guidance scale

        Returns:
            torch.Tensor: The calculated loss
        """
        # This is a simplified version of the Score Distillation Sampling loss
        # from the DreamFusion paper

        # Get only RGB channels and normalize
        views_rgb = rendered_views[:, 0:3].permute(0, 2, 3, 1)  # NCHW -> NHWC
        views_rgb = (views_rgb * 2) - 1  # Scale to [-1, 1]

        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(views_rgb.permute(
                0, 3, 1, 2)).latent_dist.sample() * 0.18215

        # Choose a random noise level
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (latents.shape[0],),
            device=self.device
        ).long()

        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Repeat for conditional and unconditional
        noisy_latents = torch.cat([noisy_latents] * 2)
        timesteps = torch.cat([timesteps] * 2)

        # Get model prediction
        with torch.no_grad():
            noise_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        # Calculate loss (simplified)
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def optimize_3d_model(
        self,
        prompt: str,
        num_iterations: int = 1000,
        learning_rate: float = 0.01,
        save_interval: int = 100,
    ):
        """
        Optimize a 3D model to match the given text prompt.

        Args:
            prompt (str): Text prompt for the desired 3D model
            num_iterations (int): Number of optimization iterations
            learning_rate (float): Learning rate for optimization
            save_interval (int): How often to save intermediate results

        Returns:
            str: Path to the saved 3D model
        """
        print(f"Optimizing 3D model for prompt: '{prompt}'")

        # Encode the text prompt
        text_embeddings = self._encode_prompt(prompt)

        # Initialize the voxel grid
        voxel_grid = self._init_voxel_grid()
        optimizer = Adam([voxel_grid], lr=learning_rate)

        # Optimization loop
        for i in range(num_iterations):
            start_time = time.time()

            # Render views
            rendered_views = self._render_views(num_views=4)

            # Calculate loss
            loss = self._diffusion_loss(rendered_views, text_embeddings)

            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if i % 10 == 0:
                print(
                    f"Iteration {i}/{num_iterations}, Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s")

            # Save intermediate results
            if (i + 1) % save_interval == 0 or i == num_iterations - 1:
                self._save_model(prompt, i + 1)

        # Save final model
        output_path = self._save_model(prompt, num_iterations)
        print(f"3D model optimization completed. Saved to {output_path}")

        return output_path

    def _save_model(self, prompt: str, iteration: int) -> str:
        """
        Save the current 3D model to disk.

        Args:
            prompt (str): The prompt used to generate the model
            iteration (int): Current iteration number

        Returns:
            str: Path to the saved model
        """
        # Create a safe filename from the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
        filename = f"{safe_prompt}_iter{iteration}"

        # Extract mesh from voxel grid (simplified)
        # In a real implementation, we would use more sophisticated methods
        density = self.voxel_grid[0, 3].detach().cpu().numpy()
        colors = self.voxel_grid[0, 0:3].permute(
            1, 2, 3, 0).detach().cpu().numpy()

        # Create a simple mesh using marching cubes (would need to import skimage.measure in real impl)
        # Here we'll create a simple cube mesh for demonstration
        vertices = []
        faces = []

        # Generate a simplified mesh for demonstration
        # In a real implementation, we would use marching cubes
        # threshold = 0.5
        # Simple solution: extract a bounding box where density > threshold

        # Extract object bounding box (simplified)
        x = np.linspace(-1, 1, self.voxel_resolution)
        y = np.linspace(-1, 1, self.voxel_resolution)
        z = np.linspace(-1, 1, self.voxel_resolution)

        # Create a simple mesh (a cube) for demonstration
        # In a real implementation, we'd use marching cubes here
        vertices = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ])
        faces = np.array([
            [0, 1, 3], [0, 3, 2],  # Left
            [4, 6, 7], [4, 7, 5],  # Right
            [0, 4, 5], [0, 5, 1],  # Bottom
            [2, 3, 7], [2, 7, 6],  # Top
            [0, 2, 6], [0, 6, 4],  # Back
            [1, 5, 7], [1, 7, 3]   # Front
        ])

        # Create a mesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Save to output directory
        output_path = os.path.join(self.output_dir, f"{filename}.obj")
        mesh.export(output_path)

        # Also save a rendered view for visualization
        if iteration % 100 == 0:
            self._save_rendered_view(prompt, iteration)

        return output_path

    def _save_rendered_view(self, prompt: str, iteration: int):
        """Save a rendered view of the current 3D model."""
        # Create a safe filename from the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
        filename = f"{safe_prompt}_iter{iteration}_view.png"

        # Render a view
        rendered_view = self._render_views(num_views=1)[0]

        # Convert to PIL Image
        rgb = rendered_view[0:3].permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1) * 255
        alpha = rendered_view[3].detach().cpu().numpy() * 255

        # Create RGBA image
        rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgba[:, :, 0:3] = rgb.astype(np.uint8)
        rgba[:, :, 3] = alpha.astype(np.uint8)

        # Save image
        img = Image.fromarray(rgba, 'RGBA')
        output_path = os.path.join(self.output_dir, filename)
        img.save(output_path)


def test_dreambooth():
    """Test the improved DreamFusionGenerator."""
    # Initialize generator
    generator = DreamFusionGenerator(
        resolution=64,  # Smaller resolution for testing
        feature_dim=16,
        num_levels=8
    )

    # Generate 3D model
    prompt = "A red toy robot with glowing eyes, high detail, studio lighting"
    mesh = generator.generate_3d_model(
        prompt=prompt,
        num_iterations=100,  # Fewer iterations for testing
        num_views=4,
        output_dir="output/test_dreambooth"
    )

    print("Test completed successfully!")


if __name__ == "__main__":
    test_dreambooth()
