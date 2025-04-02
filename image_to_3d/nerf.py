import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from PIL import Image
import time
import trimesh
from typing import List, Optional, Tuple, Dict, Any
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
import cv2
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing
import torchvision.transforms.functional as TF


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    focal_length: float = 1000.0
    principal_point: Tuple[float, float] = (256.0, 256.0)
    near: float = 0.1
    far: float = 100.0
    resolution: Tuple[int, int] = (512, 512)


class HashEncoding(nn.Module):
    """Instant-NGP hash encoding for faster training."""

    def __init__(
        self,
        num_levels: int = 16,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19,
        feature_dim: int = 2
    ):
        super().__init__()
        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.feature_dim = feature_dim

        # Initialize feature grids
        self.feature_grids = nn.ParameterList([
            nn.Parameter(torch.randn(2**log2_hashmap_size, feature_dim) * 0.1)
            for _ in range(num_levels)
        ])

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
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


class NeRFMLP(nn.Module):
    """Improved NeRF MLP with better architecture and features."""

    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        input_ch: int = 3,
        input_ch_views: int = 3,
        output_ch: int = 4,
        use_hash_encoding: bool = True
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.use_hash_encoding = use_hash_encoding

        # Initialize hash encoding if enabled
        if use_hash_encoding:
            self.hash_encoding = HashEncoding()
            input_ch = self.hash_encoding.num_levels * self.hash_encoding.feature_dim

        # Create MLP layers with skip connections
        self.pts_linears = nn.ModuleList([
            nn.Linear(input_ch, W),
            *[nn.Linear(W, W) for _ in range(D-2)],
            nn.Linear(W, W)
        ])

        # Feature vector
        self.views_linears = nn.ModuleList([
            nn.Linear(input_ch_views + W, W//2)
        ])

        # Output layers
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

        # Normal prediction
        self.normal_linear = nn.Linear(W, 3)

        # Texture prediction
        self.texture_linear = nn.Sequential(
            nn.Linear(W, W//2),
            nn.ReLU(),
            nn.Linear(W//2, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass with improved features."""
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)

        # Apply hash encoding if enabled
        if self.use_hash_encoding:
            h = self.hash_encoding(input_pts)
        else:
            h = input_pts

        # Process points through MLP with skip connections
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

        # Output RGB, alpha, normal, and texture
        rgb = self.rgb_linear(h)
        normal = self.normal_linear(feature)
        texture = self.texture_linear(feature)

        # Combine outputs
        outputs = torch.cat([rgb, alpha, normal, texture], -1)
        return outputs


class NeRFGenerator:
    """Improved NeRF generator with better features and optimizations."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        camera_params: Optional[CameraParams] = None,
        use_hash_encoding: bool = True,
        num_levels: int = 16,
        base_resolution: int = 16,
        log2_hashmap_size: int = 19
    ):
        """Initialize NeRF generator with improved components."""
        self.device = device
        self.camera_params = camera_params or CameraParams()
        self.use_hash_encoding = use_hash_encoding

        # Initialize NeRF model
        self.model = NeRFMLP(
            use_hash_encoding=use_hash_encoding
        ).to(device)

        # Initialize renderer
        self.glctx = dr.RasterizeCudaContext()

        # Initialize optimizer
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)

        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def _render_views(
        self,
        camera_poses: List[torch.Tensor],
        num_samples: int = 64,
        hierarchical: bool = True
    ) -> torch.Tensor:
        """Render views using nvdiffrast for better performance and gradients."""
        images = []
        for pose in camera_poses:
            # Extract mesh for current view
            # Lower resolution for faster rendering
            mesh = self._extract_mesh(resolution=128)

            # Convert mesh to tensors for nvdiffrast
            vertices = torch.from_numpy(mesh.vertices).float().to(self.device)
            faces = torch.from_numpy(mesh.faces).int().to(self.device)
            vertex_colors = torch.from_numpy(
                mesh.vertex_colors).float().to(self.device)
            vertex_normals = torch.from_numpy(
                mesh.vertex_normals).float().to(self.device)

            # Transform vertices to camera space
            vertices = torch.matmul(
                torch.cat([vertices, torch.ones_like(
                    vertices[..., :1])], dim=-1),
                pose.T
            )[..., :3]

            # Compute lighting
            light_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            diffuse = torch.clamp(
                torch.sum(vertex_normals * light_dir, dim=-1, keepdim=True), 0.0, 1.0)
            vertex_colors = vertex_colors * (0.7 + 0.3 * diffuse)

            # Render using nvdiffrast
            rast_out, rast_out_db = dr.rasterize(
                self.glctx,
                vertices.unsqueeze(0),
                faces.unsqueeze(0),
                vertex_colors.unsqueeze(0),
                resolution=self.camera_params.resolution
            )

            # Get rendered image
            image = rast_out[0]
            images.append(image)

        return torch.stack(images)

    def _extract_mesh(
        self,
        threshold: float = 0.5,
        resolution: int = 128  # Lower default resolution
    ) -> trimesh.Trimesh:
        """Extract and optimize mesh from NeRF with UV mapping."""
        # Create grid of points
        x = torch.linspace(-1, 1, resolution, device=self.device)
        y = torch.linspace(-1, 1, resolution, device=self.device)
        z = torch.linspace(-1, 1, resolution, device=self.device)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        points = torch.stack([xx, yy, zz], dim=-1)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(points)
            density = outputs[..., 3]
            rgb = outputs[..., :3]
            normal = outputs[..., 4:7]
            texture = outputs[..., 7:]

        # Apply marching cubes
        vertices, faces = self._marching_cubes(
            density.cpu().numpy(),
            threshold
        )

        # Get vertex attributes
        vertex_colors = self._get_vertex_colors(vertices, rgb)
        vertex_normals = self._get_vertex_normals(vertices, normal)
        vertex_textures = self._get_vertex_textures(vertices, texture)

        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            vertex_normals=vertex_normals
        )

        # Add UV coordinates
        mesh = self._add_uv_coordinates(mesh)

        # Optimize mesh
        mesh = self._optimize_mesh(mesh)

        return mesh

    def _add_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Add UV coordinates to mesh using spherical mapping."""
        # Convert vertices to spherical coordinates
        vertices = mesh.vertices
        r = np.sqrt(np.sum(vertices**2, axis=1))
        theta = np.arctan2(vertices[:, 1], vertices[:, 0])
        phi = np.arccos(vertices[:, 2] / r)

        # Convert to UV coordinates
        u = (theta + np.pi) / (2 * np.pi)
        v = phi / np.pi

        # Add UV coordinates to mesh
        mesh.visual.uv = np.stack([u, v], axis=1)

        return mesh

    def _optimize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Optimize mesh quality using various techniques."""
        # Convert to PyTorch3D mesh
        vertices = torch.from_numpy(mesh.vertices).float().to(self.device)
        faces = torch.from_numpy(mesh.faces).long().to(self.device)
        pytorch3d_mesh = Meshes(
            verts=[vertices],
            faces=[faces]
        )

        # Apply Laplacian smoothing with regularization
        loss = mesh_laplacian_smoothing(pytorch3d_mesh)
        optimizer = torch.optim.Adam([vertices], lr=1e-4)

        for _ in range(50):  # Fewer iterations for faster optimization
            optimizer.zero_grad()
            loss = mesh_laplacian_smoothing(pytorch3d_mesh)
            loss.backward()
            optimizer.step()

        # Convert back to trimesh
        mesh.vertices = vertices.detach().cpu().numpy()

        # Apply PyMeshLab optimization
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

    def _compute_loss(
        self,
        rendered_images: torch.Tensor,
        target_images: List[Image.Image]
    ) -> torch.Tensor:
        """Compute loss between rendered and target images with regularization."""
        # Convert target images to tensor
        target_tensors = []
        for img in target_images:
            target_tensors.append(self.transform(img))
        target_tensors = torch.stack(target_tensors).to(self.device)

        # Compute L1 loss
        l1_loss = F.l1_loss(rendered_images, target_tensors)

        # Compute perceptual loss using VGG
        perceptual_loss = self._compute_perceptual_loss(
            rendered_images, target_tensors
        )

        # Compute regularization loss
        reg_loss = self._compute_regularization_loss()

        # Combine losses
        total_loss = (
            l1_loss +  # Reconstruction loss
            0.1 * perceptual_loss +  # Perceptual loss
            0.01 * reg_loss  # Regularization loss
        )

        return total_loss

    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for smooth geometry."""
        # Get mesh vertices
        vertices = torch.from_numpy(self._extract_mesh(
            resolution=64).vertices).float().to(self.device)

        # Compute Laplacian loss
        laplacian_loss = mesh_laplacian_smoothing(
            Meshes(verts=[vertices], faces=[torch.from_numpy(
                self._extract_mesh(resolution=64).faces).long().to(self.device)])
        )

        # Compute total variation loss
        tv_loss = torch.mean(torch.abs(vertices[..., 1:] - vertices[..., :-1]))

        return laplacian_loss + 0.1 * tv_loss

    def _compute_weights(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute weights for volume rendering."""
        # Compute alpha values
        alpha = 1 - torch.exp(-alpha * 0.01)

        # Compute weights
        weights = alpha * torch.cumprod(1 - alpha + 1e-10, dim=-1)

        return weights

    def _sample_fine(
        self,
        t_coarse: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """Sample fine points based on coarse weights."""
        # Compute PDF
        pdf = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)

        # Sample from PDF
        t_fine = torch.cat([
            t_coarse,
            torch.sort(torch.cat([
                torch.rand_like(t_coarse) * (t_coarse[..., 1:] - t_coarse[..., :-1]) +
                t_coarse[..., :-1]
                for _ in range(num_samples)
            ], dim=-1))[0]
        ], dim=-1)

        return t_fine

    def _get_vertex_colors(
        self,
        vertices: np.ndarray,
        rgb: torch.Tensor
    ) -> np.ndarray:
        """Get colors for mesh vertices."""
        # Convert vertices to tensor
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device)

        # Get colors at vertex positions
        if self.use_hash_encoding:
            features = self.model.hash_encoding(vertices_tensor)
            colors = self.model.rgb_linear(features)
        else:
            colors = self.model.rgb_linear(vertices_tensor)

        return colors.detach().cpu().numpy()

    def _get_vertex_normals(
        self,
        vertices: np.ndarray,
        normal: torch.Tensor
    ) -> np.ndarray:
        """Get normals for mesh vertices."""
        # Convert vertices to tensor
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device)

        # Get normals at vertex positions
        if self.use_hash_encoding:
            features = self.model.hash_encoding(vertices_tensor)
            normals = self.model.normal_linear(features)
        else:
            normals = self.model.normal_linear(vertices_tensor)

        return normals.detach().cpu().numpy()

    def _get_vertex_textures(
        self,
        vertices: np.ndarray,
        texture: torch.Tensor
    ) -> np.ndarray:
        """Get textures for mesh vertices."""
        # Convert vertices to tensor
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device)

        # Get textures at vertex positions
        if self.use_hash_encoding:
            features = self.model.hash_encoding(vertices_tensor)
            textures = self.model.texture_linear(features)
        else:
            textures = self.model.texture_linear(vertices_tensor)

        return textures.detach().cpu().numpy()

    def _marching_cubes(
        self,
        density: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Implement marching cubes algorithm using scikit-image."""
        # Normalize density to [0, 1] range
        density = (density - density.min()) / (density.max() - density.min())

        # Apply marching cubes
        vertices, faces, normals, values = measure.marching_cubes(
            density, threshold, spacing=(1, 1, 1)
        )

        # Center and scale vertices
        vertices = vertices / (density.shape[0] - 1) * 2 - 1

        return vertices, faces

    def generate_3d_model(
        self,
        images: List[Image.Image],
        camera_poses: List[torch.Tensor],
        num_iterations: int = 1000,
        output_dir: Optional[str] = None,
        save_interval: int = 100
    ) -> trimesh.Trimesh:
        """Generate 3D model from images using NeRF."""
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Training loop
        for i in tqdm(range(num_iterations), desc="Training"):
            # Render views
            rendered_images = self._render_views(
                camera_poses,
                num_samples=64,
                hierarchical=True
            )

            # Compute loss
            loss = self._compute_loss(rendered_images, images)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Save intermediate results
            if output_dir and (i + 1) % save_interval == 0:
                # Save rendered images
                for j, img in enumerate(rendered_images):
                    img_pil = Image.fromarray(
                        (img.detach().cpu().numpy() * 255).astype(np.uint8)
                    )
                    img_pil.save(os.path.join(
                        output_dir, f"view_{j:03d}_iter_{i+1:04d}.png"
                    ))

                # Save mesh
                mesh = self._extract_mesh()
                mesh.export(os.path.join(
                    output_dir, f"mesh_iter_{i+1:04d}.ply"
                ))

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

    def _compute_perceptual_loss(
        self,
        rendered_images: torch.Tensor,
        target_images: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss using VGG features."""
        # Load VGG model
        vgg = torchvision.models.vgg16(
            pretrained=True).features[:16].to(self.device)
        vgg.eval()

        # Get features
        with torch.no_grad():
            rendered_features = vgg(rendered_images)
            target_features = vgg(target_images)

        # Compute loss
        loss = F.mse_loss(rendered_features, target_features)

        return loss


def test_nerf():
    """Test the improved NeRF implementation."""
    # Initialize generator
    generator = NeRFGenerator(
        use_hash_encoding=True,
        num_levels=16,
        base_resolution=16
    )

    # Load test images and poses
    images = []
    poses = []
    for i in range(4):
        # Load image
        img = Image.open(f"test_data/image_{i:03d}.png")
        images.append(img)

        # Generate pose
        angle = 2 * np.pi * i / 4
        pose = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 2],
            [0, 0, 0, 1]
        ], device=generator.device)
        poses.append(pose)

    # Generate 3D model
    mesh = generator.generate_3d_model(
        images=images,
        camera_poses=poses,
        num_iterations=100,  # Fewer iterations for testing
        output_dir="output/test_nerf"
    )

    print("Test completed successfully!")


if __name__ == "__main__":
    test_nerf()
