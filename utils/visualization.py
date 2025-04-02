import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh
import open3d as o3d
import plotly.graph_objects as go
from typing import List, Optional, Tuple, Dict, Any, Union
from skimage.metrics import structural_similarity as ssim
import cv2
from dataclasses import dataclass
import json
import torchvision.transforms as T
from scipy.spatial.transform import Rotation


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    image_size: Tuple[int, int] = (800, 800)
    lighting_intensity: float = 1.0
    ambient_light: float = 0.2
    diffuse_light: float = 0.8
    specular_light: float = 0.5
    camera_distance: float = 2.0
    camera_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    colormap: str = "viridis"
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    point_size: float = 0.01
    line_width: float = 1.0
    show_axes: bool = True
    show_grid: bool = True
    show_normals: bool = True
    show_textures: bool = True
    use_plotly: bool = True
    num_views: int = 4  # Number of views to render for 3D models
    texture_resolution: int = 1024  # Resolution for texture maps
    max_vertices: int = 100000  # Maximum number of vertices for efficient rendering
    normalize_difference: bool = True  # Whether to normalize difference maps
    show_metrics: bool = True  # Whether to show metrics in view synthesis comparison
    use_cpu_lpips: bool = False  # Whether to use CPU version of LPIPS


def visualize_images(
    images: List[Image.Image],
    titles: Optional[List[str]] = None,
    num_cols: int = 4,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """Visualize a list of images in a grid layout."""
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols

    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_3d_model(
    mesh: trimesh.Trimesh,
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[str] = None,
    texture_map: Optional[np.ndarray] = None,
    normal_map: Optional[np.ndarray] = None
) -> None:
    """Visualize 3D model using Open3D with texture and normal map support."""
    config = config or VisualizationConfig()

    # Optimize mesh if too large
    if len(mesh.vertices) > config.max_vertices:
        mesh = mesh.simplify_quadratic_decimation(config.max_vertices)

    # Convert trimesh to Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(mesh.vertex_normals)

    # Apply texture map if available
    if texture_map is not None:
        texture_map = cv2.resize(
            texture_map, (config.texture_resolution, config.texture_resolution))
        o3d_mesh.textures = [o3d.geometry.Image(texture_map)]

    # Apply normal map if available
    if normal_map is not None:
        normal_map = cv2.resize(
            normal_map, (config.texture_resolution, config.texture_resolution))
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(
            compute_normal_map_vertices(mesh, normal_map)
        )

    # Generate multiple views
    views = []
    for i in range(config.num_views):
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=config.image_size[0], height=config.image_size[1], visible=False)

        # Add mesh to visualizer
        vis.add_geometry(o3d_mesh)

        # Set lighting
        opt = vis.get_render_option()
        opt.background_color = np.array(config.background_color)
        opt.light_on = True
        opt.point_size = config.point_size
        opt.line_width = config.line_width
        opt.show_coordinate_frame = config.show_axes
        opt.mesh_show_wireframe = config.show_grid

        # Set camera position
        angle = 2 * np.pi * i / config.num_views
        camera_pos = np.array([
            config.camera_distance * np.cos(angle),
            config.camera_distance * np.sin(angle),
            config.camera_distance * 0.5
        ])
        ctr = vis.get_view_control()
        ctr.set_zoom(config.camera_distance)
        ctr.set_center(config.camera_center)
        ctr.set_up(config.camera_up)
        ctr.set_lookat(camera_pos)

        # Render and capture image
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        views.append(Image.fromarray(
            (np.asarray(image) * 255).astype(np.uint8)))

        vis.destroy_window()

    # Save or display views
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        for i, view in enumerate(views):
            view.save(f"{base_path}_view{i}.png", quality=95)
    else:
        visualize_images(
            views, titles=[f"View {i+1}" for i in range(len(views))])


def visualize_nerf_results(
    rendered_image: np.ndarray,
    target_image: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    normal_map: Optional[np.ndarray] = None,
    texture_map: Optional[np.ndarray] = None,
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[str] = None
) -> None:
    """Visualize NeRF results with all available outputs."""
    config = config or VisualizationConfig()

    # Calculate number of subplots based on available data
    num_plots = 2  # Original and rendered images
    if depth_map is not None:
        num_plots += 1
    if normal_map is not None:
        num_plots += 1
    if texture_map is not None:
        num_plots += 1

    # Create figure with appropriate layout
    fig = plt.figure(figsize=(5 * num_plots, 5))

    # Original and rendered images
    plt.subplot(1, num_plots, 1)
    plt.imshow(target_image)
    plt.title("Target Image")
    plt.axis('off')

    plt.subplot(1, num_plots, 2)
    plt.imshow(rendered_image)
    plt.title("Rendered Image")
    plt.axis('off')

    # Difference map
    diff = np.abs(target_image - rendered_image)
    if config.normalize_difference:
        diff = (diff - diff.min()) / (diff.max() - diff.min())

    plt.subplot(1, num_plots, 3)
    plt.imshow(diff, cmap=config.colormap)
    plt.colorbar()
    plt.title("Difference Map")
    plt.axis('off')

    # Depth map
    if depth_map is not None:
        plt.subplot(1, num_plots, 4)
        plt.imshow(depth_map, cmap='viridis')
        plt.colorbar()
        plt.title("Depth Map")
        plt.axis('off')

    # Normal map
    if normal_map is not None:
        plt.subplot(1, num_plots, 5)
        plt.imshow(normal_map)
        plt.colorbar()
        plt.title("Normal Map")
        plt.axis('off')

    # Texture map
    if texture_map is not None:
        plt.subplot(1, num_plots, 6)
        plt.imshow(texture_map)
        plt.colorbar()
        plt.title("Texture Map")
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def compare_view_synthesis(
    original_images: List[Image.Image],
    rendered_images: List[Image.Image],
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """Compare original and rendered images with visual and metric comparison."""
    config = config or VisualizationConfig()

    # Convert images to numpy arrays
    orig_arrays = [np.array(img) for img in original_images]
    rend_arrays = [np.array(img) for img in rendered_images]

    # Compute metrics
    metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': []
    }

    for orig, rend in zip(orig_arrays, rend_arrays):
        # Resize if needed
        if orig.shape != rend.shape:
            rend = cv2.resize(rend, (orig.shape[1], orig.shape[0]))

        # PSNR
        mse = np.mean((orig - rend) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        metrics['psnr'].append(psnr)

        # SSIM
        ssim_val = ssim(orig, rend, multichannel=True)
        metrics['ssim'].append(ssim_val)

        # LPIPS
        if config.use_cpu_lpips or torch.cuda.is_available():
            orig_tensor = torch.from_numpy(orig).float().permute(
                2, 0, 1).unsqueeze(0) / 255.0
            rend_tensor = torch.from_numpy(rend).float().permute(
                2, 0, 1).unsqueeze(0) / 255.0
            lpips_val = compute_lpips(orig_tensor, rend_tensor)
            metrics['lpips'].append(lpips_val)

    # Create visualization
    if config.show_metrics:
        fig = plt.figure(figsize=(15, 10))

        # Plot metrics
        plt.subplot(231)
        plt.plot(metrics['psnr'])
        plt.title("PSNR")
        plt.xlabel("View")
        plt.ylabel("dB")

        plt.subplot(232)
        plt.plot(metrics['ssim'])
        plt.title("SSIM")
        plt.xlabel("View")
        plt.ylabel("Score")

        if 'lpips' in metrics:
            plt.subplot(233)
            plt.plot(metrics['lpips'])
            plt.title("LPIPS")
            plt.xlabel("View")
            plt.ylabel("Distance")

        # Show image comparisons
        for i, (orig, rend) in enumerate(zip(orig_arrays, rend_arrays)):
            plt.subplot(2, 3, i + 4)
            plt.imshow(np.hstack([orig, rend]))
            plt.title(f"View {i+1}: Original | Rendered")
            plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    return {k: np.mean(v) for k, v in metrics.items()}


def compute_lpips(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute LPIPS distance between two images with CPU support."""
    import lpips

    # Use CPU version if specified
    if not torch.cuda.is_available():
        img1 = img1.cpu()
        img2 = img2.cpu()
        loss_fn = lpips.LPIPS(net='alex', verbose=False)
    else:
        loss_fn = lpips.LPIPS(net='alex').to(img1.device)

    # Compute distance
    with torch.no_grad():
        distance = loss_fn(img1, img2).item()

    return distance


def compute_normal_map_vertices(
    mesh: trimesh.Trimesh,
    normal_map: np.ndarray
) -> np.ndarray:
    """Compute vertex normals from normal map."""
    # Convert normal map to vertex normals
    vertex_normals = np.zeros_like(mesh.vertices)

    # Get UV coordinates if available
    if hasattr(mesh.visual, 'uv'):
        uvs = mesh.visual.uv
        for i, uv in enumerate(uvs):
            u, v = int(uv[0] * normal_map.shape[1]
                       ), int(uv[1] * normal_map.shape[0])
            normal = normal_map[v, u]
            vertex_normals[i] = normal
    else:
        # If no UV coordinates, use vertex positions
        vertex_normals = mesh.vertex_normals

    return vertex_normals


def plot_training_progress(
    losses: Dict[str, List[float]],
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot training progress with multiple loss types and log scales."""
    config = config or VisualizationConfig()

    if config.use_plotly:
        # Create interactive plot with Plotly
        fig = go.Figure()

        for loss_name, loss_values in losses.items():
            fig.add_trace(go.Scatter(
                y=loss_values,
                name=loss_name,
                mode='lines+markers'
            ))

        fig.update_layout(
            title="Training Progress",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            yaxis_type="log" if any(
                v > 100 for v in losses.values()) else "linear",
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    else:
        # Create static plot with Matplotlib
        plt.figure(figsize=(10, 6))

        for loss_name, loss_values in losses.items():
            plt.plot(loss_values, label=loss_name)

        plt.title("Training Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        if any(v > 100 for v in losses.values()):
            plt.yscale('log')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def test_visualization():
    """Test the visualization functions."""
    # Create test data
    dummy_images = [
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        for _ in range(4)
    ]

    # Create test mesh with texture and normal maps
    vertices = np.random.rand(100, 3)
    faces = np.random.randint(0, 100, (50, 3))
    vertex_colors = np.random.rand(100, 3)
    vertex_normals = np.random.rand(100, 3)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        vertex_normals=vertex_normals
    )

    # Create test maps
    texture_map = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    normal_map = np.random.rand(256, 256, 3)
    normal_map = normal_map / np.linalg.norm(normal_map, axis=2, keepdims=True)

    # Create test losses
    losses = {
        'total_loss': np.random.rand(100).tolist(),
        'rgb_loss': np.random.rand(100).tolist(),
        'depth_loss': np.random.rand(100).tolist()
    }

    # Test visualization functions
    config = VisualizationConfig(
        image_size=(800, 800),
        lighting_intensity=1.0,
        camera_distance=2.0,
        num_views=4
    )

    # Test image visualization
    visualize_images(
        dummy_images,
        titles=['Image 1', 'Image 2', 'Image 3', 'Image 4'],
        save_path='output/test_images.png'
    )

    # Test 3D model visualization with texture and normal maps
    visualize_3d_model(
        mesh,
        config=config,
        save_path='output/test_3d_model.png',
        texture_map=texture_map,
        normal_map=normal_map
    )

    # Test NeRF results visualization with all maps
    visualize_nerf_results(
        np.random.rand(64, 64, 3),
        np.random.rand(64, 64, 3),
        depth_map=np.random.rand(64, 64),
        normal_map=np.random.rand(64, 64, 3),
        texture_map=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        config=config,
        save_path='output/test_nerf_results.png'
    )

    # Test view synthesis comparison
    compare_view_synthesis(
        dummy_images,
        dummy_images,
        config=config,
        save_path='output/test_view_synthesis.png'
    )

    # Test training progress plotting
    plot_training_progress(
        losses,
        config=config,
        save_path='output/test_training_progress.html'
    )

    print("Visualization tests completed successfully!")


if __name__ == "__main__":
    test_visualization()
