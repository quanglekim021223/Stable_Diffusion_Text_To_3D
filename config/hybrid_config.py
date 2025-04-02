from dataclasses import dataclass
from typing import Optional


@dataclass
class Magic3DConfig:
    resolution: int = 2048
    quality: str = "high"
    num_views: int = 32
    texture_size: int = 4096
    use_denoising: bool = True
    enhance_details: bool = True


@dataclass
class NeRFConfig:
    resolution: int = 1024
    num_views: int = 64
    quality: str = "ultra"
    num_rays: int = 1024
    num_samples: int = 128
    use_viewdirs: bool = True
    use_fine_network: bool = True


@dataclass
class HybridConfig:
    # Magic3D settings
    magic3d: Magic3DConfig = Magic3DConfig()

    # NeRF settings
    nerf: NeRFConfig = NeRFConfig()

    # Hybrid settings
    blend_weight: float = 0.7  # Weight for Magic3D geometry
    texture_blend: float = 0.5  # Weight for texture blending
    use_geometry_from_magic3d: bool = True
    use_texture_from_nerf: bool = True
    use_lighting_from_nerf: bool = True

    # Output settings
    output_dir: str = "output/hybrid_model"
    save_intermediate: bool = True
    save_frequency: int = 100  # Save every N iterations

    # Device settings
    device: str = "cuda"
    num_workers: int = 4

    # Optimization settings
    learning_rate: float = 1e-4
    num_iterations: int = 1000
    batch_size: int = 4

    # Quality settings
    min_quality_score: float = 0.8
    max_retries: int = 3
