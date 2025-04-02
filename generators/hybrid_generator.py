import os
import torch
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from config.hybrid_config import HybridConfig
from generators.magic3d_generator import Magic3DGenerator
from generators.nerf_generator import NeRFGenerator
from utils.mesh_optimizer import MeshOptimizer
from utils.quality_checker import QualityChecker


class HybridGenerator:
    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self.device = torch.device(self.config.device)

        # Initialize generators
        self.magic3d = Magic3DGenerator(self.device)
        self.nerf = NeRFGenerator(self.device)

        # Initialize utilities
        self.mesh_optimizer = MeshOptimizer()
        self.quality_checker = QualityChecker()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    def generate(self, prompt: str) -> Dict:
        """Generate a 3D model using both Magic3D and NeRF"""
        print(f"Generating 3D model for prompt: {prompt}")

        # Step 1: Generate base model with Magic3D
        print("Generating base model with Magic3D...")
        magic3d_model = self.magic3d.generate_3d(
            prompt=prompt,
            resolution=self.config.magic3d.resolution,
            quality=self.config.magic3d.quality
        )

        # Step 2: Generate NeRF model
        print("Generating NeRF model...")
        nerf_model = self.nerf.generate(
            prompt=prompt,
            resolution=self.config.nerf.resolution,
            num_views=self.config.nerf.num_views
        )

        # Step 3: Combine and optimize
        print("Combining and optimizing models...")
        combined_model = self._combine_models(magic3d_model, nerf_model)

        # Step 4: Optimize mesh
        print("Optimizing mesh...")
        optimized_mesh = self.mesh_optimizer.optimize(combined_model["mesh"])

        # Step 5: Quality check
        print("Checking quality...")
        quality_score = self.quality_checker.check(optimized_mesh)

        if quality_score < self.config.min_quality_score:
            print("Quality below threshold, enhancing...")
            optimized_mesh = self._enhance_quality(optimized_mesh)

        # Step 6: Save results
        print("Saving results...")
        self._save_results(optimized_mesh, combined_model)

        return {
            "mesh": optimized_mesh,
            "textures": combined_model["textures"],
            "quality_score": quality_score
        }

    def _combine_models(self, magic3d_model: Dict, nerf_model: Dict) -> Dict:
        """Combine models from Magic3D and NeRF"""
        combined = {
            "geometry": magic3d_model["geometry"] if self.config.use_geometry_from_magic3d else nerf_model["geometry"],
            "texture": self._blend_textures(
                magic3d_model["texture"],
                nerf_model["texture"],
                self.config.texture_blend
            ),
            "lighting": nerf_model["lighting"] if self.config.use_lighting_from_nerf else magic3d_model["lighting"]
        }
        return combined

    def _blend_textures(self, magic3d_texture: torch.Tensor, nerf_texture: torch.Tensor, blend_weight: float) -> torch.Tensor:
        """Blend textures from both models"""
        return blend_weight * magic3d_texture + (1 - blend_weight) * nerf_texture

    def _enhance_quality(self, mesh: Dict) -> Dict:
        """Enhance mesh quality if below threshold"""
        # Implement quality enhancement logic
        return mesh

    def _save_results(self, mesh: Dict, combined_model: Dict):
        """Save the final results"""
        # Save mesh
        mesh_path = os.path.join(self.config.output_dir, "final_mesh.obj")
        mesh.export(mesh_path)

        # Save textures
        for name, texture in combined_model["textures"].items():
            texture_path = os.path.join(self.config.output_dir, f"{name}.png")
            texture.save(texture_path)

        # Save metadata
        metadata = {
            "prompt": self.prompt,
            "quality_score": self.quality_checker.check(mesh),
            "config": self.config.__dict__
        }
        metadata_path = os.path.join(self.config.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
