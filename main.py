import os
import argparse
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from text_to_image.stable_diffusion import StableDiffusionGenerator
from image_to_3d.dreambooth import DreamFusionGenerator
from image_to_3d.nerf import NeRFGenerator
from utils.visualization import (
    visualize_images,
    visualize_3d_model,
    visualize_nerf_results,
    compare_view_synthesis,
    plot_training_progress,
    VisualizationConfig
)
import time
import json
import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraPose:
    """Camera pose information for NeRF."""
    position: np.ndarray
    rotation: np.ndarray
    focal_length: float
    principal_point: Tuple[float, float]
    image_size: Tuple[int, int]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Text to 3D Generation Pipeline")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument(
        "--method",
        type=str,
        choices=["dreambooth", "nerf"],
        default="dreambooth",
        help="Method for 3D generation"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for 3D generation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization"
    )
    parser.add_argument(
        "--sd-model",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Stable Diffusion model to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--camera-radius",
        type=float,
        default=2.0,
        help="Radius for camera positions"
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=50.0,
        help="Focal length for camera"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Size of generated images"
    )
    return parser.parse_args()


def generate_camera_poses(
    num_poses: int,
    radius: float,
    image_size: Tuple[int, int],
    focal_length: float
) -> List[CameraPose]:
    """Generate camera poses around the object."""
    poses = []
    for i in range(num_poses):
        # Calculate camera position on a sphere
        theta = 2 * np.pi * i / num_poses
        phi = np.pi / 4  # Fixed elevation angle
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        position = np.array([x, y, z])

        # Calculate camera rotation to look at center
        forward = -position / np.linalg.norm(position)
        right = np.cross(forward, [0, 1, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        rotation = np.stack([right, up, -forward], axis=1)

        # Create camera pose
        pose = CameraPose(
            position=position,
            rotation=rotation,
            focal_length=focal_length,
            principal_point=(image_size[0]/2, image_size[1]/2),
            image_size=image_size
        )
        poses.append(pose)

    return poses


def generate_images(
    prompt: str,
    num_images: int,
    output_dir: str,
    sd_model: str,
    device: str,
    camera_poses: List[CameraPose],
    visualize: bool = False
) -> Tuple[List[Image.Image], List[CameraPose]]:
    """Generate multiple views of an object based on text prompt."""
    try:
        # Initialize Stable Diffusion generator
        generator = StableDiffusionGenerator(
            model_id=sd_model,
            device=device
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate images
        images = []
        for i, pose in enumerate(camera_poses):
            # Add camera information to prompt
            camera_prompt = f"{prompt}, view {i+1}/{num_images}"

            # Generate image
            image = generator.generate_image(
                prompt=camera_prompt,
                negative_prompt="blurry, low quality, distorted",
                num_inference_steps=50,
                guidance_scale=7.5
            )

            # Save image
            image_path = os.path.join(output_dir, f"view_{i:03d}.png")
            image.save(image_path)
            images.append(image)

            # Save camera pose
            pose_path = os.path.join(output_dir, f"pose_{i:03d}.json")
            with open(pose_path, 'w') as f:
                json.dump({
                    'position': pose.position.tolist(),
                    'rotation': pose.rotation.tolist(),
                    'focal_length': pose.focal_length,
                    'principal_point': pose.principal_point,
                    'image_size': pose.image_size
                }, f)

        # Visualize if requested
        if visualize:
            visualize_images(
                images,
                titles=[f"View {i+1}" for i in range(len(images))],
                save_path=os.path.join(output_dir, "all_views.png")
            )

        return images, camera_poses

    except Exception as e:
        logging.error(f"Error generating images: {str(e)}")
        raise


def generate_3d_model_dreambooth(
    prompt: str,
    images: List[Image.Image],
    output_dir: str,
    iterations: int,
    device: str,
    visualize: bool = False
) -> Optional[trimesh.Trimesh]:
    """Generate 3D model using DreamFusion approach."""
    try:
        # Initialize DreamFusion generator
        generator = DreamFusionGenerator(
            device=device,
            num_iterations=iterations
        )

        # Generate 3D model
        mesh, losses = generator.generate_3d_model(
            prompt=prompt,
            images=images
        )

        # Save mesh
        mesh_path = os.path.join(output_dir, "model.obj")
        mesh.export(mesh_path)

        # Save losses
        losses_path = os.path.join(output_dir, "losses.json")
        with open(losses_path, 'w') as f:
            json.dump(losses, f)

        # Visualize if requested
        if visualize:
            # Visualize training progress
            plot_training_progress(
                losses,
                save_path=os.path.join(output_dir, "training_progress.html")
            )

            # Visualize 3D model
            visualize_3d_model(
                mesh,
                save_path=os.path.join(output_dir, "model_views.png")
            )

        return mesh

    except Exception as e:
        logging.error(f"Error in DreamFusion generation: {str(e)}")
        return None


def generate_3d_model_nerf(
    images: List[Image.Image],
    camera_poses: List[CameraPose],
    output_dir: str,
    iterations: int,
    device: str,
    visualize: bool = False
) -> Optional[trimesh.Trimesh]:
    """Generate 3D model using NeRF approach."""
    try:
        # Initialize NeRF generator
        generator = NeRFGenerator(
            device=device,
            num_iterations=iterations
        )

        # Convert images to numpy arrays
        image_arrays = [np.array(img) for img in images]

        # Generate 3D model
        mesh, losses, render_results = generator.generate_3d_model(
            images=image_arrays,
            camera_poses=camera_poses
        )

        # Save mesh
        mesh_path = os.path.join(output_dir, "model.obj")
        mesh.export(mesh_path)

        # Save losses
        losses_path = os.path.join(output_dir, "losses.json")
        with open(losses_path, 'w') as f:
            json.dump(losses, f)

        # Visualize if requested
        if visualize:
            # Visualize training progress
            plot_training_progress(
                losses,
                save_path=os.path.join(output_dir, "training_progress.html")
            )

            # Visualize 3D model
            visualize_3d_model(
                mesh,
                save_path=os.path.join(output_dir, "model_views.png")
            )

            # Visualize NeRF results
            for i, (rendered, target, depth, normal, texture) in enumerate(render_results):
                visualize_nerf_results(
                    rendered_image=rendered,
                    target_image=target,
                    depth_map=depth,
                    normal_map=normal,
                    texture_map=texture,
                    save_path=os.path.join(
                        output_dir, f"nerf_results_{i:03d}.png")
                )

            # Compare view synthesis
            compare_view_synthesis(
                images,
                [Image.fromarray((rendered * 255).astype(np.uint8))
                 for rendered, _, _, _, _ in render_results],
                save_path=os.path.join(
                    output_dir, "view_synthesis_comparison.png")
            )

        return mesh

    except Exception as e:
        logging.error(f"Error in NeRF generation: {str(e)}")
        return None


def main():
    """Main function to run the text-to-3D generation pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Parse arguments
        args = parse_arguments()

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate camera poses
        camera_poses = generate_camera_poses(
            num_poses=args.num_images,
            radius=args.camera_radius,
            image_size=(args.image_size, args.image_size),
            focal_length=args.focal_length
        )

        # Generate images
        logging.info("Generating images...")
        start_time = time.time()
        images, camera_poses = generate_images(
            prompt=args.prompt,
            num_images=args.num_images,
            output_dir=str(output_dir / "images"),
            sd_model=args.sd_model,
            device=args.device,
            camera_poses=camera_poses,
            visualize=args.visualize
        )
        image_time = time.time() - start_time
        logging.info(f"Image generation completed in {image_time:.2f} seconds")

        # Generate 3D model
        logging.info(f"Generating 3D model using {args.method}...")
        start_time = time.time()

        if args.method == "dreambooth":
            mesh = generate_3d_model_dreambooth(
                prompt=args.prompt,
                images=images,
                output_dir=str(output_dir / "dreambooth"),
                iterations=args.iterations,
                device=args.device,
                visualize=args.visualize
            )
        else:  # nerf
            mesh = generate_3d_model_nerf(
                images=images,
                camera_poses=camera_poses,
                output_dir=str(output_dir / "nerf"),
                iterations=args.iterations,
                device=args.device,
                visualize=args.visualize
            )

        model_time = time.time() - start_time
        logging.info(
            f"3D model generation completed in {model_time:.2f} seconds")

        if mesh is None:
            logging.error("Failed to generate 3D model")
            return

        # Save timing information
        timing_info = {
            "image_generation_time": image_time,
            "model_generation_time": model_time,
            "total_time": image_time + model_time
        }
        with open(output_dir / "timing.json", 'w') as f:
            json.dump(timing_info, f)

        logging.info("Pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
