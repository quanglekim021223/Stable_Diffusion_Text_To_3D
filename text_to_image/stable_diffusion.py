import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np
from typing import List, Optional, Tuple, Dict, Literal
import cv2
import json
import trimesh
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm
import subprocess
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum


class ObjectType(Enum):
    ROBOT = "robot"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    FURNITURE = "furniture"
    TOOL = "tool"
    OTHER = "other"


@dataclass
class SkeletonConfig:
    """Configuration for skeleton generation."""
    joints: List[Tuple[float, float]]
    connections: List[Tuple[int, int]]
    joint_radius: int = 5
    connection_thickness: int = 2


class PoseGenerator:
    """Helper class to generate and manipulate pose maps."""

    def __init__(self, resolution: int = 512):
        self.resolution = resolution
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

        # Define skeleton configurations for different object types
        self.skeleton_configs = {
            ObjectType.ROBOT: SkeletonConfig(
                joints=[
                    (0.5, 0.2),  # head
                    (0.5, 0.4),  # neck
                    (0.5, 0.6),  # body
                    (0.3, 0.4),  # left shoulder
                    (0.7, 0.4),  # right shoulder
                    (0.3, 0.8),  # left hip
                    (0.7, 0.8),  # right hip
                    (0.3, 0.6),  # left elbow
                    (0.7, 0.6),  # right elbow
                    (0.3, 1.0),  # left knee
                    (0.7, 1.0),  # right knee
                    (0.3, 1.2),  # left foot
                    (0.7, 1.2),  # right foot
                ],
                connections=[
                    (0, 1), (1, 2), (2, 3), (2, 4), (3, 6), (4, 7),
                    (2, 5), (2, 6), (5, 9), (6, 10), (9, 11), (10, 12)
                ]
            ),
            ObjectType.ANIMAL: SkeletonConfig(
                joints=[
                    (0.5, 0.2),  # head
                    (0.5, 0.4),  # neck
                    (0.5, 0.6),  # body
                    (0.3, 0.6),  # left front leg
                    (0.7, 0.6),  # right front leg
                    (0.3, 0.8),  # left back leg
                    (0.7, 0.8),  # right back leg
                ],
                connections=[
                    (0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (2, 6)
                ]
            ),
            ObjectType.VEHICLE: SkeletonConfig(
                joints=[
                    (0.3, 0.3),  # front left
                    (0.7, 0.3),  # front right
                    (0.3, 0.7),  # back left
                    (0.7, 0.7),  # back right
                    (0.5, 0.5),  # center
                ],
                connections=[
                    (0, 1), (1, 4), (4, 3), (3, 0),  # body
                    (0, 2), (1, 3)  # diagonal connections
                ]
            ),
            ObjectType.FURNITURE: SkeletonConfig(
                joints=[
                    (0.3, 0.3),  # top left
                    (0.7, 0.3),  # top right
                    (0.3, 0.7),  # bottom left
                    (0.7, 0.7),  # bottom right
                ],
                connections=[
                    (0, 1), (1, 3), (3, 2), (2, 0)  # rectangle
                ]
            ),
            ObjectType.TOOL: SkeletonConfig(
                joints=[
                    (0.5, 0.2),  # handle top
                    (0.5, 0.8),  # handle bottom
                    (0.7, 0.5),  # tool head
                ],
                connections=[
                    (0, 1), (1, 2)  # handle and tool head
                ]
            )
        }

    def create_skeleton(self, object_type: ObjectType) -> np.ndarray:
        """Create a skeleton pose map based on object type."""
        # Create a blank image
        pose_map = np.zeros(
            (self.resolution, self.resolution, 3), dtype=np.uint8)

        # Get skeleton configuration
        config = self.skeleton_configs.get(
            object_type, self.skeleton_configs[ObjectType.OTHER])

        # Draw joints
        for joint in config.joints:
            x = int(joint[0] * self.resolution)
            y = int(joint[1] * self.resolution)
            cv2.circle(pose_map, (x, y), config.joint_radius,
                       (255, 255, 255), -1)

        # Draw connections
        for connection in config.connections:
            start = config.joints[connection[0]]
            end = config.joints[connection[1]]
            start_x = int(start[0] * self.resolution)
            start_y = int(start[1] * self.resolution)
            end_x = int(end[0] * self.resolution)
            end_y = int(end[1] * self.resolution)
            cv2.line(pose_map, (start_x, start_y), (end_x, end_y),
                     (255, 255, 255), config.connection_thickness)

        return pose_map

    def extract_pose_from_image(self, image_path: str) -> np.ndarray:
        """Extract pose from an image using MediaPipe."""
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get pose landmarks
        results = self.pose.process(image)
        if not results.pose_landmarks:
            return self.create_skeleton(ObjectType.OTHER)

        # Create pose map
        pose_map = np.zeros(
            (self.resolution, self.resolution, 3), dtype=np.uint8)

        # Draw skeleton
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            start_point = results.pose_landmarks.landmark[start_idx]
            end_point = results.pose_landmarks.landmark[end_idx]

            # Convert normalized coordinates to pixel coordinates
            start_x = int(start_point.x * self.resolution)
            start_y = int(start_point.y * self.resolution)
            end_x = int(end_point.x * self.resolution)
            end_y = int(end_point.y * self.resolution)

            cv2.line(pose_map, (start_x, start_y),
                     (end_x, end_y), (255, 255, 255), 2)

        return pose_map

    def rotate_pose(self, pose_map: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a pose map around its center."""
        center = (pose_map.shape[1] // 2, pose_map.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            pose_map, rotation_matrix, (pose_map.shape[1], pose_map.shape[0]))
        return rotated


class DepthEstimator:
    """Helper class for depth estimation using MiDaS."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "Intel/dpt-large").to(device)

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map from image using MiDaS."""
        # Convert to PIL Image
        image_pil = Image.fromarray(image)

        # Process image
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get depth prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (depth * 255).astype(np.uint8)

        return depth


class StableDiffusionGenerator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_id: str = "lllyasviel/control_v2_openpose",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_depth: bool = True,
        object_type: ObjectType = ObjectType.ROBOT
    ):
        """Initialize the Stable Diffusion generator with improved controls."""
        self.device = device
        self.use_depth = use_depth
        self.object_type = object_type

        # Load ControlNet models
        self.controlnet_pose = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        if use_depth:
            self.controlnet_depth = ControlNetModel.from_pretrained(
                "lllyasviel/control_v2_depth",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            self.depth_estimator = DepthEstimator(device)

        # Initialize pipeline with ControlNet
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=[self.controlnet_pose] +
            ([self.controlnet_depth] if use_depth else []),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        # Enable memory optimization
        self.pipeline.enable_attention_slicing()
        if device == "cuda":
            self.pipeline.enable_vae_tiling()

        # Initialize pose generator
        self.pose_generator = PoseGenerator()

        # Load camera parameters
        self.camera_params = self._load_camera_params()

    def _load_camera_params(self) -> Dict:
        """Load camera parameters for consistent view generation."""
        return {
            "focal_length": 1000,
            "principal_point": (256, 256),
            "distortion": [0, 0, 0, 0, 0]
        }

    def generate_views(
        self,
        prompt: str,
        num_views: int = 8,
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
        reference_image: Optional[str] = None
    ) -> List[Image.Image]:
        """Generate multiple views with improved consistency."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate or load base pose
        if reference_image:
            base_pose = self.pose_generator.extract_pose_from_image(
                reference_image)
        else:
            base_pose = self.pose_generator.create_skeleton(self.object_type)

        # Generate views with consistent poses
        images = []
        for i in tqdm(range(num_views), desc="Generating views"):
            # Rotate pose for different views
            angle = 360 / num_views * i
            rotated_pose = self.pose_generator.rotate_pose(base_pose, angle)

            # Generate depth map if needed
            control_images = [rotated_pose]
            if self.use_depth:
                depth_map = self.depth_estimator.estimate_depth(rotated_pose)
                control_images.append(depth_map)

            # Generate image with ControlNet
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=[
                    0.8] + ([0.5] if self.use_depth else [])
            ).images[0]

            images.append(image)

            # Save image and metadata if output directory is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                image.save(os.path.join(output_dir, f"view_{i:03d}.png"))

                # Save camera parameters for 3D reconstruction
                metadata = {
                    "camera_params": self.camera_params,
                    "view_angle": angle,
                    "seed": seed
                }
                with open(os.path.join(output_dir, f"view_{i:03d}_metadata.json"), "w") as f:
                    json.dump(metadata, f)

        # Run COLMAP reconstruction if output directory is provided
        if output_dir:
            self._run_colmap_reconstruction(output_dir)

        return images

    def _run_colmap_reconstruction(self, output_dir: str) -> None:
        """Run COLMAP reconstruction on generated images."""
        # Create COLMAP workspace
        workspace_dir = os.path.join(output_dir, "colmap_workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        # Copy images to workspace
        images_dir = os.path.join(workspace_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for i in range(len(os.listdir(output_dir))):
            if f"view_{i:03d}.png" in os.listdir(output_dir):
                os.system(
                    f"cp {os.path.join(output_dir, f'view_{i:03d}.png')} {images_dir}/")

        # Run COLMAP reconstruction
        try:
            subprocess.run([
                "colmap", "automatic_reconstructor",
                "--workspace_path", workspace_dir,
                "--image_path", images_dir,
                "--camera_model", "SIMPLE_PINHOLE",
                "--sparse", "1",
                "--dense", "1"
            ], check=True)

            # Convert to mesh if reconstruction successful
            sparse_dir = os.path.join(workspace_dir, "sparse")
            if os.path.exists(sparse_dir):
                self._convert_to_mesh(sparse_dir, output_dir)

        except subprocess.CalledProcessError as e:
            print(f"COLMAP reconstruction failed: {e}")

    def _convert_to_mesh(self, sparse_dir: str, output_dir: str) -> None:
        """Convert COLMAP reconstruction to mesh."""
        # Load point cloud
        points = []
        colors = []
        for i in range(len(os.listdir(sparse_dir))):
            if os.path.isdir(os.path.join(sparse_dir, f"{i}")):
                points3d = np.load(os.path.join(
                    sparse_dir, f"{i}", "points3D.npy"))
                points.append(points3d[:, :3])
                colors.append(points3d[:, 3:])

        points = np.concatenate(points)
        colors = np.concatenate(colors)

        # Create mesh using Poisson surface reconstruction
        mesh = trimesh.Trimesh(vertices=points, vertex_colors=colors)
        mesh.export(os.path.join(output_dir, "reconstructed_mesh.ply"))

    def __call__(
        self,
        prompt: str,
        num_views: int = 8,
        output_dir: Optional[str] = None,
        reference_image: Optional[str] = None
    ) -> List[Image.Image]:
        """Generate multiple views of an object."""
        return self.generate_views(
            prompt=prompt,
            num_views=num_views,
            output_dir=output_dir,
            reference_image=reference_image
        )


def test_stable_diffusion():
    """Test the improved StableDiffusionGenerator."""
    # Test with different object types
    for object_type in ObjectType:
        generator = StableDiffusionGenerator(
            use_depth=True,
            object_type=object_type
        )

        prompt = f"A {object_type.value} on a white background, studio lighting, 8k, high detail"
        generator.generate_views(
            prompt=prompt,
            num_views=4,
            output_dir=f"output/test_images_{object_type.value}",
            seed=42  # Fixed seed for consistency
        )

    print("Test completed successfully!")


if __name__ == "__main__":
    test_stable_diffusion()
