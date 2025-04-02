# Text-to-3D Generation

This project implements a pipeline for generating 3D models from text descriptions, combining:
- Stable Diffusion for text-to-image generation
- DreamFusion/NeRF techniques for 3D reconstruction

## Project Structure

```
generative_ai/
│
├── requirements.txt      # Project dependencies
├── main.py               # Main entry point
│
├── text_to_image/        # Text-to-image generation using Stable Diffusion
│   ├── __init__.py
│   └── stable_diffusion.py
│
├── image_to_3d/          # 3D model generation from images
│   ├── __init__.py
│   ├── dreambooth.py     # DreamFusion implementation
│   └── nerf.py           # NeRF implementation
│
└── utils/                # Utility functions
    ├── __init__.py
    └── visualization.py  # Tools for visualizing 3D models
```

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py "Your text description here"
   ```

## How It Works

1. The text description is processed using Stable Diffusion to generate multiple images
2. These images are then processed using DreamFusion/NeRF techniques to reconstruct a 3D model
3. The resulting 3D model is saved and can be visualized

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See requirements.txt for Python package dependencies 