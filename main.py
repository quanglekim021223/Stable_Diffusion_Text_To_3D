import argparse
from generators.hybrid_generator import HybridGenerator
from config.hybrid_config import HybridConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D model using Hybrid Magic3D + NeRF")
    parser.add_argument("prompt", type=str,
                        help="Text prompt describing the 3D model")
    parser.add_argument("--output-dir", type=str,
                        default="output/hybrid_model", help="Output directory")
    parser.add_argument("--resolution", type=int,
                        default=2048, help="Output resolution")
    parser.add_argument("--quality", type=str, default="high",
                        help="Quality level (low/medium/high)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Create config
    config = HybridConfig(
        output_dir=args.output_dir,
        device=args.device
    )
    config.magic3d.resolution = args.resolution
    config.magic3d.quality = args.quality

    # Initialize generator
    generator = HybridGenerator(config)

    try:
        # Generate model
        result = generator.generate(args.prompt)

        print(f"\nGeneration completed successfully!")
        print(f"Quality score: {result['quality_score']:.2f}")
        print(f"Output saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
