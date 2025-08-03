import argparse
import os
from multiprocessing import Pool, cpu_count

import torch
from PIL import (
    Image,
    ImageFile,  # Required for handling truncated images - thx gpt
)

# This is necessary for Pillow to avoid crashing on truncated images
# See: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#truncated-images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from diffusers import StableDiffusionUpscalePipeline

TARGET_SIZES = {"4k": (3840, 2160), "6k": (6144, 3456), "8k": (7680, 4320)}

global_upscale_pipeline = None
global_worker_args = {}


def load_model_for_process(worker_args=None):
    """
    Loads the Stable Diffusion Upscale model and sets up worker-specific arguments.
    This function should be called once per process (either in the main process for sequential
    or as an initializer for multiprocessing pool workers).
    """
    global global_upscale_pipeline
    global global_worker_args

    if worker_args:
        global_worker_args = worker_args

    if global_upscale_pipeline is None:
        model_id = "stabilityai/stable-diffusion-x4-upscaler"

        # device: mps for apple fanboi (me), cuda for shovelmaker, fallback to poorcpu
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"Worker {os.getpid()}: Using Apple Silicon (MPS) backend.")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"Worker {os.getpid()}: Using CUDA backend.")
        else:
            print(
                f"Worker {os.getpid()}: Using CPU backend (will be slow for Stable Diffusion)."
            )

        if device == "mps":
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, torch_dtype=torch.float32
            )
        elif device == "cuda":
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
        else:  # cpu
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)

        global_upscale_pipeline = pipeline.to(device)

        if device == "mps" or device == "cuda":
            global_upscale_pipeline.enable_attention_slicing()
            print(
                f"Worker {os.getpid()}: Attention slicing enabled for memory efficiency."
            )

    return global_upscale_pipeline


def process_image(filename):
    """
    Processes a single image file for super-resolution.
    This function is designed to be called by multiprocessing pool workers.
    """
    input_path = os.path.join(global_worker_args["input_dir"], filename)
    output_path = os.path.join(
        global_worker_args["output_dir"], f"upscaled_{filename}"
    )

    try:
        model_pipeline = load_model_for_process()

        image = Image.open(input_path).convert("RGB")

        max_upscaler_input_dim = 512
        original_width, original_height = image.size

        low_res_image_for_upscaler = image
        if max(original_width, original_height) > max_upscaler_input_dim:
            if original_width > original_height:
                new_width = max_upscaler_input_dim
                new_height = int(
                    original_height * (max_upscaler_input_dim / original_width)
                )
            else:
                new_height = max_upscaler_input_dim
                new_width = int(
                    original_width * (max_upscaler_input_dim / original_height)
                )
            low_res_image_for_upscaler = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            print(
                f"Worker {os.getpid()}: Resized {filename} from {original_width}x{original_height} to {new_width}x{new_height} for upscaler input."
            )

        prompt = (
            "a highly detailed and realistic photograph, UHD, 8k, sharp focus"
        )

        upscaled_image_intermediate = model_pipeline(
            image=low_res_image_for_upscaler, prompt=prompt
        ).images[0]

        sr_image = upscaled_image_intermediate.resize(
            global_worker_args["target_size"], Image.Resampling.LANCZOS
        )

        sr_image.save(output_path)
        print(f"Worker {os.getpid()}: Finished: {filename} -> {output_path}")
    except Exception as e:
        print(f"Worker {os.getpid()}: Error processing {filename}: {e}")
        import traceback

        traceback.print_exc()


def process_sequential(image_files, input_dir, output_dir, target_size):
    """
    Processes images sequentially without multiprocessing.
    """
    print("Running in sequential mode (1 worker)...")
    global global_worker_args
    global_worker_args = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "target_size": target_size,
    }
    model_pipeline = load_model_for_process()

    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"upscaled_{filename}")

        try:
            image = Image.open(input_path).convert("RGB")

            max_upscaler_input_dim = 512
            original_width, original_height = image.size

            low_res_image_for_upscaler = image
            if max(original_width, original_height) > max_upscaler_input_dim:
                if original_width > original_height:
                    new_width = max_upscaler_input_dim
                    new_height = int(
                        original_height
                        * (max_upscaler_input_dim / original_width)
                    )
                else:
                    new_height = max_upscaler_input_dim
                    new_width = int(
                        original_width
                        * (max_upscaler_input_dim / original_height)
                    )
                low_res_image_for_upscaler = image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                print(
                    f"Resized {filename} from {original_width}x{original_height} to {new_width}x{new_height} for upscaler input."
                )

            prompt = "a highly detailed and realistic photograph, UHD, 8k, sharp focus"

            upscaled_image_intermediate = model_pipeline(
                image=low_res_image_for_upscaler, prompt=prompt
            ).images[0]

            sr_image = upscaled_image_intermediate.resize(
                target_size, Image.Resampling.LANCZOS
            )
            sr_image.save(output_path)
            print(f"Finished: {filename} -> {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Image Super-Resolution CLI tool with optional multiprocessing using Hugging Face Diffusers"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the directory containing input images. Assumes images are directly in this folder or a 'super_resolution' subfolder.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the directory to save upscaled images.",
    )
    parser.add_argument(
        "--target_resolution",
        choices=["4k", "6k", "8k"],
        default="4k",
        help="Target output resolution for upscaled images (e.g., 4k, 6k, 8k).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to use (1 = sequential). More workers may consume more VRAM/RAM, use cautiously with Stable Diffusion.",
    )

    args = parser.parse_args()

    input_base_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    target_size = TARGET_SIZES[args.target_resolution.lower()]
    supported_ext = (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp")

    input_images_path = os.path.join(input_base_dir, "super_resolution")
    if not os.path.isdir(input_images_path):
        print(
            f"Warning: Subdirectory '{os.path.basename(input_images_path)}' not found in '{input_base_dir}'. "
            f"Using '{input_base_dir}' directly as input image path."
        )
        input_images_path = input_base_dir

    if not os.path.isdir(input_images_path):
        print(f"Error: Input directory '{input_images_path}' does not exist.")
        exit(1)

    image_files = [
        f
        for f in os.listdir(input_images_path)
        if f.lower().endswith(supported_ext)
        and os.path.isfile(os.path.join(input_images_path, f))
    ]

    if not image_files:
        print(
            f"No supported image files found in '{input_images_path}'. Exiting."
        )
        exit(0)

    print(f"Found {len(image_files)} images in '{input_images_path}'.")

    if (
        args.num_workers == 1
    ):  # If we do multiprocessing, it'll kill your computer 6 days from sunday and back
        process_sequential(
            image_files, input_images_path, output_dir, target_size
        )
    else:
        total_cores = cpu_count()
        safe_max_workers = max(1, total_cores // 2)
        num_workers = min(args.num_workers, safe_max_workers)

        if args.num_workers > safe_max_workers:
            print(
                f"Requested {args.num_workers} workers, but limiting to {safe_max_workers} for system stability "
                f"with resource-intensive Stable Diffusion models. Adjust --num_workers if you have more RAM."
            )

        print(
            f"Running in multiprocessing mode with {num_workers} workers (system has {total_cores} cores)."
        )

        worker_initialization_args = {
            "input_dir": input_images_path,
            "output_dir": output_dir,
            "target_size": target_size,
        }

        print(
            "Pre-downloading Stable Diffusion x4 Upscaler model (this might take a while)..."
        )
        try:
            # load cpu/mps
            temp_device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            _ = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler"
            ).to(temp_device)
            print("Model pre-downloaded successfully.")
        except Exception as e:
            print(
                f"Error during model pre-download: {e}. Please ensure you have internet access and sufficient disk space."
            )
            exit(1)

        with Pool(
            processes=num_workers,
            initializer=load_model_for_process,
            initargs=(worker_initialization_args,),
        ) as pool:
            pool.map(process_image, image_files)

    print("All images processed.")


if __name__ == "__main__":
    main()
