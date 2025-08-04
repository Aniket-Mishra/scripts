"""
### Before you run this, know this, deep in your heart:
### This sucks. Like, really sucks.
### I used the prettiest picture I had of my girlfriend on sdxl 1
### She turned into an alien or sth lol
### Sent it to her, she got a nightmare apparently.
### Use with caution.

# Prompt:
python3 text_n_img_to_img.py --image_path "images/img_1.jpeg"
 --prompt "cartoonify - big eyes - horns - space"
 --strength 0.6 --guidance_scale 8
 -steps 20 --model sd_xl_base_1_0
 --lora "name of lora"

 # I could not figure out a way to use a proper lora.
 # Might just train one when I get time
"""

import argparse
import os
from io import BytesIO

import requests
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image


def is_sdxl(model_name):
    return "xl" in model_name.lower()


def find_file(folder, suffix=".safetensors", must_contain=None):
    for f in os.listdir(folder):
        if f.endswith(suffix) and (
            must_contain in f if must_contain else True
        ):
            return os.path.join(folder, f)
    return None


def load_input_image(image_path, target_size):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    return image.convert("RGB").resize(target_size)


def main(args):
    model_folder = os.path.expanduser(os.path.join("~/models", args.model))
    model_path = find_file(model_folder, ".safetensors")
    # vae_path = find_file(model_folder, "vae.safetensors")
    vae_path = next(
        (
            os.path.join(model_folder, f)
            for f in os.listdir(model_folder)
            if f.endswith(".safetensors") and "vae" in f.lower()
        ),
        None,
    )
    lora_path = (
        os.path.expanduser(
            os.path.join(
                "~/models/loras", args.lora, f"{args.lora}.safetensors"
            )
        )
        if args.lora
        else None
    )

    if not model_path:
        raise FileNotFoundError(
            f"No .safetensors model file found in {model_folder}"
        )

    is_xl = is_sdxl(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Meed the model type cuz sdxl has extra params
    ## Also how do I find a lora that actually works :'(
    print(f"\nModel type: {'SDXL' if is_xl else 'SD 1.5'}")
    print(f"Model path: {model_path}")
    print(f"🖥️  Running on: {device}")
    if vae_path:
        print(
            f"VAE loaded: {vae_path}"
        )  # Idk if it did anything, the image was crap both ways lol
    if lora_path:
        print(f"🎨 LoRA loaded: {lora_path}")
    print(f"📝 Prompt: {args.prompt}\n")

    if is_xl:
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16
            if device.type == "cuda"
            else torch.float32,
            use_safetensors=True,
            vae_path=vae_path,
        ).to(device)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16
            if device.type == "cuda"
            else torch.float32,
            use_safetensors=True,
            vae_path=vae_path,
        ).to(device)

    pipe.safety_checker = lambda images, clip_input: (
        images,
        [False] * len(images),
    )
    pipe.set_progress_bar_config(disable=False)
    pipe.enable_attention_slicing()

    if lora_path and os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()

    image_size = (1024, 1024) if is_xl else (512, 512)
    init_image = load_input_image(args.image_path, image_size)

    with torch.no_grad():
        result = pipe(
            prompt=args.prompt,
            image=init_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
        )

    output_image = result.images[0]
    output_path = os.path.splitext(args.image_path)[0] + "_edited.png"
    output_image.save(output_path)
    output_image.show()
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified SD/SDXL img2img script"
    )
    parser.add_argument(
        "--model", required=True, help="Model folder inside ~/models/"
    )
    parser.add_argument(
        "--image_path", required=True, help="Path or URL to input image"
    )
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="Strength of transformation",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Prompt guidance scale",
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument("--lora", help="LoRA file inside ~/models/loras/")
    args = parser.parse_args()
    main(args)
