from pathlib import Path

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)


# ============================================
# Configuration
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

MAX_NEW_TOKENS = 128

MAX_IMAGE_SIZE = 640


# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "vlm"
    / "city_frame.jpg"
)


# ============================================
# Image Resize
# ============================================

def load_resized_image():

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image.thumbnail(
        (
            MAX_IMAGE_SIZE,
            MAX_IMAGE_SIZE
        )
    )

    return image


# ============================================
# VLM Inference
# ============================================

def run_vlm():

    print("=" * 60)
    print("Qwen2.5-VL Structured Image Inference")
    print("=" * 60)

    print(
        f"Model : {MODEL_NAME}"
    )

    print(
        f"Image : {IMAGE_PATH}"
    )

    print(
        f"Max image size : "
        f"{MAX_IMAGE_SIZE}"
    )

    print("=" * 60)


    # ----------------------------------------
    # Check Image
    # ----------------------------------------

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{IMAGE_PATH}"
        )


    # ----------------------------------------
    # Load Image
    # ----------------------------------------

    image = load_resized_image()

    print(
        f"[INFO] Resized image: "
        f"{image.size}"
    )


    # ----------------------------------------
    # Load Processor
    # ----------------------------------------

    print(
        "[INFO] Loading processor..."
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=640 * 640
    )


    # ----------------------------------------
    # Load Model
    # ----------------------------------------

    print(
        "[INFO] Loading model..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="cpu"
        )
    )

    model.eval()


    # ========================================
    # Structured Prompt
    # ========================================

    prompt = (
        "Analyze the image and return ONLY valid JSON "
        "with boolean values for the following objects: "
        "person, car, bus, bicycle, motorcycle, "
        "traffic light, umbrella. "
        "Use exactly these keys and do not add any explanation. "
        "Example format: "
        '{"person": true, "car": true, "bus": false, '
        '"bicycle": false, "motorcycle": false, '
        '"traffic light": true, "umbrella": false}'
    )


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]


    # ----------------------------------------
    # Chat Template
    # ----------------------------------------

    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )


    # ----------------------------------------
    # Processor Input
    # ----------------------------------------

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )


    # ----------------------------------------
    # Generate
    # ----------------------------------------

    print(
        "[INFO] Running inference..."
    )

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )


    # ----------------------------------------
    # Remove Prompt Tokens
    # ----------------------------------------

    generated_ids_trimmed = [

        output_ids[
            len(input_ids):
        ]

        for input_ids, output_ids

        in zip(
            inputs.input_ids,
            generated_ids
        )
    ]


    # ----------------------------------------
    # Decode
    # ----------------------------------------

    output_text = (
        processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    )


    # ----------------------------------------
    # Result
    # ----------------------------------------

    print()

    print("=" * 60)
    print("VLM Structured Result")
    print("=" * 60)

    print(
        output_text[0]
    )

    print("=" * 60)


# ============================================
# Main
# ============================================

if __name__ == "__main__":

    run_vlm()