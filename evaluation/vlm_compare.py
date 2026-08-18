from pathlib import Path
import json
import re

import torch
from PIL import Image
from ultralytics import YOLO

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

from config import Config


# ============================================
# Configuration
# ============================================

VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

TARGET_CLASSES = [
    "person",
    "car",
    "bus",
    "bicycle",
    "motorcycle",
    "traffic light",
    "umbrella"
]

YOLO_CONF_THRESHOLD = 0.25

MAX_IMAGE_SIZE = 640

MAX_NEW_TOKENS = 128


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "vlm"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "vlm_comparison.json"
)


# ============================================
# YOLO Detection
# ============================================

def run_yolo():

    print("[INFO] Running YOLO...")

    model = YOLO(
        Config.MODEL_PATH
    )

    results = model(
        str(IMAGE_PATH),
        conf=YOLO_CONF_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    detected_classes = []

    for box in result.boxes:

        class_id = int(
            box.cls.item()
        )

        class_name = result.names[
            class_id
        ]

        detected_classes.append(
            class_name
        )


    presence = {}

    for class_name in TARGET_CLASSES:

        presence[class_name] = (
            class_name in detected_classes
        )


    return (
        presence,
        detected_classes
    )


# ============================================
# Image
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
# JSON Parsing
# ============================================

def extract_json(text):

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match is None:

        raise ValueError(
            "No JSON object found "
            "in VLM output."
        )

    return json.loads(
        match.group(0)
    )


# ============================================
# VLM
# ============================================

def run_vlm():

    print(
        "[INFO] Loading VLM processor..."
    )

    processor = (
        AutoProcessor.from_pretrained(
            VLM_MODEL_NAME,
            min_pixels=256 * 28 * 28,
            max_pixels=640 * 640
        )
    )


    print(
        "[INFO] Loading VLM model..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            VLM_MODEL_NAME,
            torch_dtype="auto",
            device_map="cpu"
        )
    )

    model.eval()


    image = load_resized_image()


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


    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )


    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )


    print(
        "[INFO] Running VLM..."
    )


    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )


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


    output_text = (
        processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    )


    print()
    print("Raw VLM Output:")
    print(output_text)


    result = extract_json(
        output_text
    )


    return result


# ============================================
# Normalize VLM Result
# ============================================

def normalize_vlm_result(
    vlm_result
):

    normalized = {}

    for class_name in TARGET_CLASSES:

        value = vlm_result.get(
            class_name,
            False
        )

        normalized[class_name] = bool(
            value
        )

    return normalized


# ============================================
# Comparison
# ============================================

def compare(
    yolo_result,
    vlm_result
):

    comparison = {}

    matches = 0


    for class_name in TARGET_CLASSES:

        yolo_value = yolo_result[
            class_name
        ]

        vlm_value = vlm_result[
            class_name
        ]

        is_match = (
            yolo_value
            == vlm_value
        )


        comparison[
            class_name
        ] = {
            "yolo": yolo_value,
            "vlm": vlm_value,
            "match": is_match
        }


        if is_match:

            matches += 1


    agreement = (
        matches
        / len(TARGET_CLASSES)
    )


    return (
        comparison,
        agreement
    )


# ============================================
# Main
# ============================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 60)
    print("YOLO vs VLM Comparison")
    print("=" * 60)

    print(
        f"Image : {IMAGE_PATH}"
    )

    print("=" * 60)


    # ----------------------------------------
    # YOLO
    # ----------------------------------------

    (
        yolo_presence,
        detected_classes
    ) = run_yolo()


    print()
    print(
        "YOLO Detected Classes:"
    )

    print(
        detected_classes
    )


    # ----------------------------------------
    # VLM
    # ----------------------------------------

    vlm_raw = run_vlm()

    vlm_presence = (
        normalize_vlm_result(
            vlm_raw
        )
    )


    # ----------------------------------------
    # Compare
    # ----------------------------------------

    (
        comparison,
        agreement
    ) = compare(
        yolo_presence,
        vlm_presence
    )


    print()
    print("=" * 60)
    print("Comparison Results")
    print("=" * 60)


    for class_name in TARGET_CLASSES:

        result = comparison[
            class_name
        ]

        print(
            f"{class_name:15} "
            f"YOLO={str(result['yolo']):5} "
            f"VLM={str(result['vlm']):5} "
            f"Match={result['match']}"
        )


    print()

    print(
        f"Agreement Rate : "
        f"{agreement * 100:.2f}%"
    )

    print("=" * 60)


    # ----------------------------------------
    # Save Results
    # ----------------------------------------

    output_data = {
        "image": str(
            IMAGE_PATH
        ),

        "target_classes": (
            TARGET_CLASSES
        ),

        "yolo_detected_classes": (
            detected_classes
        ),

        "yolo_presence": (
            yolo_presence
        ),

        "vlm_presence": (
            vlm_presence
        ),

        "comparison": (
            comparison
        ),

        "agreement_rate": (
            agreement
        )
    }


    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False
        )


    print(
        f"Saved : "
        f"{OUTPUT_JSON}"
    )


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":

    main()