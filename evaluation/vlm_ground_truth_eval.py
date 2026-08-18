from pathlib import Path
import json
import random
import re

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

NUM_SAMPLES = 20

RANDOM_SEED = 42

MAX_IMAGE_SIZE = 640

MAX_NEW_TOKENS = 128


# COCO class id
TARGET_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "traffic light": 9,
    "umbrella": 25
}


# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "coco128"
)

IMAGE_DIR = (
    DATASET_ROOT
    / "images"
    / "train2017"
)

LABEL_DIR = (
    DATASET_ROOT
    / "labels"
    / "train2017"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "vlm"
    / "ground_truth_eval"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "results.json"
)


# ============================================
# Image
# ============================================

def load_resized_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image.thumbnail(
        (
            MAX_IMAGE_SIZE,
            MAX_IMAGE_SIZE
        )
    )

    return image


# ============================================
# Ground Truth
# ============================================

def read_ground_truth(image_path):

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    present_class_ids = set()


    if label_path.exists():

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(
                    float(parts[0])
                )

                present_class_ids.add(
                    class_id
                )


    ground_truth = {}

    for class_name, class_id in TARGET_CLASSES.items():

        ground_truth[class_name] = (
            class_id in present_class_ids
        )


    return ground_truth


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
# Normalize VLM Result
# ============================================

def normalize_vlm_result(
    result
):

    normalized = {}

    for class_name in TARGET_CLASSES:

        value = result.get(
            class_name,
            False
        )

        normalized[class_name] = bool(
            value
        )


    return normalized


# ============================================
# Prompt
# ============================================

def build_prompt():

    return (
        "Analyze this image and determine whether "
        "each listed object is visible. "
        "Return ONLY valid JSON with boolean values. "
        "Use exactly these keys: "
        "person, bicycle, car, motorcycle, bus, "
        "traffic light, umbrella. "
        "Do not include explanations. "
        "Example: "
        '{"person": true, '
        '"bicycle": false, '
        '"car": true, '
        '"motorcycle": false, '
        '"bus": false, '
        '"traffic light": true, '
        '"umbrella": false}'
    )


# ============================================
# VLM Inference
# ============================================

def run_vlm(
    model,
    processor,
    image_path
):

    image = load_resized_image(
        image_path
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
                    "text": build_prompt()
                }
            ]
        }
    ]


    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
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


    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]


    parsed = extract_json(
        output_text
    )


    return (
        normalize_vlm_result(
            parsed
        ),
        output_text
    )


# ============================================
# Metric Calculation
# ============================================

def calculate_metrics(
    all_ground_truth,
    all_predictions
):

    tp = 0
    tn = 0
    fp = 0
    fn = 0


    for ground_truth, prediction in zip(
        all_ground_truth,
        all_predictions
    ):

        for class_name in TARGET_CLASSES:

            gt = ground_truth[
                class_name
            ]

            pred = prediction[
                class_name
            ]


            if gt and pred:
                tp += 1

            elif not gt and not pred:
                tn += 1

            elif not gt and pred:
                fp += 1

            elif gt and not pred:
                fn += 1


    total = (
        tp
        + tn
        + fp
        + fn
    )


    accuracy = (
        (tp + tn)
        / total
        if total > 0
        else 0.0
    )


    precision = (
        tp
        / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    recall = (
        tp
        / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================
# Main
# ============================================

def main():

    print("=" * 60)
    print("VLM Ground Truth Evaluation")
    print("=" * 60)


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ----------------------------------------
    # Find Images
    # ----------------------------------------

    image_paths = sorted(
        [
            path

            for path in IMAGE_DIR.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png"
                }
            )
        ]
    )


    if not image_paths:

        raise RuntimeError(
            "No images found."
        )


    # ----------------------------------------
    # Sample Images
    # ----------------------------------------

    random.seed(
        RANDOM_SEED
    )

    samples = random.sample(
        image_paths,
        min(
            NUM_SAMPLES,
            len(image_paths)
        )
    )


    print(
        f"Images selected : "
        f"{len(samples)}"
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


    # ----------------------------------------
    # Evaluation
    # ----------------------------------------

    results = []

    all_ground_truth = []

    all_predictions = []


    for index, image_path in enumerate(
        samples,
        start=1
    ):

        print()

        print(
            f"[{index}/{len(samples)}] "
            f"{image_path.name}"
        )


        ground_truth = (
            read_ground_truth(
                image_path
            )
        )


        try:

            prediction, raw_output = (
                run_vlm(
                    model,
                    processor,
                    image_path
                )
            )


        except Exception as error:

            print(
                f"[ERROR] "
                f"{image_path.name}: "
                f"{error}"
            )

            continue


        print(
            f"GT  : "
            f"{ground_truth}"
        )

        print(
            f"VLM : "
            f"{prediction}"
        )


        results.append(
            {
                "image": (
                    image_path.name
                ),
                "ground_truth": (
                    ground_truth
                ),
                "prediction": (
                    prediction
                ),
                "raw_output": (
                    raw_output
                )
            }
        )


        all_ground_truth.append(
            ground_truth
        )

        all_predictions.append(
            prediction
        )


    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    metrics = calculate_metrics(
        all_ground_truth,
        all_predictions
    )


    print()

    print("=" * 60)
    print("Final Metrics")
    print("=" * 60)

    print(
        f"Evaluated Images : "
        f"{len(results)}"
    )

    print(
        f"TP : "
        f"{metrics['tp']}"
    )

    print(
        f"TN : "
        f"{metrics['tn']}"
    )

    print(
        f"FP : "
        f"{metrics['fp']}"
    )

    print(
        f"FN : "
        f"{metrics['fn']}"
    )

    print()

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score  : "
        f"{metrics['f1']:.4f}"
    )

    print("=" * 60)


    # ----------------------------------------
    # Save JSON
    # ----------------------------------------

    output_data = {
        "model": MODEL_NAME,
        "num_samples_requested": (
            NUM_SAMPLES
        ),
        "num_samples_evaluated": (
            len(results)
        ),
        "target_classes": (
            list(
                TARGET_CLASSES.keys()
            )
        ),
        "metrics": metrics,
        "results": results
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