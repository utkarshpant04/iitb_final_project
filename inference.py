# =============================================================
# GNR638 Project: Deep Learning MCQ Solver
# Model: Qwen2.5-VL-7B-Instruct (offline inference)
# =============================================================

import os
import re
import argparse
from collections import Counter

import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# =============================================================
# ARGS
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir",
    required=True,
    help="Absolute path to test directory containing test.csv and images/",
)
args = parser.parse_args()


# =============================================================
# PATHS
# =============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_FOLDER = os.path.join(args.test_dir, "images")
CSV_FILE = os.path.join(args.test_dir, "test.csv")
SAVE_FILE = os.path.join(BASE_DIR, "submission.csv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_qwen25vl_7b")

RUN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================
# PROMPTS
# =============================================================
PROMPTS = [
    """You are a deep learning expert solving an exam question.
Examine the MCQ image carefully.

IMPORTANT: Options A, B, C, D correspond to answer values 1, 2, 3, 4 respectively.

Instructions:
- Read the question and all options thoroughly
- Apply your deep learning knowledge
- Identify the single correct answer

End with:
ANSWER: [digit]""",

    """You are an expert deep learning engineer taking an exam.
Look at this MCQ image.
A=1, B=2, C=3, D=4.

Evaluate options carefully and choose the best answer.

Finish with:
FINAL: [digit]""",

    """Solve this deep learning MCQ.
A→1, B→2, C→3, D→4.

Determine the concept and best answer.

Output:
ANS: [digit]""",
]


# =============================================================
# PARSER
# =============================================================
MATCH_RULES = [
    r"ANSWER:\s*([1-5])",
    r"FINAL:\s*([1-5])",
    r"ANS:\s*([1-5])",
    r"answer\s+is\s+([1-5])",
    r"correct\s+(?:answer|option)\s+is\s+([1-5])",
]


# =============================================================
# ANSWER EXTRACTION
# =============================================================
def extract_choice(response_text: str) -> int:
    for expr in MATCH_RULES:
        found = re.search(expr, response_text, re.IGNORECASE)
        if found:
            return int(found.group(1))

    nums = re.findall(r"\b([1-5])\b", response_text)
    return int(nums[-1]) if nums else 5


# =============================================================
# IMAGE PREDICTION
# =============================================================
def solve_single_image(img_path: str) -> int:
    try:
        rgb_image = Image.open(img_path).convert("RGB")
    except Exception as err:
        print(f"  [ERROR] Could not load image: {err}")
        return 5

    collected_votes = []

    for idx, query in enumerate(PROMPTS):
        pred_num, model_text = generate_prediction(rgb_image, query)
        preview = model_text.strip().replace("\n", " ")[:100]
        print(f"  Prompt {idx + 1}: pred={pred_num}  raw={preview!r}")
        collected_votes.append(pred_num)

    acceptable = [x for x in collected_votes if x in (1, 2, 3, 4)]

    if not acceptable:
        return 5

    vote_counter = Counter(acceptable)
    top_answer, top_freq = vote_counter.most_common(1)[0]

    if top_freq < 2:
        return 5

    return top_answer


# =============================================================
# MODEL INFERENCE
# =============================================================
def generate_prediction(pil_img: Image.Image, instruction: str) -> tuple[int, str]:
    convo = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    compiled_text = processor.apply_chat_template(
        convo,
        tokenize=False,
        add_generation_prompt=True,
    )

    img_inputs, vid_inputs = process_vision_info(convo)

    prepared = processor(
        text=[compiled_text],
        images=img_inputs,
        videos=vid_inputs,
        padding=True,
        return_tensors="pt",
    ).to(RUN_DEVICE)

    with torch.no_grad():
        result_ids = model.generate(
            **prepared,
            max_new_tokens=400,
            do_sample=False,
        )

    trimmed = [
        full[len(inp):]
        for inp, full in zip(prepared.input_ids, result_ids)
    ]

    decoded_output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return extract_choice(decoded_output), decoded_output


# =============================================================
# MAIN EXECUTION
# =============================================================
if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Cannot find {CSV_FILE}")

    if not os.path.exists(IMG_FOLDER):
        raise FileNotFoundError(f"Cannot find images dir: {IMG_FOLDER}")

    print("Loading model (offline)...")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    print("Model loaded.\n")

    eval_df = pd.read_csv(CSV_FILE)

    print(f"Running inference on {len(eval_df)} images...\n")

    final_rows = []

    for _, sample in eval_df.iterrows():
        img_id = str(sample["image_name"])

        actual_file = img_id if img_id.endswith(".png") else f"{img_id}.png"
        full_img_path = os.path.join(IMG_FOLDER, actual_file)

        print(f"[{img_id}]")

        predicted_option = solve_single_image(full_img_path)

        print(f"  => Final prediction: {predicted_option}\n")

        final_rows.append(
            {
                "id": img_id,
                "image_name": img_id,
                "option": predicted_option,
            }
        )

    result_df = pd.DataFrame(final_rows)[["id", "image_name", "option"]]

    result_df.to_csv(SAVE_FILE, index=False)

    print(f"Saved {SAVE_FILE}")
    print(result_df.to_string(index=False))