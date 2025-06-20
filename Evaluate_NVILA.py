#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified evaluation script for VILA / NVILA video anomaly detection.

• Loads the VILA / NVILA model once.
• Feeds each video directly into the model – no FastAPI, no OpenAI client.
• Runs the same zero-shot-classification post-processing you already had.
"""
from __future__ import annotations
import os, argparse, time, json, glob, logging
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import numpy as np
import PIL
import torch
from tqdm import tqdm
from transformers import pipeline, GenerationConfig
import decord

# -------------------------------------------------------------------- #
#                    --------  MODEL UTILITIES --------                #
# -------------------------------------------------------------------- #
from llava import conversation as clib
from llava.model.builder import load_pretrained_model
from llava.media import Image, Video
from llava.mm_utils import process_images
from llava.utils import disable_torch_init, make_list, tokenizer as tok_utils
from llava.constants import MEDIA_TOKENS

FRAME_SAMPLE_RATE = 2 # 1

DEFAULT_GEN_CFG = GenerationConfig(
    temperature=0.05,
    do_sample=True,
    max_new_tokens=128,
    min_new_tokens=20,
    repetition_penalty=1.5,
)

@torch.inference_mode()
def generate_predictions(
        model, tokenizer, image_processor,
        shot_images, video_frames, conv,
        gen_cfg: GenerationConfig
):
    video_enc = process_images(video_frames, image_processor, model.config).half()

    media = {"video": [video_enc]}
    if shot_images:
        shot_enc = process_images(shot_images, image_processor, model.config).half()
        media["image"] = [t for t in shot_enc]

    inputs = tok_utils.tokenize_conversation(
        conv, tokenizer, add_generation_prompt=True
    ).to(model.device).unsqueeze(0)

    out = model.generate(
        input_ids=inputs,
        media=media,
        media_config=defaultdict(dict),
        generation_config=gen_cfg,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()



# -------------------------------------------------------------------- #
#                            ----  MAIN  ----                          #
# -------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="close_ended_dataset")
    # ap.add_argument("--video_root",  required=True,
    #                 help="Root folder that contains the videos")
    # ap.add_argument("--experiment", default="Few_Shot")
    ap.add_argument("--model_path",  default="NVILA-8B-Video")
    ap.add_argument("--eval_json",   help="Resume / append to existing json")
    # ap.add_argument("--class_to_test")
    args = ap.parse_args()

    # -------------- logging / counters ----------------
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s - %(message)s")
    global log
    log = logging.getLogger("eval")
    # log.info(f"Few shot examples: {FEW_SHOT_EXAMPLES}")

    correct, incorrect = defaultdict(int), defaultdict(int)
    adjusted_correct, top3_correct = defaultdict(float), defaultdict(int)

    # -------------- load model ONCE -------------------
    disable_torch_init()
    model_name = os.path.basename(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path = args.model_path, model_name = model_name, model_base = None)
    gen_cfg = DEFAULT_GEN_CFG
    log.info("Model loaded: %s", model_name)
    log.info("Model device: %s", model.device)

    # video_list = sorted(video_list)
    from dataset import load_dataset

    if os.path.exists(args.eval_json):
        with open(args.eval_json, "r") as f:
            result_json = json.load(f)
        already_done_ids = set(x["id"] for x in result_json)
    else:
        result_json = []
        already_done_ids = set()

    dataset = load_dataset.load_dataset(args.dataset_name)
    for idx, test_case in enumerate(dataset):
        #skip idx if already done
        if idx in already_done_ids:
            log.info(f"Skipping already-processed [{idx}] {test_case['instruction']}")
            continue

        if "otter" in model_name and idx <= 80509:
            continue
        try:
            # pred = model.generate(
            #     instruction=test_case['instruction'],
            #     images=test_case['images'],
            # )

            media = {}
            shot_enc = process_images(test_case['images'], image_processor, model.config).half()
            # media["image"] = [t for t in shot_enc]
            media["image"] = [shot_enc[0]]
            log.info(f'Images processed: {len(media["image"]),type(media["image"])}')
            
            log.info('step 1 done')
            instruction = f"<image>\n {test_case['instruction']}"
            messages = [{"from": "human", "value": instruction}]
            
            log.info(type(test_case['instruction']))
            
            log.info(f"Instruction: {instruction}")
            log.info(f"# of <image> tokens: {instruction.count('<image>')}")
            log.info(f"# of image embeddings: {len(media['image'])}")

            # messages = [{"role": "user", "content": test_case["instruction"]}]
            # messages = [{"from": "human", "value": test_case["instruction"]}]
            log.info(type(messages))
            inputs = tok_utils.tokenize_conversation(
                messages, tokenizer, add_generation_prompt=True).to(model.device).unsqueeze(0)

            log.info('step 2 done')

            pred = model.generate(
                input_ids=inputs,
                media=media,
                media_config=defaultdict(dict),
                generation_config=gen_cfg)
            
            log.info('step 3 done')

            answer = tokenizer.decode(pred[0], skip_special_tokens=True).strip()

            log.info('step 4 done')
            
            
        except Exception as error:
            pred = ""
            log.error(f'An exception occurred: {error}')
            
        log.info(f'ID:\t{idx}')
        log.info(f'Instruction:\t{test_case["instruction"]}')
        #log.info(f'Images:\t{test_case["images"]}')
        log.info(f'Answer:\t{answer}')
        log.info('-' * 60)

        result_case = {
            'id': idx if test_case.get('id') is None else test_case['id'],
            'instruction': test_case['instruction'],
            'in_images': test_case['images'],
            'answer': answer,
        }
        result_json.append(result_case)

        # Save every 10th result to avoid losing progress
        SAVE_EVERY = 10
        if idx % SAVE_EVERY == 0:
            os.makedirs(os.path.dirname(args.eval_json), exist_ok=True)
            with open(args.eval_json, "w") as json_file:
                json.dump(result_json, json_file, indent=4)

    # final save 
    os.makedirs(os.path.dirname(args.eval_json), exist_ok=True)
    with open(args.eval_json, "w") as json_file:
        json.dump(result_json, json_file, indent=4)

    # -------------- final metrics ---------------------
    total = sum(correct.values()) + sum(incorrect.values())
    overall_acc = (sum(correct.values()) / total) * 100 if total else 0
    adj_acc     = (sum(adjusted_correct.values()) / total) * 100 if total else 0
    top3_acc    = (sum(top3_correct.values()) / total) * 100 if total else 0

    log.info("Overall  accuracy: %.2f%%", overall_acc)
    log.info("Adjusted accuracy: %.2f%%", adj_acc)
    log.info("Top-3    accuracy: %.2f%%", top3_acc)
    log.info("Results written to %s", args.eval_json)


if __name__ == "__main__":
    main()
