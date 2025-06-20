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

# how many frames to sample uniformly from video
FRAME_SAMPLE_RATE = 2 # 1

DEFAULT_GEN_CFG = GenerationConfig(
    temperature=0.05,
    do_sample=True,
    max_new_tokens=128,
    min_new_tokens=20,
    repetition_penalty=1.5,
)


# ---------- basic helpers ---------- #
def load_video_frames(vr, indices):
    """Return list[PIL.Image] for given decord.VideoReader and frame IDs."""
    frames = vr.get_batch([i for i in indices if i < len(vr)]).numpy()
    return [PIL.Image.fromarray(f) for f in frames]

def _extract_image(img):
    if isinstance(img, PIL.Image.Image):
        return img.convert("RGB")
    if hasattr(img, "path"):
        return PIL.Image.open(img.path).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")

def _extract_video(video: Video, max_frames=5000):
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video.path)
    n = len(vr)
    if n < 1:
        raise ValueError(f"No frames in {video.path}")
    
    num_loaded_frames = min(n, max_frames)
    indices = torch.arange(num_loaded_frames)    
    return load_video_frames(vr, indices)

def extract_media(messages):
    """Replace Image/Video objects in conversation with special tokens."""
    media = {"video": [], "image": []}
    for msg in messages:
        val = ""
        for part in make_list(msg["value"]):
            if isinstance(part, str):
                for tok in MEDIA_TOKENS.values():
                    part = part.replace(tok, "")
                val += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                media["image"].append(_extract_image(part))
                val += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                media["video"].append(_extract_video(part))
                val += MEDIA_TOKENS["video"]
        msg["value"] = val
    return media

def pad_to_multiple_of_8(frames, timestamps):
    rem = len(frames) % 8
    frames = list(frames)
    timetamps = list(timestamps)
    rem = len(frames) % 8
    if rem:
        pad_len = 8 - rem
        frames.extend([frames[-1]] * pad_len)
        timetamps.extend([timetamps[-1]] * pad_len)
    return frames, timetamps


def uniform_sample(frames, num=500):
    if len(frames) == 0:
        return [], np.array([])
    indices = np.linspace(0, len(frames) - 1, min(num, len(frames))).astype(int)
    sampled_frames = [frames[i] for i in indices]
    
    sampled_frames, indices = pad_to_multiple_of_8(sampled_frames, indices)
    
    return sampled_frames, indices

def split_chunks(frames, timestamps, length, overlap=4):
    total = len(frames)
    if total<=length: return [ pad_to_multiple_of_8(frames, timestamps) ]
    step = max(length-overlap, (total-length)//(total//length or 1))
    chunks=[]
    for i in range(0, total-length+1, step):
        chunk_f = frames[i : i + length]
        chunk_t = timestamps[i : i + length]
        chunks.append(pad_to_multiple_of_8(chunk_f, chunk_t))
    if chunks and chunks[-1][0][-1]!=frames[-1]:
        chunk_f = frames[-length:]
        chunk_t = timestamps[-length:]
        chunks.append(pad_to_multiple_of_8(chunk_f, chunk_t))

    return chunks

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

def process_request(
        model, tokenizer, image_processor,
        prompt: str,
        video_path: str,
        gen_cfg: GenerationConfig,
        current_video=None
):
    # Build conversation ================================================
    # conv = [{"from": "system", "value": prompt}]
    # conv.append({"from": "human", "value": [Video(video_path)]})
    conv = [{"from": "human", "value": prompt}]
    conv.append({"from": "human", "value": [Video(video_path)]})


    # Replace visual objects with tokens and get real media -------------
    conv_tokens = deepcopy(conv)
    media = extract_media(conv_tokens)

    video_frames = media["video"][0]
    shot_images  = media["image"]

    # Chunk the video if needed -----------------------------------------
    budget = model.config.num_video_frames

    sampled_video, indices = uniform_sample(video_frames, len(video_frames) // FRAME_SAMPLE_RATE)
    
    log.info(f"Length of sampled_video video: {len(sampled_video)}, original: {len(video_frames)}")
    stamps = np.array(indices) / 30  # default FPS

    chunks = split_chunks(sampled_video, stamps, budget)

    answers = []
    for v_chunk, ts in chunks:
        tag = f"{ts[0]:.2f}s–{ts[-1]:.2f}s" if len(ts) else "0s"
        resp = generate_predictions(
            model, tokenizer, image_processor,
            shot_images, v_chunk, conv_tokens, gen_cfg
        )
        answers.append({"timestamp": tag, "response": resp})
    return answers

# -------------------------------------------------------------------- #
#                            ----  MAIN  ----                          #
# -------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="dataset")
    ap.add_argument("--video_root",  required=True,
                    help="Root folder that contains the videos")
    ap.add_argument("--experiment", default="experiment")
    ap.add_argument("--model_path",  default="NVILA-8B-Video")
    ap.add_argument("--eval_json",   help="Resume / append to existing json")
    ap.add_argument("--json_instructions", required=True, help="Path to the json with prompts and questions")
    args = ap.parse_args()

    # -------------- logging / counters ----------------
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s - %(message)s")
    global log
    log = logging.getLogger("eval")

    correct, incorrect = defaultdict(int), defaultdict(int)
    top3_correct = defaultdict(int)

    # -------------- load model ONCE -------------------
    disable_torch_init()
    model_name = os.path.basename(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_name, None
    )
    gen_cfg = DEFAULT_GEN_CFG
    log.info("Model loaded: %s", model_name)
    log.info("Model device: %s", model.device)

    # -------------- load json instructions -------------------
    with open(args.json_instructions) as f:
        instruction_data = json.load(f)
    instruction_map = {Path(item["image_path"]).name: item for item in instruction_data}


    # -------------- zero-shot classifier --------------
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")

    # -------------- evaluation list -------------------
    pattern = os.path.join(args.video_root, "**", "*.mp4")
    video_list = []
    for inst in instruction_map.values():
        video_path = inst["image_path"]
        full_path = os.path.join(args.video_root, video_path)
        video_list.append((full_path, inst))

    video_list = sorted(video_list)

    # -------------- resume? ---------------------------
    if args.eval_json and os.path.exists(args.eval_json):
        with open(args.eval_json) as f:
            eval_dict = json.load(f)
    else:
        direc = f"logs/{args.dataset_name}/{args.experiment}/{model_name}"
        os.makedirs(direc, exist_ok=True)
        args.eval_json = os.path.join(
            direc, f"eval_{time.strftime('%Y%m%d-%H%M%S')}.json")
        eval_dict = {}

    processed_idxs  = {int(k) for k in eval_dict.keys() if k.isdigit()}
    processed_names = {v["video_name"] for v in eval_dict.values()}

    # -------------- loop over videos ------------------
    for vid_idx, (video_path, instruction) in enumerate(tqdm(video_list)):

        if vid_idx in processed_idxs or Path(video_path).name in processed_names:
            log.info(f"Skipping already-processed [{vid_idx}] {video_path}")
            continue

        prompt = instruction["context"] #PROMPT
        CLASSES = [instruction["ans0"], instruction["ans1"], instruction["ans2"]]
        gt_label = instruction["label"]

        answer_chunks = process_request(
            model, tokenizer, image_processor,
            prompt, video_path, gen_cfg,
            current_video=video_path
        )

        # ---- post-process each chunk ------------------
        best_pred, best_score = None, -1.0
        top3 = set()
        output_dict = {}
        
        for chunk in answer_chunks:
            text = chunk["response"]
            sample = f"{instruction['context']} {text}" 


            q = instruction["question"]#QUESTION_BASED_ON_OUTPUT
            sample = q + text
            pred = classifier(sample, candidate_labels=CLASSES)
            labels = pred["labels"][:3]
            scores = pred["scores"][:3]

            # keep best top-1
            if scores[0] > best_score:
                best_score = scores[0]
                best_pred  = labels[0]
            top3.update(labels)

            timestamp = chunk["timestamp"]
            output_dict[timestamp] = {
                "LLM_output": text,
                "LLM_classes": labels,
                "LLM_confidence": scores,
            }

        # ---- scoring ----------------------------------
        top1 = 0.0
        if best_pred == gt_label:
            correct[gt_label] += 1
            top1 = 1.0
        else:
            incorrect[gt_label] += 1

        if gt_label in top3:
            top3_correct[gt_label] += 1

        # ---- save incremental -------------------------
        eval_dict[str(vid_idx)] = {
            "video_name": Path(video_path).name,
            "video_label": gt_label,
            "prediction": CLASSES.index(best_pred),
            "predicted_answer": best_pred,
            "video_prediction": output_dict,
            "top_1_score": top1,
            "instruction_idx": instruction["idx"],
            "question": q,
            "answers": CLASSES,
            "true_answer": CLASSES[gt_label],
            "category": instruction["category"]
        }
        
        with open(args.eval_json, "w") as f:
            json.dump(eval_dict, f, indent=2)

    # -------------- final metrics ---------------------
    total = sum(correct.values()) + sum(incorrect.values())
    overall_acc = (sum(correct.values()) / total) * 100 if total else 0
    top3_acc    = (sum(top3_correct.values()) / total) * 100 if total else 0

    log.info("Overall  accuracy: %.2f%%", overall_acc)
    log.info("Top-3    accuracy: %.2f%%", top3_acc)
    log.info("Results written to %s", args.eval_json)


if __name__ == "__main__":
    main()
