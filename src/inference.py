import argparse
import gc
import json
import math
import os
import os.path as osp
from pprint import pprint
import pickle
import sys

sys.path.append(osp.join(osp.dirname(osp.dirname(__file__))))

import time
from types import SimpleNamespace

import torch
from PIL import Image
from tqdm import tqdm
import yaml

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from src.conversation import conv_templates
from src.model.builder import load_pretrained_model
from src.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from src.utils import disable_torch_init

from src.logic import DimProspector, HeadFork, VARProcessor, LogicEngine
from src.stash import StashEngine, MetadataStation


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_safe_qid(line, idx):
    """
    Avoid:
        line.get("qid") or line.get("question_id")

    because qid=0 is valid, but bool(0) == False.
    """
    for key in ["qid", "question_id", "id", "image_id"]:
        if key in line and line[key] is not None:
            try:
                return int(line[key])
            except Exception:
                return idx

    return idx


def get_safe_image_file(line):
    image_file = (
        line.get("image", None)
        or line.get("image_path", None)
        or line.get("filename", None)
        or line.get("file_name", None)
    )

    if image_file is None:
        raise ValueError(f"Cannot find image field in line: {line}")

    return image_file


def get_safe_question(line):
    qs = (
        line.get("text", None)
        or line.get("question", None)
        or line.get("prompt", None)
    )

    if qs is None:
        raise ValueError(f"Cannot find question/text/prompt field in line: {line}")

    return qs


def get_safe_label(line):
    return (
        line.get("label", None)
        or line.get("answer", None)
        or line.get("gt-label", None)
        or line.get("gt_label", None)
    )


def resolve_image_path(path_image_dir, image_file):
    """
    Support both:
      1. image_file is relative to path_image_dir
      2. image_file is already an absolute path
    """
    image_file = str(image_file)

    if osp.isabs(image_file):
        image_path = image_file
    else:
        image_path = osp.join(path_image_dir, image_file)

    if osp.exists(image_path):
        return image_path, image_file

    root, ext = osp.splitext(image_path)

    if ext == "":
        for suffix in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            candidate = image_path + suffix
            if osp.exists(candidate):
                return candidate, image_file + suffix

    raise FileNotFoundError(f"Cannot find image: {image_path}")


def _candidate_token_ids(tokenizer, word):
    """
    Collect single-token ids for lowercase / capitalized / space-prefixed variants.

    Vicuna/LLaVA tokenizers are case-sensitive, so:
        left / Left / " left" / " Left"
    may have very different probabilities.
    """
    variants = [
        word,
        word.capitalize(),
        " " + word,
        " " + word.capitalize(),
    ]

    out = []

    for surface in variants:
        try:
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
        except Exception:
            token_ids = []

        if len(token_ids) == 1:
            out.append((surface, int(token_ids[0])))

    return out


def fallback_relation_from_scores(outputs, tokenizer, relations=None):
    """
    If free generation returns empty string, choose from fixed relation words
    using the first generation step logits.

    This is suitable for Controlled_Images_A:
        left / right / on / under
    """
    if relations is None:
        relations = ["left", "right", "on", "under"]

    if not hasattr(outputs, "scores"):
        return ""

    if outputs.scores is None or len(outputs.scores) == 0:
        return ""

    logits = outputs.scores[0][0].detach().float()
    probs = torch.softmax(logits, dim=-1)

    best_rel = ""
    best_prob = -1.0
    best_surface = None
    best_tid = None

    for rel in relations:
        for surface, tid in _candidate_token_ids(tokenizer, rel):
            p = float(probs[tid].item())

            if p > best_prob:
                best_prob = p
                best_rel = rel
                best_surface = surface
                best_tid = tid

    return best_rel


def decode_generated_text(outputs, input_ids, tokenizer):
    """
    Some generate implementations return:
        prompt + generated tokens
    while others return:
        only generated tokens

    Decode robustly. If decoded text is empty, return "" and let caller fallback.
    """
    seq = outputs.sequences

    if seq.shape[1] > input_ids.shape[1]:
        generated_ids = seq[:, input_ids.shape[1] :]
    else:
        generated_ids = seq

    text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return text, generated_ids


def eval_model(args):
    with open(args.exp_config, "r") as file:
        config_dict = yaml.safe_load(file)

    cfgs = SimpleNamespace(**config_dict)

    if torch.cuda.is_available():
        device_id = 0 if args.device is None else args.device
        device = f"cuda:{device_id}"
    else:
        device_id = None
        device = "cpu"

    cfgs.device = device

    print("\n\n\n")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(device_id)}-{device_id}")
    else:
        print("Using device: CPU")
    pprint(vars(cfgs))
    print("\n\n\n")

    disable_torch_init()

    path_model = os.path.expanduser(cfgs.path_model)
    name_model = get_model_name_from_path(path_model)
    cfgs.name_model = name_model

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        path_model,
        args.model_base,
        name_model,
        attn_implementation="eager",
        device_map=device,
    )

    # Activate StashEngine
    MetadataStation.activate()
    MetadataStation.export_model_config(model.config)

    # Activate logic / VAR
    if getattr(cfgs, "logic", 0) == 1:
        LogicEngine.activate(
            tau=cfgs.tau,
            rho=cfgs.rho,
            summ=cfgs.summ,
            p=cfgs.p,
            except_last_layer=cfgs.except_last_layer,
        )

    question_file_name = (
        f"{cfgs.name_category}-questions.jsonl"
        if cfgs.name_category != ""
        else "questions.jsonl"
    )

    question_file_path = osp.join(cfgs.path_question_dir, question_file_name)
    question_file_path = os.path.expanduser(question_file_path)

    questions = [json.loads(q) for q in open(question_file_path, "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answer_file_ver = (
        f"[{cfgs.name_daset}-{cfgs.name_category}]"
        f"{cfgs.name_exp}-{str(int(time.time()))}"
    )

    answers_file = osp.join("E_answers", cfgs.name_model, f"{answer_file_ver}.jsonl")
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(osp.dirname(answers_file), exist_ok=True)

    file_mode = "w"

    with open(answers_file, file_mode) as ans_file:
        setattr(model, "tokenizer", tokenizer)

        for idx, line in enumerate(tqdm(questions)):
            try:
                qid = get_safe_qid(line, idx)
                gt_label = get_safe_label(line)

                image_file = get_safe_image_file(line)
                qs = get_safe_question(line)

                image_path, image_file_for_output = resolve_image_path(
                    cfgs.path_image_dir,
                    image_file,
                )

                cur_prompt = qs

                if model.config.mm_use_im_start_end:
                    qs = (
                        DEFAULT_IM_START_TOKEN
                        + DEFAULT_IMAGE_TOKEN
                        + DEFAULT_IM_END_TOKEN
                        + "\n"
                        + qs
                    )
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_templates[cfgs.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    conv=conv,
                    return_tensors="pt",
                ).unsqueeze(0).to(device=device)

                image = Image.open(image_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0]

                with torch.inference_mode():
                    with torch.no_grad():
                        setattr(model, "tokenizer", tokenizer)

                        outputs = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().to(device),
                            image_sizes=[image.size],
                            return_dict_in_generate=True,
                            output_attentions=True,
                            output_hidden_states=True,
                            output_scores=True,
                            do_sample=False,
                            max_new_tokens=cfgs.max_new_tokens,
                            min_new_tokens=1,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                generated_texts, generated_ids = decode_generated_text(
                    outputs=outputs,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                )

                used_fallback = False

                # If generation is empty / only special tokens, use first-step logits.
                if generated_texts == "":
                    generated_texts = fallback_relation_from_scores(
                        outputs=outputs,
                        tokenizer=tokenizer,
                        relations=["left", "right", "on", "under"],
                    )
                    used_fallback = True

                ans_file.write(
                    json.dumps(
                        {
                            "question_id": qid,
                            "prompt": cur_prompt,
                            "label": gt_label,
                            "response": generated_texts,
                            "used_fallback": used_fallback,
                            "raw_sequence_len": int(outputs.sequences.shape[1]),
                            "input_len": int(input_ids.shape[1]),
                            "generated_len": int(generated_ids.shape[1]),
                            "image": image_file_for_output,
                            "model_id": name_model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                ans_file.flush()

                del outputs
                StashEngine.clear()
                LogicEngine.clear()

            except Exception as e:
                print("\n[ERROR] Failed on sample:")
                print(json.dumps(line, ensure_ascii=False, indent=2))
                raise e

    print(f"\nSaved answers to: {answers_file}")


def parse_ranges(range_string):
    st = [int(num) for num in range_string.split("-")][:-1]
    ed = [int(num) for num in range_string.split("-")][1:]
    return [[st, ed] for st, ed in zip(st, ed)]


if __name__ == "__main__":
    if sys.argv[-1] in ["debug", "--debug"]:
        import debugpy

        debugpy.listen(("localhost", 7739))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        sys.argv.pop()

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=int, default=None)

    # logic
    parser.add_argument("--logic", type=str, default=None)
    parser.add_argument("--dim_prospector", action="store_true")
    parser.add_argument("--head_fork", action="store_true")
    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--sink_rule", type=str, default="ours")
    parser.add_argument("--head_rule", type=str, default="ours")

    # exp config
    parser.add_argument("--exp_config", type=str, default=None)

    args = parser.parse_args()

    eval_model(args)
