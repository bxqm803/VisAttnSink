import argparse
import json
import math
import os
import os.path as osp
from pprint import pprint
import sys
import time
from types import SimpleNamespace

sys.path.append(osp.join(osp.dirname(osp.dirname(__file__))))

import torch
import yaml
from PIL import Image
from tqdm import tqdm

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

from src.logic import LogicEngine
from src.stash import StashEngine, MetadataStation


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_safe_qid(line, idx):
    for key in ["qid", "question_id", "id", "image_id"]:
        if key in line and line[key] is not None:
            try:
                return int(line[key])
            except Exception:
                return idx
    return idx


def get_safe_label(line):
    return (
        line.get("label", None)
        or line.get("answer", None)
        or line.get("gt-label", None)
        or line.get("gt_label", None)
    )


def resolve_image_path(path_image_dir, image_file):
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


def build_prompt_for_caption(model, cfgs, caption):
    """
    Build prompt for one candidate caption.
    We will mask the prefix and compute LM loss only on the caption-related tokens.
    """
    if model.config.mm_use_im_start_end:
        image_prefix = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
        )
    else:
        image_prefix = DEFAULT_IMAGE_TOKEN + "\n"

    user_text = image_prefix + caption

    conv = conv_templates[cfgs.conv_mode].copy()
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    conv_prefix = conv_templates[cfgs.conv_mode].copy()
    conv_prefix.append_message(conv_prefix.roles[0], image_prefix)
    conv_prefix.append_message(conv_prefix.roles[1], None)
    prefix_prompt = conv_prefix.get_prompt()

    return full_prompt, prefix_prompt


@torch.no_grad()
def score_caption_option(
    model,
    tokenizer,
    cfgs,
    device,
    image,
    image_tensor,
    caption,
    score_mode="mean",
):
    full_prompt, prefix_prompt = build_prompt_for_caption(model, cfgs, caption)

    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        conv=None,
        return_tensors="pt",
    ).unsqueeze(0).to(device=device)

    prefix_ids = tokenizer_image_token(
        prefix_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        conv=None,
        return_tensors="pt",
    ).unsqueeze(0).to(device=device)

    prefix_len = min(prefix_ids.shape[1], input_ids.shape[1])

    labels = input_ids.clone()
    labels[:, :prefix_len] = -100
    labels[labels == IMAGE_TOKEN_INDEX] = -100

    target_token_count = int((labels != -100).sum().item())

    if target_token_count == 0:
        return {
            "mean_logprob": float("-inf"),
            "sum_logprob": float("-inf"),
            "target_token_count": 0,
            "loss": float("inf"),
        }

    outputs = model(
        input_ids=input_ids,
        images=image_tensor.unsqueeze(0).half().to(device),
        image_sizes=[image.size],
        labels=labels,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    loss = float(outputs.loss.item())

    mean_logprob = -loss
    sum_logprob = -loss * target_token_count

    return {
        "mean_logprob": mean_logprob,
        "sum_logprob": sum_logprob,
        "target_token_count": target_token_count,
        "loss": loss,
    }


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

    MetadataStation.activate()
    MetadataStation.export_model_config(model.config)

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

    score_mode = getattr(cfgs, "score_mode", "mean")

    answer_file_ver = (
        f"[{cfgs.name_daset}-{cfgs.name_category}]"
        f"{cfgs.name_exp}-{str(int(time.time()))}"
    )

    answers_file = osp.join("E_answers", cfgs.name_model, f"{answer_file_ver}.jsonl")
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(osp.dirname(answers_file), exist_ok=True)

    correct = 0
    total = 0

    with open(answers_file, "w") as ans_file:
        setattr(model, "tokenizer", tokenizer)

        for idx, line in enumerate(tqdm(questions)):
            try:
                qid = get_safe_qid(line, idx)
                gt_label = get_safe_label(line)

                image_file = line.get("image", None) or line.get("image_path", None)
                if image_file is None:
                    raise ValueError(f"Cannot find image in line: {line}")

                image_path, image_file_for_output = resolve_image_path(
                    cfgs.path_image_dir,
                    image_file,
                )

                caption_options = line.get("caption_options", None)
                if caption_options is None:
                    raise ValueError(
                        "caption_options not found. "
                        "Please rebuild questions.jsonl with caption_options."
                    )

                image = Image.open(image_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0]

                option_records = []
                scores = []

                for option_idx, caption in enumerate(caption_options):
                    rec = score_caption_option(
                        model=model,
                        tokenizer=tokenizer,
                        cfgs=cfgs,
                        device=device,
                        image=image,
                        image_tensor=image_tensor,
                        caption=caption,
                        score_mode=score_mode,
                    )

                    score = (
                        rec["mean_logprob"]
                        if score_mode == "mean"
                        else rec["sum_logprob"]
                    )

                    option_records.append(
                        {
                            "option_idx": option_idx,
                            "caption": caption,
                            **rec,
                            "score": score,
                        }
                    )

                    scores.append(score)

                    StashEngine.clear()
                    LogicEngine.clear()

                pred_idx = int(torch.tensor(scores).argmax().item())
                is_correct = pred_idx == 0

                correct += int(is_correct)
                total += 1

                ans_file.write(
                    json.dumps(
                        {
                            "question_id": qid,
                            "label": gt_label,
                            "answer": 0,
                            "pred_idx": pred_idx,
                            "correct": is_correct,
                            "scores": scores,
                            "score_mode": score_mode,
                            "caption_options": caption_options,
                            "option_records": option_records,
                            "image": image_file_for_output,
                            "model_id": name_model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                ans_file.flush()

            except Exception as e:
                print("\n[ERROR] Failed on sample:")
                print(json.dumps(line, ensure_ascii=False, indent=2))
                raise e

    print(f"\nSaved answers to: {answers_file}")
    print(f"Accuracy = {correct}/{total} = {correct / total if total else 0.0:.4f}")


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

    parser.add_argument("--logic", type=str, default=None)
    parser.add_argument("--dim_prospector", action="store_true")
    parser.add_argument("--head_fork", action="store_true")
    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--sink_rule", type=str, default="ours")
    parser.add_argument("--head_rule", type=str, default="ours")

    parser.add_argument("--exp_config", type=str, default=None)

    args = parser.parse_args()

    eval_model(args)
