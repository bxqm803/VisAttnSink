import argparse
import json
import os
import re
from pathlib import Path


RELS = ["left", "right", "on", "under"]


def infer_relation_from_path(image_path: str) -> str:
    s = str(image_path).lower()

    if "left_of" in s:
        return "left"
    if "right_of" in s:
        return "right"
    if "_on_" in s:
        return "on"
    return "under"


def clean_obj(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"^[Tt]he\s+", "", x)
    x = re.sub(r"^[Aa]n?\s+", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip(" .")


def parse_objects_from_caption(caption: str):
    """
    AdaptVis caption_options[0] is the correct caption.

    Expected examples:
      The cat is left of the dog.
      The cat is right of the dog.
      The cat is on the dog.
      The cat is under the dog.
    """
    cap = str(caption).strip().strip(".")

    patterns = [
        r"^(.*?)\s+(?:is|are)\s+(?:to the\s+)?left of\s+(.*?)$",
        r"^(.*?)\s+(?:is|are)\s+(?:to the\s+)?right of\s+(.*?)$",
        r"^(.*?)\s+(?:is|are)\s+on\s+(.*?)$",
        r"^(.*?)\s+(?:is|are)\s+under\s+(.*?)$",
    ]

    for pat in patterns:
        m = re.match(pat, cap, flags=re.IGNORECASE)
        if m:
            subj = clean_obj(m.group(1))
            obj = clean_obj(m.group(2))
            if subj and obj:
                return subj, obj

    return None


def build_question(caption: str) -> str:
    parsed = parse_objects_from_caption(caption)

    if parsed is None:
        return (
            "What is the spatial relationship between the two main objects in the image? "
            "Answer with only one word from: left, right, on, under."
        )

    subj, obj = parsed

    return (
        f"Where is the {subj} relative to the {obj}? "
        "Answer with only one word from: left, right, on, under."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation",
        default="/ddnB/work/mwang32/test/AdaptVis/data/controlled_images_dataset.json",
    )
    parser.add_argument(
        "--image-root",
        default="/ddnB/work/mwang32/test/AdaptVis/data/controlled_images",
    )
    parser.add_argument(
        "--out",
        default="D_datasets/Controlled_Images_A/Questions/questions.jsonl",
    )

    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    image_root = Path(args.image_root).resolve()
    out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for idx, d in enumerate(data):
        image_path = Path(d["image_path"])

        if not image_path.is_absolute():
            # AdaptVis stores image_path relative to its repo root in many cases.
            image_path = Path("/ddnB/work/mwang32/test/AdaptVis") / image_path

        image_path = image_path.resolve()

        try:
            image_rel = str(image_path.relative_to(image_root))
        except Exception:
            # fallback: only filename
            image_rel = image_path.name

        caption_options = d.get("caption_options", [])
        if not caption_options:
            raise ValueError(f"No caption_options at index={idx}: {d}")

        correct_caption = caption_options[0]
        rel = infer_relation_from_path(d["image_path"])

        if rel not in RELS:
            raise ValueError(f"Bad relation at index={idx}, image_path={d['image_path']}")

        question = build_question(correct_caption)

        rows.append(
            {
                "qid": idx,
                "question_id": idx,
                "image": image_rel,
                "question": question,
                "text": question,
                "answer": rel,
                "label": rel,
                "caption": correct_caption,
                "caption_options": caption_options,
                "source_image_path": str(image_path),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
