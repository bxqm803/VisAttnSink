import argparse
import json
import os
import re
from pathlib import Path


RELATIONS = ["left", "right", "on", "under"]


def relation_from_image_path(path: str):
    s = str(path).lower()

    if "left_of" in s:
        return "left"
    if "right_of" in s:
        return "right"
    if "_on_" in s or "/on_" in s or " on " in s:
        return "on"
    if "under" in s:
        return "under"

    return None


def relation_from_caption(caption: str):
    s = str(caption).lower()

    if "left of" in s:
        return "left"
    if "right of" in s:
        return "right"
    if re.search(r"\bon\b", s):
        return "on"
    if "under" in s:
        return "under"

    return None


def clean_obj(x: str):
    x = str(x).strip()
    x = re.sub(r"^[Tt]he\s+", "", x)
    x = re.sub(r"^[Aa]n?\s+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" .")
    return x


def parse_objects_from_caption(caption: str):
    """
    Try to parse captions like:
      The cat is left of the dog.
      The cube is on the sphere.
      A is under B.

    If parsing fails, return None.
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


def build_question(caption: str):
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

    for i, d in enumerate(data):
        image_path = Path(d["image_path"])

        if not image_path.is_absolute():
            image_path = Path("/ddnB/work/mwang32/test/AdaptVis") / image_path

        try:
            image_rel = str(image_path.resolve().relative_to(image_root))
        except Exception:
            image_rel = image_path.name

        caption_options = d.get("caption_options", [])
        correct_caption = caption_options[0] if caption_options else ""

        rel = relation_from_image_path(d["image_path"])
        if rel is None:
            rel = relation_from_caption(correct_caption)

        if rel not in RELATIONS:
            raise ValueError(f"Cannot infer relation for index={i}, image_path={d.get('image_path')}, caption={correct_caption}")

        question = build_question(correct_caption)

        rows.append(
            {
                "qid": i,
                "image": image_rel,
                "question": question,
                "answer": rel,
                "label": rel,
                "caption": correct_caption,
                "source_image_path": str(image_path),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} questions to {out_path}")


if __name__ == "__main__":
    main()
