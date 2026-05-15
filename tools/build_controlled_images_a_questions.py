import argparse
import json
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
        "--adaptvis-root",
        default="/ddnB/work/mwang32/test/AdaptVis",
    )

    parser.add_argument(
        "--out",
        default="D_datasets/Controlled_Images_A/Questions/questions.jsonl",
    )

    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    image_root = Path(args.image_root).resolve()
    adaptvis_root = Path(args.adaptvis_root).resolve()
    out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for idx, d in enumerate(data):
        image_path_raw = d["image_path"]
        image_path = Path(image_path_raw)

        if not image_path.is_absolute():
            image_path = adaptvis_root / image_path

        image_path = image_path.resolve()

        try:
            image_rel = str(image_path.relative_to(image_root))
        except Exception:
            image_rel = image_path.name

        caption_options = d.get("caption_options", [])

        if not caption_options:
            raise ValueError(
                f"No caption_options at idx={idx}, image_path={image_path_raw}"
            )

        correct_caption = caption_options[0]

        rel = relation_from_image_path(image_path_raw)

        if rel is None:
            rel = relation_from_caption(correct_caption)

        if rel not in RELATIONS:
            raise ValueError(
                f"Cannot infer relation for idx={idx}, "
                f"image_path={image_path_raw}, caption={correct_caption}"
            )

        rows.append(
            {
                "qid": idx,
                "question_id": idx,
                "image": image_rel,
                "caption_options": caption_options,
                "answer": 0,
                "label": rel,
                "caption": correct_caption,
                "source_image_path": str(image_path),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
