

import json
import os
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# EDIT THESE TO MATCH YOUR SETUP
# ─────────────────────────────────────────────────────────────
COCO_JSON    = "bbox_coco_dataset.json"   # your annotation file
IMAGE_DIR    = "dataset"                  # folder containing all .jpg images
OUTPUT_DIR   = "wood_dataset"             # where split dataset will be written
# ─────────────────────────────────────────────────────────────

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.20
TEST_RATIO   = 0.10
SEED         = 42


def load_coco(json_path):
    print(f"  Loading {json_path} ...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Build category id → 0-indexed class id map
    # COCO uses 1-based IDs; YOLO needs 0-based
    categories   = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_cls = {cat["id"]: idx for idx, cat in enumerate(categories)}
    class_names   = [cat["name"] for cat in categories]

    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Images      : {len(data['images']):,}")
    print(f"  Annotations : {len(data['annotations']):,}")
    return data, cat_id_to_cls, class_names


def build_annotation_map(data, cat_id_to_cls):
    """Group annotations by image_id."""
    ann_map = defaultdict(list)
    skipped = 0
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            skipped += 1
            continue
        bbox = ann["bbox"]             # [x_min, y_min, w, h]  absolute px
        if len(bbox) != 4:
            skipped += 1
            continue
        cls_id = cat_id_to_cls[ann["category_id"]]
        ann_map[ann["image_id"]].append((cls_id, bbox))
    if skipped:
        print(f"  ⚠  Skipped {skipped} crowd/invalid annotations")
    return ann_map


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO  [x_min, y_min, width, height]  absolute pixels
    YOLO  [cx, cy, w, h]                 normalised 0–1
    """
    x_min, y_min, w, h = bbox
    cx = (x_min + w / 2.0) / img_w
    cy = (y_min + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    # clamp to [0, 1] in case of slight floating-point overflow
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def write_yolo_labels(data, ann_map, image_dir, output_dir):
    """
    Write one YOLO .txt file per image into output_dir/labels_all/.
    Returns list of (image_path, label_path) pairs.
    """
    labels_dir = Path(output_dir) / "labels_all"
    labels_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(image_dir)
    pairs   = []
    missing_images  = 0
    empty_label_imgs = 0

    for img_info in data["images"]:
        img_id   = img_info["id"]
        filename = img_info["file_name"]
        img_w    = img_info["width"]
        img_h    = img_info["height"]

        img_path = img_dir / filename
        if not img_path.exists():
            missing_images += 1
            continue

        anns = ann_map.get(img_id, [])

        # Write label file (empty file = background image, still valid for YOLO)
        lbl_name = Path(filename).stem + ".txt"
        lbl_path = labels_dir / lbl_name

        lines = []
        for cls_id, bbox in anns:
            cx, cy, nw, nh = coco_bbox_to_yolo(bbox, img_w, img_h)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

        if not lines:
            empty_label_imgs += 1

        pairs.append((img_path, lbl_path))

    print(f"  Written {len(pairs):,} label files  →  {labels_dir}")
    if missing_images:
        print(f"  ⚠  {missing_images} images listed in JSON were not found in '{image_dir}'")
    if empty_label_imgs:
        print(f"  ℹ  {empty_label_imgs} images have no annotations (background-only labels)")
    return pairs


def split_and_copy(pairs, output_dir):
    random.seed(SEED)
    random.shuffle(pairs)
    n       = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train : n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        img_out = Path(output_dir) / "images" / split_name
        lbl_out = Path(output_dir) / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_src, lbl_src in split_pairs:
            shutil.copy2(img_src, img_out / img_src.name)
            shutil.copy2(lbl_src, lbl_out / lbl_src.name)

        print(f"  {split_name:5s}: {len(split_pairs):6,} images  →  {img_out}")

    return splits


def write_yaml(output_dir, class_names):
    yaml_path = Path("dataset.yaml")
    config = {
        "path":  str(Path(output_dir).resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n  dataset.yaml  →  {yaml_path.resolve()}")
    return yaml_path


def print_class_distribution(data, cat_id_to_cls, class_names):
    counts = defaultdict(int)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cls_id = cat_id_to_cls[ann["category_id"]]
        counts[cls_id] += 1

    print("\n  Class distribution in full dataset:")
    print(f"  {'ID':>3}  {'Class':<22}  {'Count':>7}  {'%':>6}")
    print("  " + "-" * 45)
    total = sum(counts.values())
    for cls_id, name in enumerate(class_names):
        n   = counts.get(cls_id, 0)
        pct = 100 * n / total if total else 0
        print(f"  {cls_id:>3}  {name:<22}  {n:>7,}  {pct:>5.1f}%")
    print(f"  {'':>3}  {'TOTAL':<22}  {total:>7,}  100.0%")


def main():
    print("=" * 60)
    print("  COCO → YOLO CONVERSION + DATASET SPLIT")
    print("=" * 60)

    if not os.path.isfile(COCO_JSON):
        raise FileNotFoundError(
            f"Annotation file not found: '{COCO_JSON}'\n"
            "Edit COCO_JSON at the top of this script."
        )
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f"Image folder not found: '{IMAGE_DIR}'\n"
            "Edit IMAGE_DIR at the top of this script."
        )

    print("\n[1/5] Loading COCO annotations...")
    data, cat_id_to_cls, class_names = load_coco(COCO_JSON)

    print("\n[2/5] Class distribution...")
    print_class_distribution(data, cat_id_to_cls, class_names)

    print("\n[3/5] Building annotation map & converting to YOLO format...")
    ann_map = build_annotation_map(data, cat_id_to_cls)
    pairs   = write_yolo_labels(data, ann_map, IMAGE_DIR, OUTPUT_DIR)

    if not pairs:
        raise RuntimeError(
            "No image-label pairs produced. "
            "Check that IMAGE_DIR contains the .jpg files listed in the JSON."
        )

    print(f"\n[4/5] Splitting {len(pairs):,} pairs  "
          f"({int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)})...")
    splits = split_and_copy(pairs, OUTPUT_DIR)

    print("\n[5/5] Writing dataset.yaml...")
    yaml_path = write_yaml(OUTPUT_DIR, class_names)

    # Clean up temporary labels_all folder
    shutil.rmtree(Path(OUTPUT_DIR) / "labels_all", ignore_errors=True)

    print("\n" + "=" * 60)
    print("  CONVERSION COMPLETE")
    print(f"  Total images  : {len(pairs):,}")
    print(f"  Train         : {len(splits['train']):,}")
    print(f"  Val           : {len(splits['val']):,}")
    print(f"  Test          : {len(splits['test']):,}")
    print(f"  Classes       : {len(class_names)}  ({', '.join(class_names)})")
    print(f"  YAML          : {yaml_path.resolve()}")
    print("\n  Next step:  python train_yolo.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
