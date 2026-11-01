"""从 ABO 图片中抽样、生成总览，并把人工挑选的图片整理成展示素材。

用法：
  python prepare_showcase_images.py sample --count 600
  python prepare_showcase_images.py label --category 电子产品 --ids 3,8,21
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "abo-images-small" / "images" / "small"
DEFAULT_OUTPUT = ROOT / "showcase"
LISTINGS_ROOT = ROOT / "abo-listings" / "listings" / "metadata"
IMAGE_METADATA = ROOT / "abo-images-small" / "images" / "metadata" / "images.csv.gz"
SHEET_COLUMNS = 6
SHEET_ROWS = 5
TILE_WIDTH = 180
TILE_HEIGHT = 160


def image_paths() -> list[Path]:
    return sorted(
        path for path in IMAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def quality_score(path: Path) -> tuple[float, int, int] | None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            if min(width, height) < 48 or max(width, height) < 96:
                return None
            ratio = max(width, height) / max(1, min(width, height))
            if ratio > 3.5:
                return None
            preview = image.convert("RGB")
            preview.thumbnail((128, 128))
            stat = ImageStat.Stat(preview)
            variance = sum(stat.var) / 3
            # 只排除几乎纯色的损坏图，商品白底图仍然保留给人工选择。
            if variance < 18:
                return None
            score = min(1.0, variance / 220) * min(1.0, min(width, height) / 160)
            return score, width, height
    except (OSError, ValueError):
        return None


def write_contact_sheets(records: list[dict[str, object]], output: Path) -> None:
    sheets = output / "总览"
    sheets.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    for sheet_number, start in enumerate(range(0, len(records), per_sheet), start=1):
        batch = records[start:start + per_sheet]
        sheet = Image.new("RGB", (SHEET_COLUMNS * TILE_WIDTH, SHEET_ROWS * TILE_HEIGHT), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(batch):
            try:
                with Image.open(record["absolute_path"]) as source:
                    image = source.convert("RGB")
                    image.thumbnail((TILE_WIDTH - 12, TILE_HEIGHT - 35))
                    x = (offset % SHEET_COLUMNS) * TILE_WIDTH + (TILE_WIDTH - image.width) // 2
                    y = (offset // SHEET_COLUMNS) * TILE_HEIGHT + 5
                    sheet.paste(image, (x, y))
            except OSError:
                continue
            tile_x = (offset % SHEET_COLUMNS) * TILE_WIDTH + 6
            tile_y = (offset // SHEET_COLUMNS) * TILE_HEIGHT + TILE_HEIGHT - 27
            # 总览编号使用 ASCII，避免 Windows 默认字体缺少中文而显示方框。
            draw.text((tile_x, tile_y), f"ID {record['id']}", fill="black", font=font)
        sheet.save(sheets / f"总览_{sheet_number:03d}.jpg", quality=92)


def sample_command(args: argparse.Namespace) -> None:
    if not IMAGE_ROOT.exists():
        raise SystemExit(f"没有找到图片目录：{IMAGE_ROOT}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = image_paths()
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    records: list[dict[str, object]] = []
    for path in paths:
        result = quality_score(path)
        if result is None:
            continue
        score, width, height = result
        records.append({
            "id": len(records) + 1,
            "relative_path": path.relative_to(IMAGE_ROOT).as_posix(),
            "absolute_path": path,
            "width": width,
            "height": height,
            "quality_score": round(score, 4),
        })
        if len(records) >= args.count:
            break

    manifest_path = output / "candidates.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "relative_path", "width", "height", "quality_score"])
        writer.writeheader()
        writer.writerows({key: value for key, value in record.items() if key != "absolute_path"} for record in records)
    write_contact_sheets(records, output)
    print(f"已生成 {len(records)} 张候选图片")
    print(f"编号清单：{manifest_path}")
    print(f"图片总览：{output / '总览'}")
    print("看总览后，用 label 命令按编号整理展示素材。")


def label_command(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    manifest_path = output / "candidates.csv"
    if not manifest_path.exists():
        raise SystemExit("请先运行 sample 命令生成 showcase/candidates.csv")
    requested = {int(value.strip()) for value in args.ids.split(",") if value.strip()}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        records = {int(row["id"]): row for row in csv.DictReader(stream)}
    missing = requested - records.keys()
    if missing:
        raise SystemExit(f"找不到这些编号：{sorted(missing)}")

    category_dir = output / args.category
    category_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output / "labels.csv"
    old_rows: list[dict[str, str]] = []
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
            old_rows = list(csv.DictReader(stream))
    known = {(row["candidate_id"], row["category"]) for row in old_rows}
    for candidate_id in sorted(requested):
        row = records[candidate_id]
        key = (str(candidate_id), args.category)
        if key in known:
            continue
        source = IMAGE_ROOT / row["relative_path"]
        target = category_dir / f"abo_{candidate_id:04d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        old_rows.append({
            "candidate_id": str(candidate_id),
            "category": args.category,
            "source_path": row["relative_path"],
            "showcase_path": target.relative_to(output).as_posix(),
        })
    with labels_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate_id", "category", "source_path", "showcase_path"])
        writer.writeheader()
        writer.writerows(old_rows)
    print(f"已整理 {len(requested)} 张图片到：{category_dir}")
    print(f"标签清单：{labels_path}")


AUTO_RULES = {
    "手机配件": {
        "types": {"CELLULAR_PHONE_CASE"},
    },
    "耳机音频": {
        "types": {"HEADPHONES", "SPEAKERS"},
    },
    "电脑及数码": {
        "types": {"PERSONAL_COMPUTER", "COMPUTER_INPUT_DEVICE", "COMPUTER_ADD_ON", "NOTEBOOK_COMPUTER", "MONITOR", "TABLET", "CAMERA", "VIDEO_GAME_ACCESSORIES", "WATCH"},
    },
    "鞋包服饰": {
        "types": {"SHOES", "BOOT", "SANDAL", "HANDBAG", "BACKPACK", "HAT", "APPAREL", "DRESS", "SHIRT", "PANTS", "JEANS", "JACKET", "COAT", "SWEATER"},
    },
    "运动用品": {
        "types": {"EXERCISE_BAND", "EXERCISE_MAT", "FITNESS", "SPORTS_EQUIPMENT"},
    },
    "学习文具": {
        "types": {"WRITING_INSTRUMENT", "INPUT_PEN", "MARKING_PEN", "STORAGE_BINDER", "PAPER_PRODUCT", "BOOK"},
    },
}


def text_values(value: object) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.extend(str(v) for v in item.values() if isinstance(v, (str, int, float)))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(value or "")


def load_image_metadata() -> dict[str, str]:
    with gzip.open(IMAGE_METADATA, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return {row["image_id"]: row["path"] for row in reader}


def auto_command(args: argparse.Namespace) -> None:
    if not LISTINGS_ROOT.exists():
        raise SystemExit("没有找到 ABO 商品元数据。请将 abo-listings.tar 解压到项目根目录，目录名为 abo-listings。")
    image_paths_by_id = load_image_metadata()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    category_records: dict[str, list[dict[str, object]]] = {category: [] for category in AUTO_RULES}
    seen_images: set[str] = set()
    listing_files = sorted(LISTINGS_ROOT.glob("listings_*.json.gz"))
    for listing_file in listing_files:
        with gzip.open(listing_file, "rt", encoding="utf-8") as stream:
            for line in stream:
                product = json.loads(line)
                product_type = text_values(product.get("product_type")).upper()
                image_id = str(product.get("main_image_id") or "")
                relative_path = image_paths_by_id.get(image_id)
                if not relative_path or image_id in seen_images:
                    continue
                source = IMAGE_ROOT / relative_path
                if not source.is_file() or quality_score(source) is None:
                    continue
                matched_category = None
                for category, rule in AUTO_RULES.items():
                    type_match = any(product_type == item or product_type.startswith(item + "_") for item in rule["types"])
                    if type_match:
                        matched_category = category
                        break
                if not matched_category or len(category_records[matched_category]) >= args.per_category:
                    continue
                seen_images.add(image_id)
                category_records[matched_category].append({
                    "source_path": relative_path,
                    "image_id": image_id,
                    "item_id": product.get("item_id", ""),
                    "product_type": product_type,
                    "item_name": text_values(product.get("item_name"))[:240],
                })
                if all(len(rows) >= args.per_category for rows in category_records.values()):
                    break
        if all(len(rows) >= args.per_category for rows in category_records.values()):
            break

    labels_path = output / "metadata_labels.csv"
    with labels_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = ["category", "image_id", "source_path", "item_id", "product_type", "item_name", "showcase_path"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for category, records in category_records.items():
            target_dir = output / category
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, record in enumerate(records, start=1):
                source = IMAGE_ROOT / str(record["source_path"])
                target = target_dir / f"abo_{index:03d}{source.suffix.lower()}"
                shutil.copy2(source, target)
                writer.writerow({"category": category, **record, "showcase_path": target.relative_to(output).as_posix()})
    for category, records in category_records.items():
        print(f"{category}: {len(records)} 张")
    print(f"已按 ABO 官方商品元数据整理：{labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 ABO 商品图片展示素材")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample", help="抽样并生成图片总览")
    sample.add_argument("--count", type=int, default=600, help="候选图片数量，默认 600")
    sample.add_argument("--seed", type=int, default=20261024, help="固定随机种子")
    sample.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出目录")
    sample.set_defaults(function=sample_command)
    label = subparsers.add_parser("label", help="按编号复制图片并写入分类标签")
    label.add_argument("--category", required=True, help="分类名称，例如 电子产品")
    label.add_argument("--ids", required=True, help="编号列表，例如 3,8,21")
    label.add_argument("--output", default=str(DEFAULT_OUTPUT), help="候选清单所在目录")
    label.set_defaults(function=label_command)
    auto = subparsers.add_parser("auto", help="按 ABO 官方商品元数据自动筛选图片")
    auto.add_argument("--per-category", type=int, default=30, help="每类最多整理多少张，默认 30")
    auto.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出目录")
    auto.set_defaults(function=auto_command)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
