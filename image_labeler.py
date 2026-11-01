from __future__ import annotations

import csv
import random
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "abo-images-small" / "images" / "small"
LABEL_PATH = ROOT / "data" / "abo_image_labels.csv"
LABELS = ["电子产品", "服饰鞋包", "家居用品", "书籍与其他", "玩具游戏", "美妆护理", "其他", "不适合展示"]
SAMPLE_SIZE = 800
BATCH_SIZE = 12

st.set_page_config(page_title="图片整理", page_icon="🏷️", layout="centered")


@st.cache_data
def get_sample() -> list[str]:
    paths = list(IMAGE_ROOT.glob("*/*.jpg")) + list(IMAGE_ROOT.glob("*/*.png"))
    if not paths:
        return []
    rng = random.Random(20261024)
    return [path.relative_to(IMAGE_ROOT).as_posix() for path in rng.sample(paths, min(SAMPLE_SIZE, len(paths)))]


def load_labels() -> dict[str, str]:
    if not LABEL_PATH.exists():
        return {}
    with LABEL_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["path"]: row["label"] for row in csv.DictReader(stream)}


def save_labels(labels: dict[str, str]) -> None:
    LABEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABEL_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows({"path": path, "label": label} for path, label in sorted(labels.items()))


st.title("ABO 图片整理")
st.caption("从全量图片中固定抽取一小批样本，人工选择类别。只保存标签清单，不改动图片文件。")

if not IMAGE_ROOT.exists():
    st.error("没有找到图片目录：abo-images-small/images/small")
    st.stop()

sample = get_sample()
labels = load_labels()
pending = [path for path in sample if path not in labels]
st.metric("抽样图片", f"{len(sample)} 张", delta=f"已整理 {len(set(sample) & labels.keys())} 张")

if not pending:
    st.success("这批抽样图片已经全部整理完。主应用会按类别展示已标注图片。")
    st.stop()

if "image_review_index" not in st.session_state:
    st.session_state.image_review_index = 0
index = min(st.session_state.image_review_index, len(pending) - 1)
batch = pending[index:index + BATCH_SIZE]

st.progress(len(set(sample) & labels.keys()) / max(1, len(sample)))
st.caption(f"当前批次：第 {index + 1}–{index + len(batch)} 张未整理图片，共 {len(pending)} 张待处理")
columns = st.columns(4)
selected: dict[str, str] = {}
for offset, relative_path in enumerate(batch):
    with columns[offset % 4]:
        image_path = IMAGE_ROOT / relative_path
        st.image(str(image_path), use_column_width=True)
        st.caption(relative_path)
        selected[relative_path] = st.selectbox(
            "图片类别",
            ["跳过本张"] + LABELS,
            key=f"label_{relative_path}",
            label_visibility="collapsed",
        )

save_col, next_col = st.columns(2)
with save_col:
    if st.button("保存本批标签", type="primary", use_container_width=True):
        for relative_path, label in selected.items():
            if label != "跳过本张":
                labels[relative_path] = label
        save_labels(labels)
        st.session_state.image_review_index = 0
        st.rerun()
with next_col:
    if st.button("换下一批", use_container_width=True):
        st.session_state.image_review_index = (index + BATCH_SIZE) % len(pending)
        st.rerun()

st.caption("建议只整理 20–50 张清晰、主体明确的图片用于展示；选“不适合展示”的图片会保留在标签表中，但不会出现在主应用。")
