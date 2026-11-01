from __future__ import annotations

import json
import csv
import re
import base64
import hashlib
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.parquet"
CONVERTED_DATA_PATH = ROOT / "data" / "data_cny.parquet"
MODEL_PATH = ROOT / "models" / "price_model.joblib"
METRICS_PATH = ROOT / "metrics.json"
COMPARISON_PATH = ROOT / "model_comparison.json"
TRANSLATION_CACHE_PATH = ROOT / "data" / "translation_cache.json"
DB_PATH = ROOT / "data" / "campus_market.db"
UPLOAD_DIR = ROOT / "data" / "user_uploads"
CAMPUS_TOP_CATEGORIES = {"Electronics", "Women", "Men", "Sports & Outdoors", "Other"}
SHOWCASE_CATEGORIES = {"手机配件", "耳机音频", "电脑及数码", "鞋包服饰", "运动用品", "学习文具"}


def database_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seller_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                condition_label TEXT NOT NULL,
                delivery TEXT NOT NULL,
                image_path TEXT,
                status TEXT NOT NULL DEFAULT '在售',
                created_at TEXT NOT NULL,
                FOREIGN KEY (seller_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL UNIQUE,
                buyer_id INTEGER NOT NULL,
                seller_id INTEGER,
                listing_id INTEGER,
                source_item_index INTEGER,
                title TEXT NOT NULL,
                price REAL NOT NULL,
                delivery TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT '待交付',
                created_at TEXT NOT NULL,
                FOREIGN KEY (buyer_id) REFERENCES users(id),
                FOREIGN KEY (seller_id) REFERENCES users(id),
                FOREIGN KEY (listing_id) REFERENCES listings(id)
            );
            """
        )
        demo_password = hashlib.sha256("123456".encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT OR IGNORE INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
            ("student", demo_password, datetime.now().isoformat(timespec="seconds")),
        )


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def authenticate_user(username: str, password: str) -> dict[str, object] | None:
    username = username.strip()
    # 页面使用中文演示账号提示，同时兼容数据库中原有的英文账号记录。
    lookup_name = "student" if username in {"演示账号", "学生账号"} else username
    with database_connection() as connection:
        row = connection.execute(
            "SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
            (lookup_name, hash_password(password)),
        ).fetchone()
    return dict(row) if row else None


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 2 or len(password) < 6:
        return False, "账号至少 2 个字符，密码至少 6 个字符。"
    try:
        with database_connection() as connection:
            connection.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hash_password(password), datetime.now().isoformat(timespec="seconds")),
            )
    except sqlite3.IntegrityError:
        return False, "这个账号已经存在，请直接登录。"
    return True, "注册成功，请登录。"


def current_user() -> dict[str, object] | None:
    value = st.session_state.get("current_user")
    return value if isinstance(value, dict) else None


def render_account_sidebar() -> dict[str, object] | None:
    user = current_user()
    with st.sidebar:
        st.divider()
        st.header("账号中心")
        if user:
            visible_username = "演示账号" if user["username"] == "student" else sanitize_chinese_display(str(user["username"]), "校园用户")
            st.success(f"已登录：{visible_username}")
            if st.button("退出登录", key="logout_button", use_container_width=True):
                st.session_state.pop("current_user", None)
                st.rerun()
            return user
        st.caption("登录后可以发布商品、购买商品并查看交易记录。")
        with st.form("login_form"):
            login_name = st.text_input("账号", key="login_name")
            login_password = st.text_input("密码", type="password", key="login_password")
            login_submitted = st.form_submit_button("登录", use_container_width=True)
        if login_submitted:
            authenticated = authenticate_user(login_name, login_password)
            if authenticated:
                st.session_state["current_user"] = authenticated
                st.rerun()
            st.error("账号或密码不正确。")
        with st.expander("注册新账号"):
            with st.form("register_form"):
                register_name = st.text_input("新账号", key="register_name")
                register_password = st.text_input("新密码", type="password", key="register_password")
                register_submitted = st.form_submit_button("注册", use_container_width=True)
            if register_submitted:
                success, message = register_user(register_name, register_password)
                (st.success if success else st.warning)(message)
        st.caption("课堂演示账号：演示账号 / 123456")
    return None

CN_SEARCH_ALIASES = {
    "耳机": "headphones earbuds headset",
    "头戴式耳机": "headphones",
    "蓝牙耳机": "wireless headphones earbuds",
    "手机": "cell phone smartphone mobile",
    "平板": "tablet ipad",
    "电脑": "computer laptop",
    "笔记本": "laptop notebook",
    "键盘": "keyboard",
    "鼠标": "mouse",
    "相机": "camera",
    "游戏机": "video game console",
    "计算器": "calculator",
    "台灯": "desk lamp lamp",
    "风扇": "fan",
    "背包": "backpack bag",
    "书": "book textbook",
    "教材": "book textbook study",
    "衣服": "clothing clothes",
    "鞋": "shoes sneakers",
    "二手": "used pre-owned",
    "全新": "new brand new",
    "九成新": "good condition",
    "成色好": "good condition excellent",
    "充电宝": "power bank portable charger",
    "显示器": "monitor computer screen",
    "平板电脑": "tablet ipad",
    "羽绒服": "coat jacket puffer",
    "教材": "textbook book",
    "运动鞋": "sneakers athletic shoes",
    "苹果": "apple iphone",
    "华为": "huawei",
    "小米": "xiaomi",
    "索尼": "sony",
}

SEARCH_CATEGORY_HINTS = {
    "耳机": "Electronics", "蓝牙耳机": "Electronics", "手机": "Electronics",
    "苹果": "Electronics", "华为": "Electronics", "小米": "Electronics",
    "索尼": "Electronics", "平板": "Electronics", "电脑": "Electronics",
    "笔记本": "Electronics", "显示器": "Electronics", "键盘": "Electronics",
    "鼠标": "Electronics", "相机": "Electronics", "游戏机": "Electronics",
    "充电宝": "Electronics", "衣服": "Women", "羽绒服": "Women",
    "鞋": "Women", "运动鞋": "Women", "书": "Other", "教材": "Other",
    "台灯": "Home", "风扇": "Home", "背包": "Women",
}

CATEGORY_LABELS = {
    "Women": "女装",
    "Beauty": "美妆护肤",
    "Kids": "儿童用品",
    "Electronics": "电子产品",
    "Men": "男装",
    "Home": "家居用品",
    "Vintage & Collectibles": "复古与收藏",
    "Other": "其他",
    "Handmade": "手工制品",
    "Sports & Outdoors": "运动与户外",
    "Unknown": "未分类",
}

CATEGORY_WORD_LABELS = {
    "Cell Phones & Accessories": "手机及配件",
    "Video Games & Consoles": "电子游戏与游戏机",
    "TV, Audio & Surveillance": "电视、音频与监控",
    "Computers & Tablets": "电脑与平板",
    "Cameras & Photography": "相机与摄影",
    "Tops & Blouses": "上装与衬衫",
    "Athletic Apparel": "运动服饰",
    "Women's Accessories": "女式配饰",
    "Women's Handbags": "女式手提包",
    "Men's Accessories": "男式配饰",
    "Home D��cor": "家居装饰",
    "Seasonal D��cor": "季节装饰",
    "Home Decor": "家居装饰",
    "Books": "书籍",
    "Book": "书籍",
    "Other": "其他",
    "Unknown": "未分类",
}

CATEGORY_TOKEN_LABELS = {
    "Pants": "裤装", "Tights": "连裤袜", "Leggings": "打底裤", "Shorts": "短裤",
    "T-Shirts": "T恤", "T-shirts": "T恤", "Shirts": "衬衫", "Blouse": "女式衬衫",
    "Dresses": "连衣裙", "Skirts": "半身裙", "Jackets": "夹克", "Coats": "外套",
    "Sweaters": "毛衣", "Cardigan": "开衫", "Hooded": "连帽款", "Jeans": "牛仔裤",
    "Shoes": "鞋类", "Boots": "靴子", "Sandals": "凉鞋", "Sneakers": "运动鞋",
    "Necklaces": "项链", "Bracelets": "手链", "Earrings": "耳饰", "Rings": "戒指",
    "Wallets": "钱包", "Sunglasses": "太阳镜", "Hats": "帽子", "Backpack": "背包",
    "Shoulder Bag": "单肩包", "Handbag": "手提包", "Totes": "托特包", "Messenger": "斜挎包",
    "Makeup": "彩妆", "Face": "面部", "Lips": "唇部", "Eyes": "眼部", "Nails": "美甲",
    "Skin Care": "护肤", "Hair Care": "护发", "Fragrance": "香水", "Bath": "沐浴",
    "Games": "游戏", "Dolls": "玩偶", "Toys": "玩具", "Books": "书籍", "Book": "书籍",
    "Headphones": "耳机", "Cell Phone": "手机", "Cases": "保护壳", "Covers": "外壳",
    "Computer": "电脑", "Laptop": "笔记本电脑", "Camera": "相机", "Watches": "手表",
    "Exercise": "运动健身", "Fitness": "健身", "Sports": "运动", "Outdoors": "户外",
    "Palettes": "彩妆盘", "Bras": "内衣", "Organization": "收纳", "Storage": "收纳",
    "Accessories": "配件", "Other": "其他", "Unknown": "未分类",
}

st.set_page_config(
    page_title="校园淘淘｜二手商品智能助手",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    source_path = CONVERTED_DATA_PATH if CONVERTED_DATA_PATH.exists() else DATA_PATH
    frame = pd.read_parquet(source_path)
    for column in ["name", "item_description", "cat1", "cat2", "cat3", "brand_name"]:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), "Unknown").astype(str)
    if "price_cny" not in frame.columns:
        frame["price_cny"] = np.expm1(frame["log_price"]).clip(lower=0) * 6.6702
    frame["search_text"] = frame["name"] + " " + frame["item_description"]
    # 预先计算是否能匹配到已整理的具体类型示例图，首页只展示有图片的记录。
    curated_images = load_curated_images()
    frame["showcase_category"] = frame.apply(
        lambda item: browse_image_label(item)
        if curated_examples_for_item(item, curated_images)
        else None,
        axis=1,
    )
    return frame


@st.cache_resource
def load_assets():
    # 修改图片匹配规则后递增缓存版本，避免旧的 showcase_category 留在内存中。
    image_match_cache_version = "v3"
    bundle = load(MODEL_PATH)
    data = load_data()
    vectorizer = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(data["search_text"])
    title_vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    title_matrix = title_vectorizer.fit_transform(data["name"])
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    # 兼容人民币改版前生成的旧指标文件，避免页面因缓存或旧进程崩溃。
    rate = float(metrics.get("usd_to_cny", 6.6702))
    if "mae_cny" not in metrics and "mae_usd" in metrics:
        metrics["mae_cny"] = float(metrics["mae_usd"]) * rate
    metrics.setdefault("usd_to_cny", rate)
    metrics.setdefault("exchange_rate_date", "2026-10-23")
    return bundle, data, vectorizer, matrix, title_vectorizer, title_matrix, metrics


def category_options(data: pd.DataFrame, top_category: str) -> tuple[list[str], list[str]]:
    subset = data[data["cat1"] == top_category]
    second = sorted(subset["cat2"].dropna().unique().tolist())
    third = sorted(subset["cat3"].dropna().unique().tolist())
    return second, third


def category_label(value: str) -> str:
    value = str(value)
    if value in CATEGORY_LABELS:
        return CATEGORY_LABELS[value]
    parts = value.split("/")
    translated = [CATEGORY_WORD_LABELS.get(part) for part in parts]
    if all(translated):
        return " / ".join(translated)
    token_pattern = re.compile(r"[A-Za-z&']+")
    tokens = token_pattern.findall(value)
    if tokens:
        result = token_pattern.sub("\u0000", value)
        for token in tokens:
            replacement = CATEGORY_TOKEN_LABELS.get(token, token)
            result = result.replace("\u0000", replacement, 1)
        if not token_pattern.search(result):
            return result.replace("&", "与")
    return "其他商品类别"


def category_display_map(values: list[str]) -> dict[str, str]:
    """给下拉选项生成不含英文的显示名称，同时保留英文原值供模型使用。"""
    result: dict[str, str] = {}
    fallback_index = 1
    for value in values:
        label = category_label(value)
        if label == "其他商品类别":
            label = f"其他商品类别（选项 {fallback_index}）"
            fallback_index += 1
        result[value] = label
    return result


def normalize_chinese_terms(text: str) -> str:
    normalized = text
    for chinese, english in sorted(CN_SEARCH_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(chinese, f" {english} ")
    return normalized


def infer_search_category(query: str) -> str | None:
    for term in sorted(SEARCH_CATEGORY_HINTS, key=len, reverse=True):
        if term in query:
            return SEARCH_CATEGORY_HINTS[term]
    return None


def find_similar(
    query: str,
    data: pd.DataFrame,
    vectorizer,
    matrix,
    title_vectorizer,
    title_matrix,
    top_n: int = 5,
    category_filter: str | None = None,
    sort_mode: str = "相似度最高",
    price_min: float = 0.0,
    price_max: float | None = None,
    condition_filter: int | None = None,
    only_with_images: bool = False,
) -> pd.DataFrame:
    normalized_query = normalize_chinese_terms(query).strip()
    if not normalized_query:
        return data.iloc[0:0].copy()
    query_vector = vectorizer.transform([normalized_query])
    title_query_vector = title_vectorizer.transform([normalized_query])
    full_scores = linear_kernel(query_vector, matrix).ravel()
    title_scores = linear_kernel(title_query_vector, title_matrix).ravel()
    # 商品标题比描述更能代表商品本身；描述只用于补充排序信息。
    scores = 0.75 * title_scores + 0.25 * full_scores
    category_hint = category_filter or infer_search_category(query)
    candidates = np.arange(len(data))
    if category_hint == "校园常见":
        matching_category = data["cat1"].isin(CAMPUS_TOP_CATEGORIES).to_numpy()
        candidates = candidates[matching_category]
    elif category_hint:
        matching_category = data["cat1"].to_numpy() == category_hint
        if matching_category.any():
            candidates = candidates[matching_category]
    price_values = data["price_cny"].to_numpy()
    candidates = candidates[price_values[candidates] >= price_min]
    if price_max is not None:
        candidates = candidates[price_values[candidates] <= price_max]
    if condition_filter is not None:
        condition_values = pd.to_numeric(data["item_condition_id"], errors="coerce").fillna(-1).to_numpy()
        candidates = candidates[condition_values[candidates] == condition_filter]
    if only_with_images:
        has_image = data["showcase_category"].notna().to_numpy()
        candidates = candidates[has_image[candidates]]
    if len(candidates) == 0:
        return data.iloc[0:0].copy()
    ordered = candidates[np.argsort(scores[candidates])[::-1]]
    ordered = ordered[scores[ordered] >= 0.06][: max(top_n, 12)]
    result = data.iloc[ordered].copy()
    result.insert(0, "similarity", scores[ordered])
    if sort_mode == "价格从低到高":
        result = result.sort_values(["price_cny", "similarity"], ascending=[True, False])
    elif sort_mode == "价格从高到低":
        result = result.sort_values(["price_cny", "similarity"], ascending=[False, False])
    else:
        result = result.sort_values("similarity", ascending=False)
    return result.head(top_n)


def recommend_for_item(
    reference_index: int,
    data: pd.DataFrame,
    matrix,
    title_matrix,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rank other listings against one listing, within its product category."""
    position = data.index.get_loc(reference_index)
    reference = data.iloc[position]
    same_detail_category = (
        data["cat2"].eq(reference["cat2"]) & data["cat3"].eq(reference["cat3"])
    ).to_numpy()
    same_subcategory = data["cat2"].eq(reference["cat2"]).to_numpy()
    if reference["cat3"] != "Unknown" and same_detail_category.sum() > top_n:
        candidates = np.flatnonzero(same_detail_category)
    elif reference["cat2"] != "Unknown" and same_subcategory.sum() > top_n:
        candidates = np.flatnonzero(same_subcategory)
    else:
        candidates = np.flatnonzero(data["cat1"].eq(reference["cat1"]).to_numpy())
    candidates = candidates[candidates != position]
    if len(candidates) == 0:
        return data.iloc[0:0].copy()
    title_scores = linear_kernel(title_matrix[position], title_matrix[candidates]).ravel()
    full_scores = linear_kernel(matrix[position], matrix[candidates]).ravel()
    scores = 0.75 * title_scores + 0.25 * full_scores
    ranked = np.argsort(scores)[::-1]
    ranked = ranked[scores[ranked] >= 0.06][:top_n]
    result = data.iloc[candidates[ranked]].copy()
    result.insert(0, "similarity", scores[ranked])
    return result


def input_completeness(name: str, description: str, brand: str, top_category: str, second_category: str, third_category: str) -> tuple[int, list[str]]:
    score = 0
    missing: list[str] = []
    if len(name.strip()) >= 4:
        score += 25
    else:
        missing.append("商品名称")
    if len(description.strip()) >= 20:
        score += 25
    else:
        missing.append("商品描述")
    if brand.strip() and brand.strip().lower() not in {"unknown", "未填写"}:
        score += 15
    else:
        missing.append("品牌")
    if top_category != "Unknown":
        score += 20
    else:
        missing.append("商品大类")
    if second_category != "Unknown":
        score += 10
    else:
        missing.append("商品小类")
    if third_category != "Unknown":
        score += 5
    return score, missing


def price_interval(predicted_price: float, metrics: dict[str, object]) -> tuple[float, float, str]:
    if predicted_price < 50:
        band_key = "0-50"
    elif predicted_price < 100:
        band_key = "50-100"
    elif predicted_price < 300:
        band_key = "100-300"
    else:
        band_key = "300+"
    intervals = metrics.get("residual_intervals", {})
    band_intervals = intervals.get("by_predicted_price_band", {}).get(band_key, {}) if isinstance(intervals, dict) else {}
    if not band_intervals and isinstance(intervals, dict):
        band_intervals = intervals.get("by_actual_price_band", {}).get(band_key, {})
    p10 = float(band_intervals.get("residual_p10_cny", intervals.get("p10_cny", -metrics["mae_cny"])))
    p90 = float(band_intervals.get("residual_p90_cny", intervals.get("p90_cny", metrics["mae_cny"])))
    low = max(0.0, predicted_price + min(p10, 0.0))
    high = max(low, predicted_price + max(p90, 10.0))
    return low, high, band_key


@st.cache_data(show_spinner=False)
def baseline_evaluation(data: pd.DataFrame) -> dict[str, float]:
    """用同一测试划分计算简单价格基线，帮助课堂解释模型是否学到了规律。"""
    positions = np.arange(len(data))
    train_positions, test_positions = train_test_split(positions, test_size=0.2, random_state=42)
    train = data.iloc[train_positions]
    test = data.iloc[test_positions]
    global_median = float(train["price_cny"].median())
    category_medians = train.groupby("cat1")["price_cny"].median()
    predictions = test["cat1"].map(category_medians).fillna(global_median).to_numpy(dtype=float)
    actual = test["price_cny"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(actual - predictions)))
    ss_res = float(np.sum((actual - predictions) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {"mae_cny": mae, "r2_cny": 1.0 - ss_res / ss_tot if ss_tot else 0.0}


def calculate_reference_estimate(
    model,
    data: pd.DataFrame,
    metrics: dict[str, object],
    name: str,
    description: str,
    top_category: str,
    second_category: str,
    third_category: str,
    brand: str,
    condition: int,
    shipping: bool,
    quoted_price: float,
) -> dict[str, object]:
    """用统一的输入结构计算参考价，供估价页和商品详情页复用。"""
    row = pd.DataFrame(
        [
            {
                "name": normalize_chinese_terms(name),
                "item_description": normalize_chinese_terms(description),
                "cat1": top_category,
                "cat2": second_category,
                "cat3": third_category,
                "brand_name": brand or "Unknown",
                "item_condition_id": condition,
                "shipping": int(shipping),
            }
        ]
    )
    predicted_log = float(model.predict(row)[0])
    predicted_price = max(0.0, float(np.expm1(predicted_log)))
    low, high, band_key = price_interval(predicted_price, metrics)
    completeness, missing_fields = input_completeness(
        name, description, brand, top_category, second_category, third_category
    )
    same_category = data[data["cat1"] == top_category]
    category_median = float(same_category["price_cny"].median()) if not same_category.empty else None
    return {
        "predicted_price": predicted_price,
        "low": low,
        "high": high,
        "band_key": band_key,
        "completeness": completeness,
        "missing_fields": missing_fields,
        "category_median": category_median,
        "quoted_price": quoted_price,
        "condition": condition,
        "shipping": shipping,
        "name": name,
        "description": description,
        "brand": brand,
        "top_category": top_category,
        "second_category": second_category,
        "third_category": third_category,
    }


def campus_fit_assessment(
    name: str,
    description: str,
    top_category: str,
    second_category: str,
    third_category: str,
) -> tuple[int, str, list[str]]:
    """基于校园常见类别和使用场景的规则评分，不参与价格预测。"""
    text = f"{name} {description} {top_category} {second_category} {third_category}".lower()
    score = 50
    reasons: list[str] = []
    category_scores = {
        "Electronics": 24,
        "Other": 18,
        "Sports & Outdoors": 16,
        "Women": 10,
        "Men": 10,
    }
    score += category_scores.get(top_category, 4)
    if any(term in text for term in ["耳机", "headphone", "earbud", "手机", "phone", "电脑", "laptop", "tablet", "键盘", "keyboard"]):
        score += 8
        reasons.append("适合学习、通讯或宿舍日常使用")
    if any(term in text for term in ["教材", "书", "book", "textbook", "calculator", "计算器"]):
        score += 8
        reasons.append("与课程学习场景直接相关")
    if any(term in text for term in ["运动", "鞋", "sneaker", "sports", "fitness", "exercise"]):
        score += 5
        reasons.append("适合校园运动场景")
    if any(term in text for term in ["便携", "portable", "small", "轻便", "宿舍"]):
        score += 5
        reasons.append("便于宿舍或校园内携带")
    if top_category in {"Home", "Kids", "Automotive"}:
        score -= 18
        reasons.append("校园普适性相对较低")
    score = max(0, min(100, score))
    if score >= 78:
        label = "很适合校园交易"
    elif score >= 60:
        label = "适合校园交易"
    elif score >= 42:
        label = "校园场景一般"
    else:
        label = "不建议作为校园重点展示"
    if not reasons:
        reasons.append("主要依据商品类别和描述中的校园使用场景判断")
    return score, label, reasons[:3]


def render_price_radar(result: dict[str, object], mode: str) -> None:
    predicted = float(result["predicted_price"])
    low = float(result["low"])
    high = float(result["high"])
    quoted = float(result["quoted_price"])
    upper = max(high, predicted, quoted, 1.0)
    low_position = max(0.0, min(100.0, low / upper * 100))
    high_position = max(low_position, min(100.0, high / upper * 100))
    predicted_position = max(0.0, min(100.0, predicted / upper * 100))
    quoted_position = max(0.0, min(100.0, quoted / upper * 100)) if quoted > 0 else None
    quoted_marker = (
        f'<span class="price-marker current" style="left:{quoted_position:.1f}%">'
        f'<span>当前报价 ¥{quoted:,.0f}</span></span>'
        if quoted_position is not None else ""
    )
    mode_text = "建议挂牌价" if mode == "卖家出售" else "建议接受价"
    st.markdown(
        f'''<div class="price-radar">
          <div class="price-radar-head"><strong>价格雷达</strong><span>{mode_text} ¥{predicted:,.2f}</span></div>
          <div class="price-track">
            <span class="price-range" style="left:{low_position:.1f}%;width:{max(2.0, high_position-low_position):.1f}%"></span>
            <span class="price-marker model" style="left:{predicted_position:.1f}%"><span>模型参考 ¥{predicted:,.0f}</span></span>
            {quoted_marker}
          </div>
          <div class="price-radar-labels"><span>¥{low:,.0f}</span><span>合理区间</span><span>¥{high:,.0f}</span></div>
        </div>''',
        unsafe_allow_html=True,
    )


def render_estimate_insights(
    result: dict[str, object],
    mode: str,
    top_category: str,
    second_category: str,
    third_category: str,
    name: str,
    description: str,
    brand: str,
) -> None:
    predicted_price = float(result["predicted_price"])
    low = float(result["low"])
    high = float(result["high"])
    quoted_price = float(result["quoted_price"])
    st.markdown('<p class="eyebrow">估价结果</p>', unsafe_allow_html=True)
    a, b, c, d = st.columns([1.05, 1.15, 1.05, 1], gap="medium")
    a.metric("建议参考价", f"¥{predicted_price:,.2f}")
    b.metric("合理价格范围", f"¥{low:,.0f} - ¥{high:,.0f}")
    if quoted_price > high:
        c.metric("报价判断", "可能偏高", delta=f"高出约 ¥{quoted_price - predicted_price:.0f}", delta_color="inverse")
        judgment = "当前报价高于模型校准的合理区间。"
    elif quoted_price and quoted_price < low:
        c.metric("报价判断", "可能偏低", delta=f"低于约 ¥{predicted_price - quoted_price:.0f}")
        judgment = "当前报价低于模型校准的合理区间，可进一步核实商品成色和描述。"
    else:
        c.metric("报价判断", "较为合理")
        judgment = "当前报价落在模型校准的合理区间内。"
    d.metric("信息完整度", f"{result['completeness']}%")
    render_price_radar(result, mode)
    fit_score, fit_label, fit_reasons = campus_fit_assessment(
        name, description, top_category, second_category, third_category
    )
    explain_left, explain_right = st.columns([1, 1], gap="large")
    with explain_left:
        st.markdown("**这次估价为什么是这个结果**")
        st.write(judgment)
        category_median = result.get("category_median")
        if category_median is not None:
            median_relation = "高于" if predicted_price > float(category_median) else "低于"
            st.write(
                f"同类商品历史中位价约 ¥{float(category_median):,.0f}，"
                f"模型参考价{median_relation}中位水平。"
            )
        factors = [f"商品类别：{category_label(top_category)}"]
        if brand.strip() and brand.strip().lower() not in {"unknown", "未填写"}:
            factors.append(f"品牌信息：{brand_display_name(brand.strip())}")
        factors.append(f"成色等级：{condition_labels_text(int(result['condition']))}")
        factors.append("描述完整度会影响文本特征的判断")
        st.markdown("、".join(factors) + "。")
    with explain_right:
        st.markdown("**校园适配度**")
        st.progress(fit_score / 100, text=f"{fit_label} · {fit_score} 分")
        st.caption("这是根据类别和校园使用场景的规则评分，不参与价格预测。")
        st.write("；".join(fit_reasons) + "。")
    if result["completeness"] < 70:
        st.warning(f"当前输入信息较少，建议补充：{'、'.join(result['missing_fields'])}。信息完整度不等于模型准确率。")
    else:
        st.success("商品名称、描述和分类信息较完整，估价结果更有参考价值。")
    st.caption(f"价格区间根据验证集的 {result['band_key']} 价格段残差校准，仅供参考，不是交易承诺。")


def condition_labels_text(value: int) -> str:
    return {1: "全新", 2: "较新", 3: "一般", 4: "较旧", 5: "明显使用痕迹"}.get(value, "未提供")


def sanitize_chinese_display(text: str, fallback: str = "商品信息暂未提供") -> str:
    """清理页面展示文本中的残留英文；不改变模型使用的原始字段。"""
    value = str(text or "")
    value = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda match: f"第{match.group(1)}", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(\d+(?:\.\d+)?)\s*ml\b", r"\1毫升", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(\d+(?:\.\d+)?)\s*oz\b", r"\1盎司", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(\d+(?:\.\d+)?)\s*in(?:ch(?:es)?)?\b", r"\1英寸", value, flags=re.IGNORECASE)
    # 清除单词内残留字母，例如型号中的 5s、尺码中的 2x；数字和中文保留。
    value = re.sub(r"[A-Za-z]+", "", value)
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,.;:!?/|-")
    value = re.sub(r"\s+([，。；：！？、])", r"\1", value)
    if not value or not re.search(r"[\u4e00-\u9fff]", value):
        return fallback
    return value


def brand_display_name(value: str) -> str:
    labels = {
        "Sony": "索尼", "Bose": "博士", "Nike": "耐克", "Adidas": "阿迪达斯",
        "Apple": "苹果", "iPhone": "苹果手机", "Samsung": "三星", "Microsoft": "微软",
        "Xbox": "游戏机", "Nintendo": "任天堂", "Michael Kors": "迈克高仕",
        "Ray Ban": "雷朋", "Ray-Ban": "雷朋", "Hanes": "汉斯", "Cole Haan": "蔻尔哈恩",
        "JBL": "杰宝", "L.L. Bean": "豆角牌", "Under Armour": "安德玛",
        "Blue Microphones": "蓝色麦克风", "FOREVER 21": "永远二十一", "Hello Kitty": "凯蒂猫",
        "Brandy Melville": "布兰迪梅尔维尔", "Tory Burch": "汤丽柏琦", "Jeep": "吉普",
        "eos": "伊欧诗", "INC International Concepts": "国际概念",
    }
    text = str(value).strip()
    if text in {"", "Unknown", "None", "nan", "未填写"}:
        return "未提供"
    translated = labels.get(text)
    if translated:
        return sanitize_chinese_display(translated, "其他品牌")
    return sanitize_chinese_display(text, "其他品牌")


def render_purchase_flow(
    item: pd.Series,
    item_index: int,
    user: dict[str, object] | None,
) -> None:
    """课堂演示用的购买确认闭环，不连接真实支付或订单系统。"""
    purchase_key = f"purchase_{item_index}"
    order_key = f"purchase_order_{item_index}"
    listed_price = float(item.get("price_cny", 0.0))
    st.markdown("**交易操作**")
    if user is None:
        st.info("请先在左侧账号中心登录，登录后才能发起购买并保存交易记录。")
        return
    if order_key in st.session_state:
        order = st.session_state[order_key]
        st.success(f"成交成功：订单 {order['order_id']}，成交金额 ¥{order['price']:,.2f}。")
        st.caption("这是课堂演示订单，商品状态已记录在当前页面会话中；实际交易请与卖家线下确认。")
        if st.button("重新发起购买", key=f"purchase_again_{item_index}"):
            st.session_state.pop(order_key, None)
            st.session_state[purchase_key] = False
            st.rerun()
        return

    if st.button("立即购买", key=f"purchase_open_{item_index}", type="primary", use_container_width=True):
        st.session_state[purchase_key] = True
        st.rerun()
    if not st.session_state.get(purchase_key):
        st.caption("点击后确认成交价格，完成课堂演示中的购买流程。")
        return

    with st.form(f"purchase_form_{item_index}"):
        st.markdown("**确认购买信息**")
        st.write(f"商品：{display_product_name(item)}")
        st.write(f"成交金额：¥{listed_price:,.2f}")
        buyer_name = st.text_input("买家称呼", value="校园用户", key=f"buyer_name_{item_index}")
        meetup = st.selectbox(
            "交付方式",
            ["校园当面交付", "校内自提", "与卖家协商"],
            key=f"meetup_{item_index}",
        )
        agreed = st.checkbox("我已确认商品信息和成交金额", key=f"purchase_agreed_{item_index}")
        confirmed = st.form_submit_button("确认成交", type="primary", use_container_width=True)
    if confirmed:
        if not agreed:
            st.warning("请先确认商品信息和成交金额。")
        else:
            order_id = f"校园{int(item_index):05d}-{secrets.token_hex(3).upper()}"
            created_at = datetime.now().isoformat(timespec="seconds")
            with database_connection() as connection:
                connection.execute(
                    """INSERT INTO transactions
                    (order_id, buyer_id, seller_id, listing_id, source_item_index, title, price, delivery, status, created_at)
                    VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, '待交付', ?)""",
                    (
                        order_id,
                        int(user["id"]),
                        int(item_index),
                        display_product_name(item),
                        listed_price,
                        meetup,
                        created_at,
                    ),
                )
            st.session_state[order_key] = {
                "order_id": order_id,
                "price": listed_price,
                "buyer": buyer_name.strip() or "校园用户",
                "meetup": meetup,
            }
            st.session_state.pop(purchase_key, None)
            st.rerun()


def render_publish_tab(user: dict[str, object] | None) -> None:
    st.subheader("发布闲置商品")
    st.caption("填写商品信息并上传图片，商品会保存到本地演示账号的上架记录中。")
    if user is None:
        st.info("请先在左侧账号中心登录，再发布商品。")
        return
    with st.form("publish_listing_form"):
        title = st.text_input("商品名称", placeholder="例如：二手蓝牙耳机")
        description = st.text_area("商品描述", height=130, placeholder="描述使用时间、成色、配件和功能情况")
        category = st.selectbox("商品类别", ["电子产品", "学习用品", "运动用品", "鞋包服饰", "宿舍用品", "其他"])
        price = st.number_input("出售价格（元）", min_value=0.0, value=100.0, step=10.0)
        condition = st.selectbox("商品成色", ["全新", "较新", "一般", "较旧", "明显使用痕迹"])
        delivery = st.selectbox("交付方式", ["校园当面交付", "校内自提", "与买家协商"])
        uploaded_image = st.file_uploader("商品图片（可选）", type=["jpg", "jpeg", "png", "webp"])
        st.caption("支持拖放或点击选择文件，单个文件最大 200MB，支持 JPG、JPEG、PNG、WEBP 格式。")
        submitted = st.form_submit_button("发布商品", type="primary", use_container_width=True)
    if submitted:
        if not title.strip() or not description.strip():
            st.warning("请填写商品名称和商品描述。")
            return
        image_path = None
        if uploaded_image is not None:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            suffix = Path(uploaded_image.name).suffix.lower() or ".jpg"
            image_path = str(UPLOAD_DIR / f"{secrets.token_hex(8)}{suffix}")
            Path(image_path).write_bytes(uploaded_image.getvalue())
        with database_connection() as connection:
            connection.execute(
                """INSERT INTO listings
                (seller_id, title, description, category, price, condition_label, delivery, image_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(user["id"]), title.strip(), description.strip(), category,
                    float(price), condition, delivery, image_path,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        st.success("商品已发布，可以在“账号与记录”中查看上架状态。")


def render_owned_listing_detail(
    listing: pd.Series,
    data: pd.DataFrame,
    model,
    metrics: dict[str, object],
) -> None:
    listing_id = int(listing["id"])
    if st.button("返回我的上架商品", key=f"owned_listing_back_{listing_id}"):
        st.session_state.pop("selected_owned_listing_id", None)
        st.session_state.pop(f"owned_estimate_result_{listing_id}", None)
        st.rerun()

    st.divider()
    st.subheader(sanitize_chinese_display(translate_description(str(listing["title"])), "校园闲置商品"))
    detail_left, detail_right = st.columns([1, 1.5], gap="large")
    image_path = listing.get("image_path")
    with detail_left:
        if isinstance(image_path, str) and Path(image_path).is_file():
            st.image(image_path, use_column_width=True)
        else:
            st.info("这件商品没有上传图片。")
    with detail_right:
        st.metric("挂牌价格", f"¥{float(listing['price']):,.2f}")
        st.write(f"**商品类别：** {listing['category']}")
        st.write(f"**商品成色：** {listing['condition_label']}")
        st.write(f"**交付方式：** {listing['delivery']}")
        st.write(f"**上架状态：** {listing['status']}")
        description = translate_description(str(listing["description"])) if listing["description"] else "商品描述暂未提供"
        st.write(f"**商品描述：** {description}")
        st.caption(f"发布于 {listing['created_at']}")

    st.divider()
    st.markdown("### 智能估价")
    st.caption("已根据这件上架商品的信息预填，可补充或调整后生成模型参考价。")
    category_to_model = {
        "电子产品": "Electronics",
        "学习用品": "Other",
        "运动用品": "Sports & Outdoors",
        "鞋包服饰": "Women",
        "宿舍用品": "Other",
        "其他": "Other",
    }
    top_categories = sorted(data["cat1"].dropna().unique().tolist())
    suggested_top = category_to_model.get(str(listing["category"]), "Other")
    if suggested_top not in top_categories:
        suggested_top = top_categories[0] if top_categories else "Unknown"
    second_options, third_options = category_options(data, suggested_top)
    second_values = ["Unknown"] + second_options
    third_values = ["Unknown"] + third_options
    condition_to_code = {"全新": 1, "较新": 2, "一般": 3, "较旧": 4, "明显使用痕迹": 5}
    default_condition = condition_to_code.get(str(listing["condition_label"]), 3)
    estimate_key = f"owned_estimate_form_{listing_id}"

    with st.form(estimate_key):
        form_left, form_right = st.columns(2, gap="large")
        with form_left:
            estimate_name = st.text_input(
                "商品名称",
                value=sanitize_chinese_display(translate_description(str(listing["title"])), "校园闲置商品"),
                key=f"owned_estimate_name_{listing_id}",
            )
            estimate_top = st.selectbox(
                "商品大类",
                top_categories,
                index=top_categories.index(suggested_top) if suggested_top in top_categories else 0,
                format_func=category_label,
                key=f"owned_estimate_top_{listing_id}",
            )
            estimate_second_options, estimate_third_options = category_options(data, estimate_top)
            estimate_second_values = ["Unknown"] + estimate_second_options
            estimate_third_values = ["Unknown"] + estimate_third_options
            estimate_second = st.selectbox(
                "商品小类", estimate_second_values, format_func=category_label,
                key=f"owned_estimate_second_{listing_id}",
            )
            estimate_third = st.selectbox(
                "商品细分类", estimate_third_values, format_func=category_label,
                key=f"owned_estimate_third_{listing_id}",
            )
            estimate_brand = st.text_input("品牌（可选）", key=f"owned_estimate_brand_{listing_id}")
        with form_right:
            estimate_condition = st.selectbox(
                "商品成色", options=[1, 2, 3, 4, 5], index=default_condition - 1,
                format_func=condition_labels_text, key=f"owned_estimate_condition_{listing_id}",
            )
            estimate_shipping = st.checkbox("卖家承担运费", key=f"owned_estimate_shipping_{listing_id}")
            estimate_description = st.text_area(
                "商品描述",
                value=translate_description(str(listing["description"])) or "商品描述暂未提供",
                height=132,
                key=f"owned_estimate_description_{listing_id}",
            )
            estimate_quoted_price = st.number_input(
                "当前挂牌价格（元）", min_value=0.0, value=float(listing["price"]), step=10.0,
                key=f"owned_estimate_quoted_{listing_id}",
            )
        submitted = st.form_submit_button("生成这件商品的参考价", type="primary", use_container_width=True)

    result_key = f"owned_estimate_result_{listing_id}"
    if submitted:
        st.session_state[result_key] = calculate_reference_estimate(
            model, data, metrics, estimate_name, estimate_description,
            estimate_top, estimate_second, estimate_third, estimate_brand,
            int(estimate_condition), estimate_shipping, float(estimate_quoted_price),
        )
    result = st.session_state.get(result_key)
    if result:
        render_estimate_insights(
            result, "卖家出售", str(result["top_category"]), str(result["second_category"]),
            str(result["third_category"]), str(result["name"]), str(result["description"]),
            str(result["brand"]),
        )


def render_account_tab(
    user: dict[str, object] | None,
    data: pd.DataFrame,
    model,
    metrics: dict[str, object],
) -> None:
    st.subheader("账号与交易记录")
    if user is None:
        st.info("请先在左侧账号中心登录。登录后可以查看上架商品、购买记录和成交状态。")
        return
    with database_connection() as connection:
        listings = pd.read_sql_query(
            "SELECT id, title, description, category, price, condition_label, delivery, status, created_at, image_path FROM listings WHERE seller_id = ? ORDER BY id DESC",
            connection,
            params=(int(user["id"]),),
        )
        purchases = pd.read_sql_query(
            "SELECT order_id, title, price, delivery, status, created_at FROM transactions WHERE buyer_id = ? ORDER BY id DESC",
            connection,
            params=(int(user["id"]),),
        )
        sales = pd.read_sql_query(
            "SELECT order_id, title, price, delivery, status, created_at FROM transactions WHERE seller_id = ? ORDER BY id DESC",
            connection,
            params=(int(user["id"]),),
        )
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("我的上架", f"{len(listings)} 件")
    metric_b.metric("我的购买", f"{len(purchases)} 笔")
    metric_c.metric("我的成交", f"{len(sales)} 笔")
    st.markdown("**我的上架商品**")
    selected_listing_id = st.session_state.get("selected_owned_listing_id")
    selected_listing = listings[listings["id"] == int(selected_listing_id)] if selected_listing_id is not None else listings.iloc[0:0]
    if not selected_listing.empty:
        render_owned_listing_detail(selected_listing.iloc[0], data, model, metrics)
        return
    if listings.empty:
        st.caption("还没有发布商品。可以先到“发布商品”上传一件校园闲置。")
    else:
        for _, listing in listings.iterrows():
            left, right = st.columns([1, 3], gap="medium")
            with left:
                image_path = listing.get("image_path")
                if isinstance(image_path, str) and Path(image_path).is_file():
                    st.image(image_path, use_column_width=True)
                else:
                    st.info("暂无图片")
            with right:
                listing_title = sanitize_chinese_display(translate_description(str(listing["title"])), "校园闲置商品")
                st.markdown(f"**{listing_title}**")
                st.write(f"¥{float(listing['price']):,.2f} · {listing['category']} · {listing['condition_label']}")
                st.caption(f"{listing['status']} · {listing['delivery']} · 发布于 {listing['created_at']}")
                if st.button("查看商品详情并智能估价", key=f"owned_listing_detail_{int(listing['id'])}", use_container_width=True):
                    st.session_state["selected_owned_listing_id"] = int(listing["id"])
                    st.rerun()
    st.divider()
    st.markdown("**我的购买记录**")
    if purchases.empty:
        st.caption("还没有购买记录。打开商品详情并确认成交后，记录会出现在这里。")
    else:
        purchase_view = purchases.rename(columns={
            "order_id": "订单编号", "title": "商品", "price": "成交金额（元）",
            "delivery": "交付方式", "status": "状态", "created_at": "成交时间",
        })
        purchase_view["商品"] = purchase_view["商品"].map(
            lambda value: sanitize_chinese_display(translate_description(str(value)), "校园闲置商品")
        )
        st.dataframe(purchase_view, hide_index=True, use_container_width=True)
    if not sales.empty:
        st.divider()
        st.markdown("**我的出售记录**")
        sales_view = sales.rename(columns={
                "order_id": "订单编号", "title": "商品", "price": "成交金额（元）",
                "delivery": "交付方式", "status": "状态", "created_at": "成交时间",
            })
        sales_view["商品"] = sales_view["商品"].map(
            lambda value: sanitize_chinese_display(translate_description(str(value)), "校园闲置商品")
        )
        st.dataframe(sales_view, hide_index=True, use_container_width=True)


def render_user_market_tab(user: dict[str, object] | None) -> None:
    st.subheader("校内上架商品")
    st.caption("这里展示平台注册用户发布的演示商品，成交后会同步写入买家和卖家的交易记录。")
    with database_connection() as connection:
        listings = pd.read_sql_query(
            """SELECT listings.*, users.username AS seller_name
            FROM listings JOIN users ON users.id = listings.seller_id
            WHERE listings.status = '在售' ORDER BY listings.id DESC""",
            connection,
        )
    if listings.empty:
        st.info("目前还没有校内用户上架商品，可以先到“发布商品”添加一件闲置。")
        return
    selected_listing_id = st.session_state.get("selected_user_listing")
    if selected_listing_id is not None:
        selected = listings[listings["id"] == int(selected_listing_id)]
        if not selected.empty:
            listing = selected.iloc[0]
            st.divider()
            listing_title = sanitize_chinese_display(translate_description(str(listing["title"])), "校园闲置商品")
            seller_name = sanitize_chinese_display(str(listing["seller_name"]), "校园用户")
            st.markdown(f"### 确认购买：{listing_title}")
            st.write(f"卖家：{seller_name} · 成交金额：¥{float(listing['price']):,.2f}")
            st.write(f"成色：{listing['condition_label']} · 交付方式：{listing['delivery']}")
            if user is None:
                st.info("请先登录后再购买校内上架商品。")
            elif int(listing["seller_id"]) == int(user["id"]):
                st.info("这是你自己发布的商品，不能购买自己的商品。")
            else:
                with st.form(f"user_listing_purchase_{int(listing['id'])}"):
                    agreed = st.checkbox("我已确认商品信息和成交金额")
                    confirmed = st.form_submit_button("确认购买", type="primary", use_container_width=True)
                if confirmed:
                    if not agreed:
                        st.warning("请先确认商品信息和成交金额。")
                    else:
                        order_id = f"校内{int(listing['id']):05d}-{secrets.token_hex(3).upper()}"
                        created_at = datetime.now().isoformat(timespec="seconds")
                        with database_connection() as connection:
                            connection.execute(
                                """INSERT INTO transactions
                                (order_id, buyer_id, seller_id, listing_id, source_item_index, title, price, delivery, status, created_at)
                                VALUES (?, ?, ?, ?, NULL, ?, ?, ?, '待交付', ?)""",
                                (
                                    order_id, int(user["id"]), int(listing["seller_id"]), int(listing["id"]),
                                    listing["title"], float(listing["price"]), listing["delivery"], created_at,
                                ),
                            )
                            connection.execute("UPDATE listings SET status = '已售出' WHERE id = ?", (int(listing["id"]),))
                        st.session_state.pop("selected_user_listing", None)
                        st.success(f"购买成功，订单编号：{order_id}。交易记录已保存。")
                        st.rerun()
            if st.button("返回上架列表", key="back_user_market"):
                st.session_state.pop("selected_user_listing", None)
                st.rerun()
    st.divider()
    for start in range(0, len(listings), 5):
        columns = st.columns(5, gap="small")
        for column, (_, listing) in zip(columns, listings.iloc[start:start + 5].iterrows()):
            with column:
                with st.container(border=True):
                    image_path = listing.get("image_path")
                    if isinstance(image_path, str) and Path(image_path).is_file():
                        st.image(image_path, use_column_width=True)
                    else:
                        st.info("暂无商品图片")
                    listing_title = sanitize_chinese_display(translate_description(str(listing["title"])), "校园闲置商品")
                    st.markdown(f"**{listing_title}**")
                    st.markdown(f"### ¥{float(listing['price']):,.2f}")
                    seller_name = sanitize_chinese_display(str(listing["seller_name"]), "校园用户")
                    st.caption(f"{listing['category']} · {listing['condition_label']} · 卖家 {seller_name}")
                    st.write(translate_listing_description(str(listing["description"]))[:100])
                    if user is not None and int(listing["seller_id"]) == int(user["id"]):
                        st.caption("这是你发布的商品")
                    elif st.button("查看并购买", key=f"buy_listing_{int(listing['id'])}", use_container_width=True):
                        st.session_state["selected_user_listing"] = int(listing["id"])
                        st.rerun()


def apply_demo_case() -> None:
    cases = {
        "蓝牙耳机": ("二手蓝牙耳机", "使用一学期，左右耳连接正常，续航正常，外壳有轻微划痕，配充电盒和充电线。", "Sony"),
        "大学教材": ("大学教材 高等数学", "大学课程教材，使用一学期，内页完整，有少量重点标记，封面边角轻微磨损。", "Unknown"),
        "运动鞋": ("二手运动鞋", "日常穿着半年，鞋底磨损较轻，鞋面干净，无开胶，尺码 42。", "Nike"),
    }
    selected = st.session_state.demo_case
    if selected not in cases:
        return
    name, description, brand = cases[selected]
    st.session_state.estimate_name = name
    st.session_state.estimate_description = description
    st.session_state.estimate_brand = "未填写" if brand == "Unknown" else brand
    st.session_state.estimate_top_category = {
        "蓝牙耳机": "Electronics",
        "大学教材": "Other",
        "运动鞋": "Women",
    }[selected]
    st.session_state.estimate_second_category = "Unknown"
    st.session_state.estimate_third_category = "Unknown"


def load_curated_images() -> dict[str, list[dict[str, str]]]:
    manifest = ROOT / "showcase_校园筛选_官方类型" / "metadata_labels.csv"
    if not manifest.exists():
        return {}
    groups: dict[str, list[dict[str, str]]] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label = row.get("category", "")
            image_path = ROOT / "showcase_校园筛选_官方类型" / row.get("showcase_path", "")
            if label and image_path.is_file():
                groups.setdefault(label, []).append({
                    "image": str(image_path),
                    "product_type": row.get("product_type", ""),
                    "item_name": row.get("item_name", ""),
                })
    return groups


def infer_image_label(query: str) -> str | None:
    search_rules = [
        ("手机配件", ["手机壳", "手机套", "手机保护壳"]),
        ("耳机音频", ["耳机", "音箱", "扬声器", "蓝牙音频"]),
        ("电脑及数码", ["电脑", "笔记本", "鼠标", "键盘", "相机", "游戏机", "手表"]),
        ("鞋包服饰", ["鞋", "运动鞋", "背包", "包", "衣服", "外套", "羽绒服"]),
        ("运动用品", ["瑜伽垫", "拉力带", "健身", "运动用品"]),
        ("学习文具", ["文具", "笔", "文件夹", "笔记本", "学习用品"]),
    ]
    for label, terms in search_rules:
        if any(term in query for term in terms):
            return label
    return None


def preferred_image_types(query: str, item: pd.Series) -> set[str] | None:
    # Use the title and structured product categories. Descriptions often mention
    # unrelated accessories, and query words must not override the listing itself.
    text = " ".join(
        str(item.get(field, ''))
        for field in ('name', 'cat1', 'cat2', 'cat3', 'category_name')
    ).lower()
    def contains(term: str) -> bool:
        if re.search(r"[a-z]", term):
            return re.search(rf"(?<![a-z]){re.escape(term)}s?(?![a-z])", text) is not None
        return term in text

    rules = [
        ({"HEADPHONES"}, ["耳机", "耳塞", "headphone", "earbud", "headset"]),
        ({"SPEAKERS"}, ["音箱", "扬声器", "speaker"]),
        ({"NOTEBOOK_COMPUTER"}, ["笔记本电脑", "laptop", "notebook computer"]),
        ({"PERSONAL_COMPUTER"}, ["台式电脑", "个人电脑", "desktop computer", "personal computer"]),
        ({"COMPUTER_INPUT_DEVICE"}, ["键盘", "鼠标", "keyboard", "mouse"]),
        ({"VIDEO_GAME_ACCESSORIES"}, ["游戏手柄", "游戏机", "游戏设备", "video game", "console", "controller", "gamepad"]),
        ({"WATCH"}, ["手表", "watch"]),
        ({"CELLULAR_PHONE_CASE"}, ["手机壳", "手机套", "phone case", "mobile cover", "iphone case"]),
        ({"SHOES"}, ["运动鞋", "sneaker", "shoes", "shoe"]),
        ({"BOOT"}, ["靴", "boots", "boot"]),
        ({"SANDAL"}, ["凉鞋", "sandals", "sandal"]),
        ({"BACKPACK"}, ["背包", "backpack"]),
        ({"EXERCISE_MAT"}, ["瑜伽垫", "运动垫", "exercise mat", "yoga mat"]),
        ({"EXERCISE_BAND"}, ["拉力带", "阻力带", "exercise band", "resistance band"]),
        ({"WRITING_INSTRUMENT"}, ["钢笔", "圆珠笔", "书写笔", "writing instrument", "ballpoint pen", "fountain pen"]),
        ({"MARKING_PEN"}, ["马克笔", "彩笔", "marker"]),
        ({"INPUT_PEN"}, ["触控笔", "手写笔", "stylus"]),
        ({"STORAGE_BINDER"}, ["文件夹", "活页夹", "binder", "folder"]),
        # 当前校园展示素材没有裙装、上衣等服饰图，识别到这些词时返回空集合，避免拿鞋图冒充服装图。
        (set(), ["裙", "skirt", "dress", "blouse", "shirt", "top", "jacket", "coat", "jeans", "pants", "clothing"]),
    ]
    for product_types, terms in rules:
        if any(contains(term) for term in terms):
            return product_types
    return None


def product_type_label(value: str) -> str:
    labels = {
        "CELLULAR_PHONE_CASE": "手机壳及手机配件",
        "HEADPHONES": "耳机",
        "SPEAKERS": "音箱",
        "COMPUTER_ADD_ON": "电脑配件",
        "COMPUTER_INPUT_DEVICE": "电脑输入设备",
        "PERSONAL_COMPUTER": "个人电脑",
        "NOTEBOOK_COMPUTER": "笔记本电脑",
        "VIDEO_GAME_ACCESSORIES": "游戏设备配件",
        "WATCH": "手表",
        "CAMERA_BAGS_AND_CASES": "相机包",
        "CAMERA_OTHER_ACCESSORIES": "相机配件",
        "CAMERA_TRIPOD": "相机支架",
        "CAMERA_LENS_FILTERS": "镜头滤镜",
        "SHOES": "鞋类",
        "SANDAL": "凉鞋",
        "BOOT": "靴子",
        "BACKPACK": "背包",
        "HANDBAG": "手提包",
        "EXERCISE_MAT": "运动垫",
        "EXERCISE_BAND": "拉力带",
        "WRITING_INSTRUMENT": "书写文具",
        "MARKING_PEN": "彩笔",
        "INPUT_PEN": "触控笔",
        "STORAGE_BINDER": "文件夹",
        "PAPER_PRODUCT": "纸品文具",
    }
    return labels.get(value, "商品示例")


def browse_image_label(item: pd.Series) -> str | None:
    product_types = preferred_image_types("", item)
    if not product_types:
        return None
    type_categories = {
        "HEADPHONES": "耳机音频", "SPEAKERS": "耳机音频",
        "CELLULAR_PHONE_CASE": "手机配件",
        "NOTEBOOK_COMPUTER": "电脑及数码", "PERSONAL_COMPUTER": "电脑及数码",
        "COMPUTER_INPUT_DEVICE": "电脑及数码", "VIDEO_GAME_ACCESSORIES": "电脑及数码",
        "WATCH": "电脑及数码",
        "SHOES": "鞋包服饰", "BOOT": "鞋包服饰", "SANDAL": "鞋包服饰",
        "BACKPACK": "鞋包服饰",
        "EXERCISE_MAT": "运动用品", "EXERCISE_BAND": "运动用品",
        "WRITING_INSTRUMENT": "学习文具", "MARKING_PEN": "学习文具",
        "INPUT_PEN": "学习文具", "STORAGE_BINDER": "学习文具",
    }
    return type_categories.get(next(iter(product_types)))


def curated_examples_for_item(
    item: pd.Series,
    curated_images: dict[str, list[dict[str, str]]],
    query: str = "",
) -> list[dict[str, str]]:
    """按商品类别和具体类型筛选示例图，避免用不相干图片填充卡片。"""
    image_category = browse_image_label(item)
    product_types = preferred_image_types(query, item)
    if not image_category or not product_types:
        return []
    return [
        example
        for example in curated_images.get(image_category, [])
        if example["product_type"] in product_types
    ]


def browse_results(
    query: str,
    category: str,
    sort_mode: str,
    data: pd.DataFrame,
    vectorizer,
    matrix,
    title_vectorizer,
    title_matrix,
    price_min: float = 0.0,
    price_max: float | None = None,
    condition_filter: int | None = None,
    only_with_images: bool = False,
) -> pd.DataFrame:
    category_filter = None if category == "不限制" else category
    if query.strip():
        return find_similar(
            query,
            data,
            vectorizer,
            matrix,
            title_vectorizer,
            title_matrix,
            top_n=12,
            category_filter=category_filter,
            sort_mode=sort_mode,
            price_min=price_min,
            price_max=price_max,
            condition_filter=condition_filter,
            only_with_images=only_with_images,
        )
    subset = data
    if category_filter == "校园常见":
        subset = subset[subset["cat1"].isin(CAMPUS_TOP_CATEGORIES)]
    elif category_filter is not None:
        subset = subset[subset["cat1"] == category_filter]
    subset = subset[subset["price_cny"] >= price_min]
    if price_max is not None:
        subset = subset[subset["price_cny"] <= price_max]
    if condition_filter is not None:
        subset = subset[pd.to_numeric(subset["item_condition_id"], errors="coerce") == condition_filter]
    if only_with_images:
        subset = subset[subset["showcase_category"].notna()]
    if subset.empty:
        return subset
    sample = subset.sample(n=min(12, len(subset)), random_state=20261024).copy()
    if sort_mode == "价格从低到高":
        sample = sample.sort_values("price_cny")
    elif sort_mode == "价格从高到低":
        sample = sample.sort_values("price_cny", ascending=False)
    return sample


def render_clickable_image(
    image_path: str, detail_index: int, source: str = "browse", reference_index: int | None = None
) -> None:
    image_data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    reference_param = f"&recommend_from={reference_index}" if reference_index is not None else ""
    st.markdown(
        f'<a href="?detail={int(detail_index)}&detail_from={source}{reference_param}" target="_self" aria-label="查看商品详情">'
        f'<img src="data:image/jpeg;base64,{image_data}" alt="点击查看商品详情" '
        'style="width:100%;height:190px;object-fit:contain;cursor:pointer;display:block">'
        '</a>',
        unsafe_allow_html=True,
    )


def render_detail_link(detail_index: int, source: str, reference_index: int | None = None) -> None:
    reference_param = f"&recommend_from={reference_index}" if reference_index is not None else ""
    st.markdown(
        f'<a class="detail-link" href="?detail={int(detail_index)}&detail_from={source}{reference_param}" target="_self">查看详情</a>',
        unsafe_allow_html=True,
    )


def render_recommendation_cards(
    reference_index: int,
    results: pd.DataFrame,
    data: pd.DataFrame,
    curated_images: dict[str, list[dict[str, str]]],
) -> None:
    reference_price = float(data.loc[reference_index, "price_cny"])
    for start in range(0, len(results), 5):
        columns = st.columns(5, gap="small")
        for column, (item_index, item) in zip(columns, results.iloc[start:start + 5].iterrows()):
            with column:
                with st.container(border=True):
                    examples = curated_examples_for_item(item, curated_images, str(item["name"]))
                    if examples:
                        example = examples[int(item_index) % len(examples)]
                        render_clickable_image(
                            example["image"], int(item_index), source="similar", reference_index=reference_index
                        )
                        st.caption("类别图，非实拍")
                    else:
                        st.info("暂无类别示例图")
                    st.markdown(f"**{display_product_name(item)}**")
                    st.markdown(f"### ¥{float(item['price_cny']):,.2f}")
                    difference = float(item["price_cny"]) - reference_price
                    if difference < 0:
                        price_comparison = f"低 ¥{abs(difference):,.2f}"
                    elif difference > 0:
                        price_comparison = f"高 ¥{difference:,.2f}"
                    else:
                        price_comparison = "同价"
                    st.caption(f"匹配 {float(item['similarity']):.0%}")
                    st.caption(f"差价：{price_comparison}")
                    st.caption(f"成色：{condition_labels_text(int(item['item_condition_id']))}")
                    render_detail_link(int(item_index), "similar", reference_index)


def render_product_detail(
    item: pd.Series,
    item_index: int,
    curated_images: dict[str, list[dict[str, str]]],
    model,
    data: pd.DataFrame,
    matrix,
    title_matrix,
    metrics: dict[str, object],
    user: dict[str, object] | None,
) -> None:
    source = st.query_params.get("detail_from") or st.session_state.get("detail_source") or "browse"
    back_label = "返回参照商品" if source == "similar" else "返回商品列表"
    try:
        origin_index = int(st.query_params.get("recommend_from")) if source == "similar" else None
    except (TypeError, ValueError):
        origin_index = None
    if source == "similar" and origin_index in data.index:
        st.markdown(
            f'<a class="detail-link" href="?detail={origin_index}" target="_self">{back_label}</a>',
            unsafe_allow_html=True,
        )
    elif st.button(back_label, key="detail_back"):
        st.session_state.pop("selected_detail_index", None)
        st.session_state.pop("detail_source", None)
        st.session_state.pop("detail_estimate_open", None)
        st.session_state.pop(f"detail_estimate_result_{item_index}", None)
        for key in ("detail", "detail_from", "recommend_from"):
            if key in st.query_params:
                del st.query_params[key]
        st.rerun()

    st.divider()
    detail_title = display_product_name(item)
    st.title(detail_title or "未提供商品名称")
    detail_left, detail_right = st.columns([1, 1.5], gap="large")
    image_category = browse_image_label(item)
    detail_examples = curated_examples_for_item(
        item,
        curated_images,
        " ".join([str(item.get("name", "")), str(item.get("item_description", ""))]),
    )
    if detail_examples:
        example_index = int(item_index) % len(detail_examples)
        detail_left.image(detail_examples[example_index]["image"], use_column_width=True)
        detail_left.caption(f"{product_type_label(detail_examples[example_index]['product_type'])}示例图 · 与当前商品记录不对应")
    else:
        detail_left.info("暂无对应类别示例图")
    with detail_right:
        st.metric("记录价格", f"¥{float(item['price_cny']):,.2f}")
        st.write(f"**商品大类：** {category_label(str(item['cat1']))}")
        st.write(f"**商品小类：** {category_label(str(item['cat2']))}")
        st.write(f"**商品细分类：** {category_label(str(item['cat3']))}")
        brand_value = str(item["brand_name"])
        st.write(f"**品牌：** {brand_display_name(brand_value)}")
        condition_labels = {1: "全新", 2: "较新", 3: "一般", 4: "较旧", 5: "明显使用痕迹"}
        condition_value = int(item["item_condition_id"])
        st.write(f"**成色等级：** {condition_labels.get(condition_value, '未提供')}")
        st.write(f"**运费：** {'卖家承担' if int(item['shipping']) else '买家承担'}")
        raw_description = str(item["item_description"])
        if raw_description not in {"Unknown", "nan"}:
            st.markdown("**商品描述（中文翻译）**")
            try:
                translated_description = translate_listing_description(raw_description)
                st.write(translated_description or "未提供描述")
            except Exception:
                st.write(sanitize_chinese_display(raw_description, "商品描述暂未提供"))
            st.caption("页面描述仅供浏览参考，价格模型仍使用原始数据特征。")
        else:
            st.markdown("**商品描述**")
            st.write("未提供描述")
    st.caption("数据来自美国二手交易平台，价格按参考汇率换算为人民币，仅用于课程演示，不代表校园真实成交价。")

    st.divider()
    st.subheader("同类商品对比")
    if st.button("查找与这件商品相似的商品", key=f"recommend_open_button_{item_index}", use_container_width=True):
        st.session_state[f"recommend_open_{item_index}"] = True
    if st.session_state.get(f"recommend_open_{item_index}"):
        recommendations = recommend_for_item(item_index, data, matrix, title_matrix)
        if recommendations.empty:
            st.info("暂时没有找到足够相近的同类商品。")
        else:
            st.caption("根据这件商品的标题、描述和细分类推荐；价格差以当前记录价格为参照。")
            render_recommendation_cards(item_index, recommendations, data, curated_images)

    st.divider()
    render_purchase_flow(item, item_index, user)
    st.divider()
    estimate_header, estimate_action = st.columns([2.8, 1], gap="large")
    with estimate_header:
        st.subheader("智能估价")
        st.caption("使用当前商品信息预填估价表单，可修改后查看模型参考价。")
    with estimate_action:
        if st.button(
            "使用当前商品信息估价",
            key=f"open_detail_estimate_{item_index}",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["detail_estimate_open"] = True
            st.rerun()

    if not st.session_state.get("detail_estimate_open"):
        return

    raw_name = str(item.get("name", ""))
    raw_description = str(item.get("item_description", ""))
    try:
        default_name = display_product_name(item)
    except Exception:
        default_name = raw_name
    try:
        default_description = translate_description(raw_description) if raw_description not in {"Unknown", "nan"} else ""
    except Exception:
        default_description = raw_description if raw_description not in {"Unknown", "nan"} else ""
    default_top = str(item.get("cat1", "Unknown"))
    top_categories = sorted(data["cat1"].dropna().unique().tolist())
    if default_top not in top_categories:
        default_top = top_categories[0] if top_categories else "Unknown"
    second_options, third_options = category_options(data, default_top)
    second_values = ["Unknown"] + second_options
    third_values = ["Unknown"] + third_options
    default_second = str(item.get("cat2", "Unknown"))
    default_third = str(item.get("cat3", "Unknown"))
    if default_second not in second_values:
        default_second = "Unknown"
    if default_third not in third_values:
        default_third = "Unknown"
    condition_labels = {1: "全新", 2: "较新", 3: "一般", 4: "较旧", 5: "明显使用痕迹"}
    default_condition = int(item.get("item_condition_id", 3))
    if default_condition not in condition_labels:
        default_condition = 3
    brand_value = str(item.get("brand_name", "Unknown"))
    default_brand = "" if brand_value in {"Unknown", "None", "nan"} else brand_display_name(brand_value)

    with st.form(f"detail_estimate_form_{item_index}"):
        form_left, form_right = st.columns([1, 1], gap="large")
        with form_left:
            estimate_name = st.text_input("商品名称", value=default_name, key=f"detail_estimate_name_{item_index}")
            estimate_top = st.selectbox(
                "商品大类",
                top_categories,
                index=top_categories.index(default_top) if default_top in top_categories else 0,
                format_func=lambda value: category_label(value),
                key=f"detail_estimate_top_{item_index}",
            )
            estimate_second_options, estimate_third_options = category_options(data, estimate_top)
            estimate_second_values = ["Unknown"] + estimate_second_options
            estimate_third_values = ["Unknown"] + estimate_third_options
            estimate_second = st.selectbox(
                "商品小类",
                estimate_second_values,
                index=estimate_second_values.index(default_second) if default_second in estimate_second_values else 0,
                format_func=lambda value: category_label(value),
                key=f"detail_estimate_second_{item_index}",
            )
            estimate_third = st.selectbox(
                "商品细分类",
                estimate_third_values,
                index=estimate_third_values.index(default_third) if default_third in estimate_third_values else 0,
                format_func=lambda value: category_label(value),
                key=f"detail_estimate_third_{item_index}",
            )
            estimate_brand = st.text_input("品牌（可选）", value=default_brand, key=f"detail_estimate_brand_{item_index}")
        with form_right:
            estimate_condition = st.selectbox(
                "商品成色",
                options=list(condition_labels),
                index=list(condition_labels).index(default_condition),
                format_func=lambda value: condition_labels[value],
                key=f"detail_estimate_condition_{item_index}",
            )
            estimate_shipping = st.checkbox(
                "卖家承担运费",
                value=bool(int(item.get("shipping", 0))),
                key=f"detail_estimate_shipping_{item_index}",
            )
            estimate_description = st.text_area(
                "商品描述",
                value=default_description,
                height=132,
                key=f"detail_estimate_description_{item_index}",
            )
            estimate_quoted_price = st.number_input(
                "卖家报价（元，可选）",
                min_value=0.0,
                value=float(item.get("price_cny", 0.0)),
                step=10.0,
                key=f"detail_estimate_quoted_{item_index}",
            )
        submitted = st.form_submit_button("生成这件商品的参考价", type="primary", use_container_width=True)

    result_key = f"detail_estimate_result_{item_index}"
    if submitted:
        st.session_state[result_key] = calculate_reference_estimate(
            model,
            data,
            metrics,
            estimate_name,
            estimate_description,
            estimate_top,
            estimate_second,
            estimate_third,
            estimate_brand,
            int(estimate_condition),
            estimate_shipping,
            float(estimate_quoted_price),
        )
    result = st.session_state.get(result_key)
    if result:
        render_estimate_insights(
            result,
            "卖家出售",
            str(result.get("top_category", item.get("cat1", "Unknown"))),
            str(result.get("second_category", item.get("cat2", "Unknown"))),
            str(result.get("third_category", item.get("cat3", "Unknown"))),
            str(result.get("name", default_name)),
            str(result.get("description", default_description)),
            str(result.get("brand", default_brand)),
        )


def fallback_translate_text(text: str) -> str:
    """使用本地商品词汇翻译常见标题和描述，不访问在线服务。"""
    replacements = [
        ("wireless headphones", "无线耳机"), ("wireless speaker", "无线音箱"),
        ("bluetooth speaker", "蓝牙音箱"), ("bluetooth", "蓝牙"),
        ("paperback", "平装书"), ("hardcover", "精装书"), ("nwt", "全新带吊牌"),
        ("nwot", "全新无吊牌"), ("zip up", "拉链款"), ("hello kitty", "凯蒂猫"),
        ("forever 21", "永远二十一"), ("under armour", "安德玛"), ("tory burch", "汤丽柏琦"),
        ("brandy melville", "布兰迪梅尔维尔"), ("blue microphones", "蓝色麦克风"),
        ("lularoe", "露露罗"), ("ll bean", "豆角牌"), ("iphone", "苹果手机"),
        ("headphones", "耳机"), ("headset", "耳机"), ("earbuds", "耳塞"),
        ("speaker", "音箱"), ("laptop", "笔记本电脑"), ("notebook", "笔记本电脑"),
        ("computer", "电脑"), ("keyboard", "键盘"), ("mouse", "鼠标"),
        ("phone case", "手机壳"), ("cell phone", "手机"), ("camera", "相机"),
        ("backpack", "背包"), ("sneakers", "运动鞋"), ("shoes", "鞋子"),
        ("sandals", "凉鞋"), ("boots", "靴子"), ("sunglasses", "太阳镜"),
        ("watch", "手表"), ("controller", "游戏手柄"), ("game", "游戏"),
        ("skirt", "半身裙"), ("dress", "连衣裙"), ("shirt", "衬衫"),
        ("blouse", "女式衬衫"), ("jacket", "夹克"), ("coat", "外套"),
        ("pants", "裤子"), ("jeans", "牛仔裤"), ("socks", "袜子"),
        ("sandals", "凉鞋"), ("ballet flats", "芭蕾平底鞋"), ("boots", "靴子"),
        ("book", "书籍"), ("textbook", "教材"), ("calculator", "计算器"),
        ("used", "二手"), ("pre owned", "二手"), ("brand new", "全新"),
        ("new", "全新"), ("good condition", "成色良好"), ("very good", "成色很好"),
        ("with case", "带保护壳"), ("with charger", "带充电器"),
        ("with cable", "带数据线"), ("free shipping", "包邮"),
        ("wireless", "无线"), ("portable", "便携式"), ("small", "小号"),
        ("large", "大号"), ("black", "黑色"), ("white", "白色"),
        ("blue", "蓝色"), ("pink", "粉色"), ("red", "红色"),
        ("gold", "金色"), ("silver", "银色"), ("with", "带"),
        ("size", "尺码"), ("for", "适用"), ("lot of", "一批"),
        ("pair", "一双"), ("pairs", "双"), ("condition", "成色"),
        ("works normally", "功能正常"), ("minor scratches", "轻微划痕"),
        ("barely used", "几乎未使用"), ("slight wear", "轻微磨损"),
        ("battery", "电池"), ("batteries", "电池"), ("scratch", "划痕"),
        ("scratches", "划痕"), ("minimal", "很少"), ("discoloration", "褪色"),
        ("rubbing off", "掉色"), ("average size", "普通尺码"),
        ("fit", "适合"), ("clean", "干净"), ("slight dent", "轻微凹痕"),
        ("includes", "包含"), ("includes cable", "包含数据线"),
        ("authentic", "正品"), ("sealed", "未拆封"),
    ]
    result = text
    for source, target in replacements:
        result = re.sub(
            rf"(?<![A-Za-z]){re.escape(source)}(?![A-Za-z])",
            target,
            result,
            flags=re.IGNORECASE,
        )
    return sanitize_chinese_display(result)


@st.cache_data(show_spinner=False)
def load_translation_cache(cache_mtime_ns: int) -> dict[str, str]:
    if TRANSLATION_CACHE_PATH.exists():
        try:
            loaded = json.loads(TRANSLATION_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return {str(key): str(value) for key, value in loaded.items()}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    return {}


def translate_description(text: str) -> str:
    if not text.strip() or text in {"Unknown", "nan"}:
        return ""
    cache_mtime_ns = TRANSLATION_CACHE_PATH.stat().st_mtime_ns if TRANSLATION_CACHE_PATH.exists() else 0
    cache = load_translation_cache(cache_mtime_ns)
    cache_key = text.strip()
    if cache_key in cache:
        # 旧缓存可能只完成了部分翻译，重新经过本地词典补齐常见词。
        return fallback_translate_text(cache[cache_key])
    translated = fallback_translate_text(cache_key)
    return translated


def translate_listing_description(text: str) -> str:
    """Avoid displaying isolated translated keywords as a complete description."""
    translated = translate_description(text)
    if len(text) >= 40:
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", translated))
        if "•" in translated or chinese_count < max(8, int(len(text) * 0.18)):
            return "该条商品的描述暂无可靠中文译文。"
    return translated


def display_product_name(item: pd.Series) -> str:
    """把商品标题整理成中文显示，避免原始英文标题直接出现在页面上。"""
    raw_name = str(item.get("name", "")).strip()
    if not raw_name or raw_name in {"Unknown", "nan"}:
        return f"{category_label(str(item.get('cat2', item.get('cat1', ''))))}商品"
    try:
        translated = translate_description(raw_name).strip()
    except Exception:
        translated = ""
    # 品牌和型号常常不会被在线翻译，展示时保留中文、数字和常用标点，避免英文标题破坏页面语言一致性。
    cleaned = re.sub(r"[A-Za-z]+", "", translated or raw_name)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_,/|")
    if not re.search(r"[\u4e00-\u9fff]", cleaned) or cleaned == "商品信息暂未提供":
        return f"{category_label(str(item.get('cat2', item.get('cat1', ''))))}商品"
    return cleaned[:96]


def main() -> None:
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        st.error("请先运行 python train_model.py 生成模型，并确认 data/data.parquet 存在。")
        st.stop()

    initialize_database()
    bundle, data, vectorizer, matrix, title_vectorizer, title_matrix, metrics = load_assets()
    curated_images = load_curated_images()
    curated_count = sum(map(len, curated_images.values()))
    model = bundle["model"]
    st.markdown(
        """
        <style>
        .stApp { background: #f4f6f3; color: #1b2824; overflow-x: hidden; }
        [data-testid="stAppViewContainer"] > .main { overflow-x: visible !important; }
        .block-container { width: calc(100% - 2.5rem); max-width: none; margin: 0;
                padding: 1.2rem 1.5rem 2.2rem; box-sizing: border-box; }
        [data-testid="stHorizontalBlock"] { width: calc(100% - 1.5rem) !important; max-width: calc(100% - 1.5rem) !important;
                min-width: 0 !important; box-sizing: border-box; }
        [data-testid="stHorizontalBlock"], [data-testid="column"] { min-width: 0 !important; box-sizing: border-box; }
        [data-testid="stHorizontalBlock"] > div { min-width: 0 !important; max-width: 100% !important; }
        [data-testid="stSelectbox"], [data-testid="stNumberInput"],
        [data-testid="stTextInput"], [data-testid="stTextArea"],
        [data-testid="stFileUploader"], [data-testid="stButton"] {
            width: calc(100% - .8rem) !important; max-width: calc(100% - .8rem) !important;
            min-width: 0 !important; box-sizing: border-box !important;
        }
        [data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div,
        [data-testid="stTextInput"] > div, [data-testid="stTextArea"] > div,
        [data-testid="stFileUploader"] > div, [data-testid="stButton"] > button {
            width: 100% !important; max-width: 100% !important; box-sizing: border-box !important;
        }
        .stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"] {
            color: #26352f !important;
        }
        .stApp [data-testid="stCaptionContainer"] { color: #586760 !important; }
        .stApp .hero .hero-title { color: #f8f7ef !important; }
        .stApp .hero .hero-sub { color: #d6e4dc !important; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="input"] > div, .stApp [data-baseweb="textarea"] > div {
            background-color: #ffffff !important;
            color: #1b2824 !important;
            border-color: #cbd6cf !important;
            border-radius: 8px !important;
            box-sizing: border-box !important;
            max-width: 100% !important;
        }
        .stApp input, .stApp textarea { min-width: 0 !important; }
        .stApp input::placeholder, .stApp textarea::placeholder { color: #75827b !important; opacity: 1; }
        .stApp [data-baseweb="select"] span, .stApp [data-baseweb="select"] input,
        .stApp [data-baseweb="select"] svg { color: #26352f !important; fill: #26352f !important; }
        [data-baseweb="popover"] ul, [data-baseweb="popover"] li,
        [data-baseweb="menu"] { background-color: #ffffff !important; color: #26352f !important; }
        [data-baseweb="popover"] li:hover { background-color: #e9efeb !important; }
        .stApp [data-testid="stCheckbox"] label, .stApp [data-testid="stRadio"] label {
            color: #26352f !important;
        }
        .stApp [data-testid="stAlert"] p { color: #26352f !important; }
        .stMarkdown, [data-testid="stCaptionContainer"], [data-testid="stAlert"] { overflow-wrap: anywhere; }
        [data-testid="stHeader"] { display: none !important; height: 0 !important; }
        [data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="stDecoration"] {
            display: none !important; visibility: hidden !important; height: 0 !important;
        }
        [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], .stDeployButton {
            display: none !important; visibility: hidden !important;
        }
        [data-testid="stToolbar"] *, [data-testid="stStatusWidget"] * { display: none !important; }
        #MainMenu, footer { display: none !important; visibility: hidden !important; }
        [data-testid="stSidebar"] { width: 330px !important; min-width: 330px !important; max-width: 330px !important; }
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebar"] section,
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { width: 100% !important; max-width: 100% !important; box-sizing: border-box; }
        [data-testid="stSidebar"] { background: #e9efeb; border-right: 1px solid #d8e1dc; overflow-x: hidden; }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #35443e !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { max-width: 100%; overflow-wrap: anywhere; white-space: normal !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { max-width: 100% !important; white-space: normal !important;
                overflow-wrap: anywhere !important; word-break: break-word !important; }
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-testid="stSidebar"] [data-baseweb="password"] > div { max-width: 100% !important; }
        h1, h2, h3 { color: #1b2824; letter-spacing: 0 !important; }
        h1 { font-size: 2.15rem !important; }
        .hero { width: calc(100% - 1.5rem); max-width: calc(100% - 1.5rem); box-sizing: border-box; padding: 1.45rem 1.65rem 1.35rem; background: #173c34;
                color: #f8f7ef; border-radius: 10px; margin: .2rem 0 1rem;
                border-left: 5px solid #d26b55; box-shadow: 0 8px 22px rgba(23,60,52,.12); }
        .hero-title { position: relative; z-index: 1; font-size: 1.8rem; font-weight: 750; margin: 0 0 .28rem; }
        .hero-sub { position: relative; z-index: 1; color: #d6e4dc; margin: 0; font-size: 1rem; }
        [data-testid="stMetric"] { background: #fff; border: 1px solid #dfe6e1;
             border-radius: 7px; padding: 1rem 1.1rem; box-sizing: border-box; }
        [data-testid="stMetric"]:has([data-testid="stMetricDelta"]) { min-height: 172px; }
        [data-testid="stMetric"] { min-height: 172px; }
        [data-testid="stMetricLabel"] { color: #56665f; }
        [data-testid="stSidebar"] [data-testid="stMetric"] { min-height: 84px; padding: .75rem 1rem; margin-bottom: .55rem; }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.55rem !important; }
        [data-testid="stSidebar"] [data-testid="stAlert"] { padding: .9rem 1rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
        [data-baseweb="tab-list"] { gap: .35rem; padding: .25rem .25rem 0; border-bottom: 1px solid #d8e1dc; }
        [data-baseweb="tab"] { background: transparent; border-radius: 7px 7px 0 0; padding: .62rem .95rem; color: #586760; transition: background .15s ease, color .15s ease; }
        [data-baseweb="tab"]:hover { background: #e8ede9; color: #17614d; }
        [aria-selected="true"][data-baseweb="tab"] { background: #fff; color: #17614d; box-shadow: inset 0 -2px 0 #d26b55; }
        div.stButton > button[kind="primary"] { background: #d26b55; border: 0; color: white; }
        div.stButton > button, .stApp button { border-radius: 8px !important; }
        [data-testid="stAlert"], [data-testid="stExpander"] { border-radius: 8px !important; overflow: hidden; }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { border-radius: 8px !important; }
        .detail-link { display: flex; align-items: center; justify-content: center; min-height: 40px;
            width: calc(100% - 1rem); max-width: calc(100% - 1rem); box-sizing: border-box; padding: .45rem .35rem; border: 1px solid #cbd6cf;
            border-radius: 6px; background: #fff; color: #26352f !important; text-decoration: none !important; }
        .detail-link:hover { border-color: #c75b43; color: #9d3e2d !important; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-color: #dfe6e1; border-radius: 9px; background: #fff; box-shadow: 0 3px 12px rgba(27,40,36,.045); }
        [data-testid="stVerticalBlockBorderWrapper"] { overflow: hidden; }
        [data-testid="stVerticalBlockBorderWrapper"] > div { padding: .72rem .78rem .78rem; }
        [data-testid="stImage"] { margin-bottom: .2rem; }
        [data-testid="stImage"] img { max-height: 112px; object-fit: contain; border-radius: 6px; background: #f7f9f7; }
        [data-testid="stVerticalBlockBorderWrapper"] h3 { margin-top: .25rem; margin-bottom: .25rem; }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p { margin-bottom: .3rem; }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] { min-height: 1.25rem; }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] h3 { font-size: 1.05rem !important; line-height: 1.25 !important; }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] button { min-height: 2.1rem; padding: .25rem .35rem !important; font-size: .86rem !important; }
        [data-testid="stMarkdownContainer"] p { overflow-wrap: anywhere; }
        div[data-baseweb="select"] > div { min-height: 42px; }
        @media (max-width: 850px) {
            .block-container { width: calc(100% - 1rem); padding-left: .5rem; padding-right: .5rem; }
            [data-testid="stSidebar"] { width: 285px !important; min-width: 285px !important; max-width: 285px !important; }
            .hero { padding: 1.15rem 1.05rem; }
            .hero-title { font-size: 1.5rem; }
            h1 { font-size: 1.85rem !important; }
            [data-baseweb="tab"] { padding: .48rem .58rem; }
            [data-testid="stImage"] img { max-height: 96px; }
        }
        .eyebrow { color: #b55340; text-transform: uppercase; font-size: .75rem; font-weight: 700; }
        .price-radar { background: #fff; border: 1px solid #dfe6e1; border-radius: 7px; padding: 1rem 1.1rem 1.15rem; margin: 1rem 0; }
        .price-radar-head { display: flex; justify-content: space-between; gap: 1rem; color: #26352f; margin-bottom: 1.5rem; }
        .price-track { height: 12px; position: relative; background: #e8ede9; border-radius: 8px; margin: 0 .35rem; }
        .price-range { position: absolute; top: 0; height: 12px; background: #b9d7c8; border-radius: 8px; }
        .price-marker { position: absolute; top: -8px; width: 3px; height: 28px; background: #173c34; transform: translateX(-50%); z-index: 2; }
        .price-marker.current { background: #d26b55; z-index: 3; }
        .price-marker span { position: absolute; top: 30px; left: 50%; white-space: nowrap; transform: translateX(-50%); font-size: .74rem; color: #56665f; }
        .price-radar-labels { display: flex; justify-content: space-between; color: #56665f; font-size: .78rem; margin-top: 1.7rem; }
        /* Streamlit 文件上传器的默认英文提示改为中文。 */
        [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child { font-size: 0 !important; }
        [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child::after {
            content: "将文件拖放到这里"; font-size: .95rem; color: #26352f;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2) { font-size: 0 !important; }
        [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2)::after {
            content: "单个文件最大 200MB · 支持 JPG、JPEG、PNG、WEBP"; font-size: .78rem; color: #75827b;
        }
        [data-testid="stFileUploaderDropzone"] button { font-size: 0 !important; }
        [data-testid="stFileUploaderDropzone"] button::after {
            content: "选择文件"; font-size: .9rem; color: #26352f;
        }
        </style>
        <section class="hero">
          <p class="hero-title">校园淘淘</p>
          <p class="hero-sub">让每件闲置，都有更合适的价格。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # 账号入口放在侧栏顶部，用户打开页面即可看到登录区域。
    user = render_account_sidebar()

    with st.sidebar:
        st.header("项目概况")
        st.metric("商品样本", f"{metrics['rows_total']:,} 条")
        st.metric("已整理展示图片", f"{curated_count:,} 张")
        st.metric("测试集平均误差", f"¥{metrics['mae_cny']:.2f}")
        with st.expander("课堂演示流程"):
            st.markdown("1. 在**商品浏览**打开一件商品。\n2. 在详情页查找相似商品，比较文本相似度和价格差。\n3. 使用智能估价，切换买家/卖家场景并修改成色。\n4. 在**模型表现**查看四种方法的测试误差。")

    detail_value = st.query_params.get("detail")
    if detail_value is None:
        detail_value = st.session_state.get("selected_detail_index")
    try:
        selected_detail_index = int(detail_value) if detail_value is not None else None
    except (TypeError, ValueError):
        selected_detail_index = None
    if selected_detail_index is not None and selected_detail_index in data.index:
        render_product_detail(
            data.loc[selected_detail_index],
            selected_detail_index,
            curated_images,
            model,
            data,
            matrix,
            title_matrix,
            metrics,
            user,
        )
        return

    browse_tab, estimate_tab, search_tab, gallery_tab, model_tab, publish_tab, market_tab, account_tab = st.tabs(
        ["商品浏览", "智能估价", "相似商品", "校园图片示例", "模型表现", "发布商品", "校内上架", "账号与记录"]
    )

    with estimate_tab:
        st.subheader("商品估价")
        st.caption("填写商品信息，模型会综合标题、描述、分类、品牌、成色和运费估算参考价。")
        estimate_mode = st.radio(
            "使用场景",
            ["买家购买", "卖家出售"],
            horizontal=True,
            key="estimate_mode",
            help="买家模式重点判断报价是否合理，卖家模式重点给出建议挂牌价。",
        )
        st.selectbox(
            "快速演示案例",
            ["不使用案例", "蓝牙耳机", "大学教材", "运动鞋"],
            key="demo_case",
            on_change=apply_demo_case,
        )
        left, right = st.columns([1, 1], gap="large")
        with left:
            name = st.text_input("商品名称", "二手蓝牙耳机", key="estimate_name")
            top_categories = sorted(data["cat1"].dropna().unique().tolist())
            top_labels = category_display_map(top_categories)
            top_category = st.selectbox("商品大类", top_categories, format_func=lambda value: top_labels[value], index=top_categories.index("Electronics") if "Electronics" in top_categories else 0, key="estimate_top_category")
            second_options, third_options = category_options(data, top_category)
            second_values = ["Unknown"] + second_options
            third_values = ["Unknown"] + third_options
            second_labels = category_display_map(second_values)
            third_labels = category_display_map(third_values)
            second_category = st.selectbox("商品小类", second_values, format_func=lambda value: second_labels[value], key="estimate_second_category")
            third_category = st.selectbox("商品细分类", third_values, format_func=lambda value: third_labels[value], key="estimate_third_category")
            brand = st.text_input("品牌（可选）", "未填写", key="estimate_brand") or "Unknown"
        with right:
            condition = st.selectbox("商品成色", options=[1, 2, 3, 4, 5], index=1, format_func=lambda x: {1: "全新", 2: "较新", 3: "一般", 4: "较旧", 5: "明显使用痕迹"}[x], help="数据集使用 1 到 5 的成色编码")
            shipping = st.checkbox("卖家承担运费")
            description = st.text_area(
                "商品描述",
                "使用一学期，功能正常，有轻微划痕，包含充电线。",
                height=132,
                key="estimate_description",
            )
            quoted_price = st.number_input("卖家报价（元，可选）", min_value=0.0, value=300.0, step=10.0)

        if st.button("生成参考价格", type="primary", use_container_width=True):
            estimate_result = calculate_reference_estimate(
                model, data, metrics, name, description, top_category, second_category,
                third_category, brand, int(condition), shipping, float(quoted_price)
            )
            st.divider()
            render_estimate_insights(
                estimate_result,
                estimate_mode,
                top_category,
                second_category,
                third_category,
                name,
                description,
                brand,
            )
            with st.expander("查看模型依据"):
                st.write("模型综合使用商品名称、商品描述、商品大类/小类/细分类、品牌、成色等级和运费信息。")
                st.write(f"当前输入：{name} · {category_label(top_category)} · {condition_labels_text(int(condition))} · {'卖家承担运费' if shipping else '买家承担运费'}")
                if estimate_result["category_median"] is not None:
                    comparison = "高于" if estimate_result["predicted_price"] > estimate_result["category_median"] else "低于"
                    st.write(f"模型参考价{comparison}该大类测试数据的中位价，差额约 ¥{abs(estimate_result['predicted_price'] - estimate_result['category_median']):,.0f}。")
                st.caption("同类中位价用于解释整体价格水平，不是第二个预测模型。")
            with st.expander("交互实验：修改特征，观察价格变化"):
                st.caption("保持其他信息不变，只改变成色或运费，观察模型预测如何变化。")
                what_if_condition = st.select_slider(
                    "实验成色",
                    options=[1, 2, 3, 4, 5],
                    value=int(condition),
                    format_func=condition_labels_text,
                    key="what_if_condition",
                )
                what_if_shipping = st.checkbox(
                    "实验中由卖家承担运费",
                    value=bool(shipping),
                    key="what_if_shipping",
                )
                what_if_result = calculate_reference_estimate(
                    model, data, metrics, name, description, top_category, second_category,
                    third_category, brand, int(what_if_condition), what_if_shipping, float(quoted_price)
                )
                change = float(what_if_result["predicted_price"]) - float(estimate_result["predicted_price"])
                change_text = f"上涨 ¥{change:,.2f}" if change >= 0 else f"下降 ¥{abs(change):,.2f}"
                w1, w2 = st.columns(2)
                w1.metric("实验后参考价", f"¥{what_if_result['predicted_price']:,.2f}")
                w2.metric("相对原结果", change_text)

    with browse_tab:
        st.subheader("商品浏览")
        st.caption(
            f"浏览美国二手交易平台商品记录；价格按 1 美元≈¥{float(metrics.get('usd_to_cny', 6.6702)):.2f} 换算，"
            "卡片图片是匹配的类别示例，不是该条记录的实拍图。"
        )
        browse_query = st.text_input("搜索商品名称或描述", key="browse_query", placeholder="例如：蓝牙耳机、背包、运动鞋")
        browse_controls = st.columns(2, gap="medium")
        browse_categories = ["校园常见", "不限制"] + sorted(data["cat1"].dropna().unique().tolist())
        browse_category = browse_controls[0].selectbox(
            "商品大类",
            browse_categories,
            format_func=lambda value: value if value in {"校园常见", "不限制"} else category_label(value),
            key="browse_category",
        )
        browse_sort = browse_controls[1].selectbox(
            "排序方式", ["相似度最高", "价格从低到高", "价格从高到低"], key="browse_sort"
        )
        price_controls = st.columns(2, gap="medium")
        browse_min_price = price_controls[0].number_input(
            "最低价格（元）", min_value=0.0, value=0.0, step=10.0, key="browse_min_price"
        )
        browse_max_price = price_controls[1].number_input(
            "最高价格（元）", min_value=0.0, value=2000.0, step=50.0, key="browse_max_price"
        )
        browse_condition_label = st.selectbox(
            "成色筛选", ["全部成色", "全新", "较新", "一般", "较旧", "明显使用痕迹"], key="browse_condition"
        )
        only_with_images = st.checkbox(
            "只显示有类别示例图的商品",
            value=True,
            help="图片是类别示例图，不是该条交易记录的实拍图。",
        )
        condition_lookup = {"全新": 1, "较新": 2, "一般": 3, "较旧": 4, "明显使用痕迹": 5}
        browse_condition = condition_lookup.get(browse_condition_label)
        if browse_min_price > browse_max_price:
            st.warning("最低价格不能高于最高价格。")
            browse_items = data.iloc[0:0].copy()
        else:
            browse_items = browse_results(
                browse_query, browse_category, browse_sort, data,
                vectorizer, matrix, title_vectorizer, title_matrix,
                price_min=browse_min_price,
                price_max=browse_max_price,
                condition_filter=browse_condition,
                only_with_images=only_with_images,
            )
        detail_value = st.query_params.get("detail")
        try:
            selected_index = int(detail_value) if detail_value is not None else None
        except ValueError:
            selected_index = None
        if selected_index is not None and selected_index in data.index:
            item = data.loc[selected_index]
            back_col, _ = st.columns([1, 5])
            if back_col.button("返回商品列表", key="browse_back"):
                if "detail" in st.query_params:
                    del st.query_params["detail"]
                st.rerun()
            st.divider()
            detail_title = display_product_name(item)
            st.subheader(detail_title or "未提供商品名称")
            detail_left, detail_right = st.columns([1, 1.5], gap="large")
            image_category = browse_image_label(item)
            detail_examples = curated_examples_for_item(item, curated_images)
            if detail_examples:
                example_index = int(selected_index) % len(detail_examples)
                detail_left.image(detail_examples[example_index]["image"], use_column_width=True)
                detail_left.caption(f"{image_category}类别示例图 · 与当前商品记录不对应")
            else:
                detail_left.info("暂无对应类别示例图")
            with detail_right:
                st.metric("记录价格", f"¥{float(item['price_cny']):,.2f}")
                st.write(f"**商品大类：** {category_label(str(item['cat1']))}")
                st.write(f"**商品小类：** {category_label(str(item['cat2']))}")
                st.write(f"**商品细分类：** {category_label(str(item['cat3']))}")
                brand_value = str(item["brand_name"])
                st.write(f"**品牌：** {brand_display_name(brand_value)}")
                condition_labels = {1: "全新", 2: "较新", 3: "一般", 4: "较旧", 5: "明显使用痕迹"}
                condition_value = int(item["item_condition_id"])
                st.write(f"**成色等级：** {condition_labels.get(condition_value, '未提供')}")
                st.write(f"**运费：** {'卖家承担' if int(item['shipping']) else '买家承担'}")
                raw_description = str(item["item_description"])
                if raw_description not in {"Unknown", "nan"}:
                    st.markdown("**商品描述（中文翻译）**")
                    try:
                        translated_description = translate_listing_description(raw_description)
                        st.write(translated_description or "未提供描述")
                    except Exception:
                        st.write(sanitize_chinese_display(raw_description, "商品描述暂未提供"))
                    st.caption("页面描述仅供浏览参考，价格模型仍使用原始数据特征。")
                else:
                    st.markdown("**商品描述**")
                    st.write("未提供描述")
            st.caption("数据来自美国二手交易平台，价格按参考汇率换算为人民币，仅用于课程演示，不代表校园真实成交价。")
        else:
            # Re-check against the current manifest so stale Streamlit data caches
            # cannot put a no-image record back into the image-first grid.
            if only_with_images:
                browse_items = browse_items[
                    browse_items.apply(lambda row: bool(curated_examples_for_item(row, curated_images)), axis=1)
                ]
            if browse_items.empty:
                st.warning("没有找到符合条件的商品，试试更短或更常见的关键词。")
            else:
                st.caption(f"显示 {len(browse_items)} 条商品记录。点击卡片可查看商品细节。")
                for start in range(0, len(browse_items), 5):
                    columns = st.columns(5, gap="small")
                    for column, (item_index, item) in zip(columns, browse_items.iloc[start:start + 5].iterrows()):
                        with column:
                            with st.container(border=True):
                                image_category = browse_image_label(item)
                                examples = curated_examples_for_item(item, curated_images)
                                if examples:
                                    image = examples[int(item_index) % len(examples)]
                                    render_clickable_image(image["image"], int(item_index), source="browse")
                                    st.caption(f"{product_type_label(image['product_type'])}示例图")
                                else:
                                    st.info("暂无匹配的类别示例图")
                                st.markdown(f"**{display_product_name(item)}**")
                                st.markdown(f"### ¥{float(item['price_cny']):,.2f}")
                                st.caption(f"{category_label(str(item['cat1']))} · 成色 {int(item['item_condition_id'])}")
                                if st.button("查看详情", key=f"browse_detail_{item_index}", use_container_width=True):
                                    st.session_state["selected_detail_index"] = int(item_index)
                                    st.session_state["detail_source"] = "browse"
                                    st.rerun()

    with search_tab:
        st.subheader("相似商品")
        st.caption("选一件商品作参照，查看同类商品、文本相似度和价格差。也可以从任意商品详情页发起对比。")
        reference_options = [index for index in [5815, 16040, 19801, 67208, 45293] if index in data.index]
        reference_index = st.selectbox(
            "参照商品",
            reference_options,
            format_func=lambda index: f"{display_product_name(data.loc[index])} · ¥{float(data.loc[index, 'price_cny']):,.2f}",
        )
        reference_item = data.loc[reference_index]
        reference_image = curated_examples_for_item(reference_item, curated_images, str(reference_item["name"]))
        reference_left, reference_right = st.columns([1, 3], gap="medium")
        with reference_left:
            if reference_image:
                st.image(reference_image[0]["image"], width=180)
                st.caption("类别示例图 · 非商品实拍图")
        with reference_right:
            st.markdown(f"**{display_product_name(reference_item)}**")
            st.metric("参照价格", f"¥{float(reference_item['price_cny']):,.2f}")
            st.caption(f"{category_label(str(reference_item['cat1']))} · {category_label(str(reference_item['cat2']))} · 成色 {condition_labels_text(int(reference_item['item_condition_id']))}")
        st.markdown("#### 可比较的同类商品")
        recommendations = recommend_for_item(reference_index, data, matrix, title_matrix)
        if recommendations.empty:
            st.info("暂时没有找到足够相近的同类商品。")
        else:
            render_recommendation_cards(reference_index, recommendations, data, curated_images)
        st.caption("优先在同一细分类中推荐；标题权重 75%，标题加描述权重 25%。图片仅为类别示例，不参与匹配。")

    with gallery_tab:
        st.subheader("校园商品图片示例")
        st.caption("图片按公开商品图片数据集的官方类型整理，仅作为校园商品类别示意，不与交易记录对应，也不代表二手成色。")
        available_categories = [name for name in curated_images if curated_images[name]]
        if not available_categories:
            st.info("未找到已整理图片，请先运行图片整理脚本生成校园展示素材。")
        else:
            selected_category = st.selectbox("商品类别", available_categories, key="gallery_category")
            category_images = curated_images[selected_category]
            page_size = 8
            page_count = max(1, (len(category_images) + page_size - 1) // page_size)
            page = st.selectbox("查看页码", list(range(1, page_count + 1)), format_func=lambda value: f"第 {value} 页 / 共 {page_count} 页", key="gallery_page")
            st.caption(f"{selected_category} · 共 {len(category_images)} 张")
            page_images = category_images[(page - 1) * page_size:page * page_size]
            for start in range(0, len(page_images), 5):
                columns = st.columns(5, gap="small")
                for column, example in zip(columns, page_images[start:start + 5]):
                    with column:
                        with st.container(border=True):
                            st.image(example["image"], use_column_width=True)
                            st.caption(product_type_label(example["product_type"]))

    with model_tab:
        st.subheader("模型实验")
        st.caption("这里展示从数据、特征工程到预测结果的完整机器学习流程。")
        st.markdown("**模型流程**")
        st.code(
            "商品名称、描述\n"
            "    ↓\n"
            "TF-IDF 文本特征 + 类别/品牌独热编码 + 成色/运费数值特征\n"
            "    ↓\n"
            "岭回归（在训练集内部验证正则化强度）\n"
            "    ↓\n"
            "人民币价格预测 + 残差分位数价格区间",
            language="text",
        )
        a, b, c, d = st.columns(4)
        a.metric("训练样本", f"{metrics['rows_train']:,}")
        b.metric("测试样本", f"{metrics['rows_test']:,}")
        c.metric("平均绝对误差", f"¥{metrics['mae_cny']:.2f}")
        d.metric("决定系数 R²", f"{metrics['r2_cny']:.3f}")
        st.write(f"最终模型：文本特征 + 类别/品牌编码 + 数值特征 + 岭回归（正则化强度 α={metrics.get('ridge_alpha', 3.0)}）")
        st.markdown("#### 同一测试集上的模型对比")
        if COMPARISON_PATH.exists():
            comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
            model_names = {
                "category_median": "按商品大类取中位价",
                "random_forest": "随机森林",
                "xgboost": "XGBoost",
                "ridge": "岭回归",
            }
            ordered_models = sorted(
                comparison["models"].items(), key=lambda entry: entry[1]["mae_cny"]
            )
            comparison_frame = pd.DataFrame([
                {
                    "方法": model_names[name],
                    "用途": "当前在线估价" if name == "ridge" else "离线对比",
                    "平均绝对误差": f"¥{values['mae_cny']:.2f}",
                    "均方根误差": f"¥{values['rmse_cny']:.2f}",
                    "决定系数 R²": f"{values['r2_cny']:.3f}",
                }
                for name, values in ordered_models
            ])
            st.dataframe(comparison_frame, hide_index=True, use_container_width=True)
            chart_data = pd.DataFrame([
                {
                    "方法": "类别中位价" if name == "category_median" else model_names[name],
                    "平均绝对误差（元）": values["mae_cny"],
                    "用途": "当前在线估价" if name == "ridge" else "离线对比",
                }
                for name, values in ordered_models
            ])
            chart_base = alt.Chart(chart_data).encode(
                x=alt.X(
                    "平均绝对误差（元）:Q", title="平均绝对误差（元）",
                    scale=alt.Scale(domain=[0, chart_data["平均绝对误差（元）"].max() * 1.07]),
                ),
                y=alt.Y("方法:N", sort=chart_data["方法"].tolist(), title=None),
                tooltip=["方法:N", alt.Tooltip("平均绝对误差（元）:Q", format=".2f")],
            )
            bars = chart_base.mark_bar(size=24).encode(
                color=alt.condition(
                    alt.datum.用途 == "当前在线估价", alt.value("#2d806b"), alt.value("#9eafa7")
                )
            )
            labels = chart_base.mark_text(align="right", dx=-7).encode(
                text=alt.Text("平均绝对误差（元）:Q", format=".2f"),
                color=alt.condition(
                    alt.datum.用途 == "当前在线估价", alt.value("#ffffff"), alt.value("#26352f")
                ),
            )
            st.altair_chart((bars + labels).properties(height=220), use_container_width=True)
            st.caption(
                "四种方法使用相同的 8 万条训练记录和 2 万条测试记录；三个学习模型使用相同的文本、类别和数值特征，"
                "均在还原为人民币价格后计算误差。误差越低越好。"
            )
            st.info(
                "当前实验中岭回归误差最低，XGBoost 和随机森林没有超过它。"
                "树模型使用固定演示参数，尚未系统调参；这不能证明岭回归在所有设置下都最好。"
            )
        else:
            baseline = baseline_evaluation(data)
            comparison_frame = pd.DataFrame([
                {"方法": "按商品大类取中位价", "平均绝对误差": f"¥{baseline['mae_cny']:.2f}"},
                {"方法": "岭回归", "平均绝对误差": f"¥{metrics['mae_cny']:.2f}"},
            ])
            st.dataframe(comparison_frame, hide_index=True, use_container_width=True)
            st.caption("运行 compare_price_models.py 后可展示随机森林和 XGBoost 的离线对比。")
        scores = metrics.get("alpha_validation_scores", {})
        if scores:
            score_frame = pd.DataFrame(
                [(f"α={alpha}", value) for alpha, value in scores.items()],
                columns=["正则化强度", "验证集平均绝对误差（元）"],
            )
            st.caption("候选正则化强度在训练集内部验证，误差越低越好。")
            st.dataframe(score_frame, hide_index=True, use_container_width=True)
        band_metrics = metrics.get("price_band_metrics", {})
        if band_metrics:
            st.markdown("#### 不同实际价格段的误差")
            band_frame = pd.DataFrame(
                [
                    {
                        "实际价格段": f"¥{label.replace('-', ' - ¥').replace('+', ' 以上')}",
                        "样本数": values["count"],
                        "平均绝对误差": f"¥{values['mae_cny']:.2f}",
                        "中位相对误差": f"{values['median_relative_error_pct']:.1f}%",
                    }
                    for label, values in band_metrics.items()
                ]
            )
            st.dataframe(band_frame, hide_index=True, use_container_width=True)
            st.caption("低价商品的相对误差更能反映实际体验；价格范围按模型验证集对应价格段的绝对误差估算。")
        counts = data["cat1"].value_counts().rename_axis("类别").reset_index(name="商品数")
        counts["类别"] = counts["类别"].map(category_label)
        st.bar_chart(counts.set_index("类别"))

    with publish_tab:
        render_publish_tab(user)

    with market_tab:
        render_user_market_tab(user)

    with account_tab:
        render_account_tab(user, data, model, metrics)


if __name__ == "__main__":
    main()
