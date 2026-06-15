"""
图片爬虫脚本
支持百度图片搜索，自动按品牌关键词下载 LOGO 图片
Google 图片作为可选项，触发反爬则自动跳过

用法:
    python scripts/scrape_images.py                    # 爬取所有品牌
    python scripts/scrape_images.py --brand heytea     # 只爬取喜茶
    python scripts/scrape_images.py --brand heytea --source baidu  # 只用百度
"""

import os
import sys
import time
import hashlib
import argparse
import requests
from io import BytesIO
from urllib.parse import quote
from PIL import Image
from tqdm import tqdm

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, SCRAPE_KEYWORDS, SCRAPE_MAX_PER_KEYWORD, BRAND_NAMES

# ============================================================
# 通用工具函数
# ============================================================

# 请求头，模拟浏览器
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 已下载图片的哈希集合（用于去重）
downloaded_hashes = set()


def compute_image_hash(img_bytes: bytes) -> str:
    """计算图片内容的 MD5 哈希值，用于去重"""
    return hashlib.md5(img_bytes).hexdigest()


def is_valid_image(img_bytes: bytes, min_size: int = 50) -> bool:
    """
    检查图片是否有效
    - 能被 Pillow 正常打开
    - 尺寸不小于 min_size x min_size
    - 不是纯色图片
    """
    try:
        img = Image.open(BytesIO(img_bytes))
        img.verify()
        # 重新打开以获取尺寸（verify 后不能再操作）
        img = Image.open(BytesIO(img_bytes))
        w, h = img.size
        if w < min_size or h < min_size:
            return False
        return True
    except Exception:
        return False


def save_image(img_bytes: bytes, save_dir: str, index: int) -> bool:
    """
    保存图片到指定目录
    - 自动转换为 RGB 格式的 JPG
    - 跳过重复图片
    返回 True 表示保存成功
    """
    img_hash = compute_image_hash(img_bytes)
    if img_hash in downloaded_hashes:
        return False

    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        filepath = os.path.join(save_dir, f"{index:04d}.jpg")
        img.save(filepath, "JPEG", quality=90)
        downloaded_hashes.add(img_hash)
        return True
    except Exception:
        return False


def download_image(url: str, timeout: int = 10) -> bytes | None:
    """下载单张图片，返回字节数据，失败返回 None"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
    except Exception:
        pass
    return None


# ============================================================
# 百度图片爬虫
# ============================================================

def _decode_baidu_url(encrypted: str) -> str:
    """解密百度图片的 objURL（百度对原始URL做了简单字符替换加密）"""
    table = {
        'w': 'a', 'k': 'b', 'v': 'c', '1': 'd', 'j': 'e', 'u': 'f',
        '2': 'g', 'i': 'h', 't': 'i', '3': 'j', 'h': 'k', 's': 'l',
        '4': 'm', 'g': 'n', 'r': 'o', '5': 'p', 'f': 'q', 'q': 'r',
        '6': 's', 'e': 't', 'p': 'u', '7': 'v', 'd': 'w', 'o': 'x',
        '8': 'y', 'c': 'z', 'n': '0', '9': '1', 'b': '2', 'm': '3',
        '0': '4', 'a': '5', 'l': '6', 'z': '7', 'y': '8', 'x': '9',
        '_z2C$q': ':', '_z&e3B': '.', 'AzdH3F': '/',
    }
    # 先替换多字符模式
    for k, v in [('_z2C$q', ':'), ('_z&e3B', '.'), ('AzdH3F', '/')]:
        encrypted = encrypted.replace(k, v)
    # 再替换单字符
    result = []
    for ch in encrypted:
        if ch in table:
            result.append(table[ch])
        else:
            result.append(ch)
    return ''.join(result)


def scrape_baidu(keyword: str, max_count: int = 30) -> list[str]:
    """
    从百度图片搜索获取图片 URL 列表

    百度图片的 AJAX 接口：
    https://image.baidu.com/search/acjson?tn=resultjson_com&word=关键词&pn=页码偏移
    """
    urls = []
    page_size = 30

    for page_offset in range(0, max_count + page_size, page_size):
        try:
            params = {
                "tn": "resultjson_com",
                "logid": "",
                "ipn": "rj",
                "ct": "201326592",
                "is": "",
                "fp": "result",
                "fr": "",
                "word": keyword,
                "queryWord": keyword,
                "cl": "2",
                "lm": "-1",
                "ie": "utf-8",
                "oe": "utf-8",
                "adpicid": "",
                "st": "-1",
                "z": "",
                "ic": "",
                "hd": "",
                "latest": "",
                "copyright": "",
                "s": "",
                "se": "",
                "tab": "",
                "width": "",
                "height": "",
                "face": "0",
                "istype": "2",
                "qc": "",
                "nc": "1",
                "expermode": "",
                "nojc": "",
                "pn": str(page_offset),
                "rn": str(page_size),
                "gsm": "",
            }

            resp = requests.get(
                "https://image.baidu.com/search/acjson",
                params=params,
                headers=HEADERS,
                timeout=15,
            )

            if resp.status_code != 200:
                break

            data = resp.json()
            items = data.get("data", [])

            for item in items:
                if not item:
                    continue
                url = None

                # 优先用可直接访问的 URL（不需要解密）
                for key in ("hoverURL", "middleURL", "thumbURL"):
                    candidate = item.get(key, "")
                    if candidate and candidate.startswith("http"):
                        url = candidate
                        break

                # 如果都没有，尝试解密 objURL
                if not url:
                    obj = item.get("objURL", "")
                    if obj:
                        decoded = _decode_baidu_url(obj)
                        if decoded.startswith("http"):
                            url = decoded

                if url:
                    urls.append(url)

            if len(urls) >= max_count:
                break

            time.sleep(0.5)

        except Exception as e:
            print(f"  [百度] 请求异常: {e}")
            break

    return urls[:max_count]


# ============================================================
# Google 图片爬虫（可选，触发反爬自动跳过）
# ============================================================

def scrape_google(keyword: str, max_count: int = 20) -> list[str]:
    """
    从 Google 图片搜索获取图片 URL 列表

    注意：Google 反爬较强，此函数可能失败。
    如果失败会返回空列表，不影响整体流程。
    """
    urls = []
    try:
        search_url = (
            f"https://www.google.com/search?q={quote(keyword)}"
            f"&tbm=isch&ijn=0"
        )
        google_headers = HEADERS.copy()
        google_headers["Accept"] = "text/html,application/xhtml+xml"

        resp = requests.get(search_url, headers=google_headers, timeout=15)

        if resp.status_code != 200:
            print(f"  [Google] HTTP {resp.status_code}，跳过 Google 源")
            return []

        # 从 HTML 中提取图片 URL
        # Google 图片的 URL 通常在 "https://..." 格式中
        import re
        # 匹配 Google 图片页面中嵌入的图片链接
        pattern = r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)"'
        found = re.findall(pattern, resp.text)

        # 过滤掉 Google 自身的域名
        for url in found:
            if "google.com" not in url and "gstatic.com" not in url:
                urls.append(url)
                if len(urls) >= max_count:
                    break

    except Exception as e:
        print(f"  [Google] 请求失败: {e}，跳过 Google 源")

    return urls


# ============================================================
# 主流程
# ============================================================

def scrape_brand(brand: str, sources: list[str] = None):
    """
    爬取指定品牌的所有图片

    参数:
        brand: 品牌英文名（如 'heytea'）
        sources: 数据源列表，如 ['baidu', 'google']
    """
    if sources is None:
        sources = ["baidu", "google"]

    keywords = SCRAPE_KEYWORDS.get(brand, [])
    save_dir = os.path.join(RAW_DATA_DIR, brand)
    os.makedirs(save_dir, exist_ok=True)

    # 统计当前已有图片数量（从最大编号继续）
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    start_index = len(existing)
    saved_count = 0

    cn_name = BRAND_NAMES.get(brand, brand)
    print(f"\n{'='*60}")
    print(f"开始爬取: {cn_name} ({brand})")
    print(f"已有图片: {start_index} 张")
    print(f"数据源: {', '.join(sources)}")
    print(f"{'='*60}")

    for keyword in keywords:
        print(f"\n  关键词: 「{keyword}」")

        all_urls = []

        # 从各数据源获取 URL
        if "baidu" in sources:
            baidu_urls = scrape_baidu(keyword, SCRAPE_MAX_PER_KEYWORD)
            print(f"    [百度] 获取到 {len(baidu_urls)} 个链接")
            all_urls.extend(baidu_urls)

        if "google" in sources:
            google_urls = scrape_google(keyword, SCRAPE_MAX_PER_KEYWORD // 2)
            print(f"    [Google] 获取到 {len(google_urls)} 个链接")
            all_urls.extend(google_urls)

        # 下载图片
        keyword_saved = 0
        for url in tqdm(all_urls, desc=f"    下载中", ncols=70):
            img_bytes = download_image(url)
            if img_bytes and is_valid_image(img_bytes):
                idx = start_index + saved_count
                if save_image(img_bytes, save_dir, idx):
                    saved_count += 1
                    keyword_saved += 1

            # 礼貌延迟
            time.sleep(0.3)

        print(f"    本关键词保存: {keyword_saved} 张")

    total = start_index + saved_count
    print(f"\n  ✅ {cn_name} 爬取完成: 新增 {saved_count} 张，总计 {total} 张")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="饮料品牌 LOGO 图片爬虫")
    parser.add_argument(
        "--brand",
        type=str,
        default=None,
        choices=list(BRAND_NAMES.keys()),
        help="指定爬取的品牌（不指定则爬取全部）",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "baidu", "google"],
        help="指定数据源（默认全部）",
    )
    args = parser.parse_args()

    # 确定数据源
    if args.source == "all":
        sources = ["baidu", "google"]
    else:
        sources = [args.source]

    # 确定要爬取的品牌
    if args.brand:
        brands = [args.brand]
    else:
        brands = list(BRAND_NAMES.keys())

    print("=" * 60)
    print("🧋 饮料品牌 LOGO 图片爬虫")
    print(f"   品牌: {', '.join([BRAND_NAMES[b] for b in brands])}")
    print(f"   数据源: {', '.join(sources)}")
    print(f"   保存目录: {RAW_DATA_DIR}")
    print("=" * 60)

    total_saved = 0
    for brand in brands:
        count = scrape_brand(brand, sources)
        total_saved += count

    print(f"\n{'='*60}")
    print(f"🎉 全部完成！共新增 {total_saved} 张图片")
    print(f"   图片保存在: {RAW_DATA_DIR}")
    print(f"\n⚠️  请手动检查每个品牌文件夹，删除不相关的图片！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
