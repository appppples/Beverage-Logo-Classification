"""
图片预筛理脚本
用于在人工筛选前，自动删除损坏、过小、比例极端或高度重复的图片
"""

import os
import sys
from PIL import Image
import imagehash
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, BRAND_NAMES


def clean_brand_folder(brand_dir: str, brand_name: str):
    """
    清理指定品牌文件夹下的图片
    """
    if not os.path.exists(brand_dir):
        return

    images = [f for f in os.listdir(brand_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    if not images:
        return

    print(f"\n🔍 正在清理: {brand_name} ({len(images)} 张图片)")

    deleted_corrupted = 0
    deleted_small = 0
    deleted_ratio = 0
    deleted_dupe = 0
    
    # 存储已保留图片的哈希值，用于查重
    seen_hashes = []
    
    # 遍历所有图片
    for img_name in tqdm(images, desc="扫描中", ncols=80):
        img_path = os.path.join(brand_dir, img_name)
        
        try:
            with Image.open(img_path) as img:
                # 1. 检查图片是否能正常加载
                img.verify()
            
            # 重新打开以获取属性和计算哈希
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                
                # 2. 检查分辨率（剔除长宽小于 100px 的极小图）
                if w < 100 or h < 100:
                    deleted_small += 1
                    os.remove(img_path)
                    continue
                
                # 3. 检查长宽比例（剔除非常细长的横幅或截图，比例超过 1:4 或 4:1）
                ratio = max(w / h, h / w)
                if ratio > 4.0:
                    deleted_ratio += 1
                    os.remove(img_path)
                    continue
                
                # 4. 感知哈希查重（pHash 可以识别出稍微缩放、裁剪或变色的重复图）
                #    Hamming 距离 <= 5 视为高度相似
                img_hash = imagehash.phash(img)
                
                is_duplicate = False
                for h_seen in seen_hashes:
                    if img_hash - h_seen <= 5:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    deleted_dupe += 1
                    os.remove(img_path)
                    continue
                    
                # 通过了所有测试，保留
                seen_hashes.append(img_hash)

        except Exception as e:
            # 文件损坏无法打开
            deleted_corrupted += 1
            if os.path.exists(img_path):
                os.remove(img_path)

    total_deleted = deleted_corrupted + deleted_small + deleted_ratio + deleted_dupe
    remaining = len(images) - total_deleted
    
    print(f"  ✅ 清理完成！剩余 {remaining} 张有效图片。")
    if total_deleted > 0:
        print(f"  🗑️ 共删除 {total_deleted} 张:")
        if deleted_corrupted > 0: print(f"     - 损坏/无效: {deleted_corrupted} 张")
        if deleted_small > 0: print(f"     - 尺寸过小(<100px): {deleted_small} 张")
        if deleted_ratio > 0: print(f"     - 比例极端(>4:1): {deleted_ratio} 张")
        if deleted_dupe > 0: print(f"     - 高度相似重复: {deleted_dupe} 张")


def main():
    print("=" * 60)
    print("🧹 原始数据自动化预筛理")
    print("   自动剔除：损坏图片、过小图片、极端比例长图、高度相似图")
    print("=" * 60)
    
    for brand, cn_name in BRAND_NAMES.items():
        brand_dir = os.path.join(RAW_DATA_DIR, brand)
        clean_brand_folder(brand_dir, cn_name)
        
    print("\n" + "=" * 60)
    print("🎉 自动化预筛理完成！")
    print("⚠️ 接下来请打开 data/raw/ 下的各个文件夹，以【大图标】视图进行最终的人工核对。")
    print("   将无关的风景、人像、无关文字截图删掉即可。")
    print("=" * 60)


if __name__ == "__main__":
    main()
