import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, BRAND_NAMES

def unify_and_reorder():
    print("=" * 60)
    print("📂 数据集格式统一、重新排序与统计")
    print("=" * 60)
    
    total_images = 0
    counts = {}

    for brand in BRAND_NAMES.keys():
        brand_dir = os.path.join(RAW_DATA_DIR, brand)
        if not os.path.exists(brand_dir):
            counts[brand] = 0
            continue
            
        images = sorted([f for f in os.listdir(brand_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))])
        
        valid_images = []
        
        # 第一步：统一格式为 .jpg 并暂时重命名防止冲突
        for i, img_name in enumerate(images):
            old_path = os.path.join(brand_dir, img_name)
            temp_path = os.path.join(brand_dir, f"temp_{i:04d}.jpg")
            
            try:
                with Image.open(old_path) as img:
                    img = img.convert("RGB")
                    img.save(temp_path, "JPEG", quality=95)
                # 转换并保存成功后，删除原文件（如果不和新文件重名的话）
                if old_path != temp_path:
                    os.remove(old_path)
                valid_images.append(temp_path)
            except Exception as e:
                print(f"⚠️ 无法处理图片 {old_path}: {e}")
                # 如果处理失败，删除损坏文件
                if os.path.exists(old_path):
                    os.remove(old_path)
                    
        # 第二步：将临时名字正式改名为顺序编号
        temp_images = sorted([f for f in os.listdir(brand_dir) if f.startswith('temp_')])
        for i, temp_name in enumerate(temp_images):
            temp_path = os.path.join(brand_dir, temp_name)
            final_name = f"{i:04d}.jpg"
            final_path = os.path.join(brand_dir, final_name)
            os.rename(temp_path, final_path)
            
        final_count = len(temp_images)
        counts[brand] = final_count
        total_images += final_count
            
        print(f"✅ {BRAND_NAMES[brand]} ({brand}): 共有 {counts[brand]} 张")

    print("=" * 60)
    print(f"📊 总计: {total_images} 张纯正 JPG 图片")
    print("=" * 60)

if __name__ == "__main__":
    unify_and_reorder()
