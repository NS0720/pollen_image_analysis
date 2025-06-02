import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

def calculate_intensity_offset_refined(image_path, blur=True, debug=False):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"❌ 読み込み失敗: {image_path}")
        return None

    if blur:
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # 二値化 + 輪郭抽出
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠ 輪郭検出失敗: {image_path}")
        return None

    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    cx, cy, r = int(cx), int(cy), int(r)

    h, w = img_gray.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask_circle = dist_from_center <= 0.8 * r
    selected_pixels = img_gray[mask_circle]

    if selected_pixels.size == 0:
        print(f"⚠ 花粉中心領域が空: {image_path}")
        return None

    # 明暗しきい値（パーセンタイル）
    high_thresh = np.percentile(selected_pixels, 80)
    low_thresh = np.percentile(selected_pixels, 20)

    coords = np.column_stack(np.where(mask_circle))
    intensities = img_gray[mask_circle]

    high_coords = coords[intensities >= high_thresh]
    low_coords = coords[intensities <= low_thresh]
    combined_coords = np.vstack((high_coords, low_coords))

    if len(high_coords) == 0 or len(low_coords) == 0:
        print(f"⚠ 明るい/暗い領域抽出失敗: {image_path}")
        return None

    # 高輝度領域の重心
    gy_high, gx_high = high_coords.mean(axis=0)
    dist_high = np.linalg.norm([gx_high - cx, gy_high - cy])
    offset_ratio_high = dist_high / r

    # 低輝度領域の重心
    gy_low, gx_low = low_coords.mean(axis=0)
    dist_low = np.linalg.norm([gx_low - cx, gy_low - cy])
    offset_ratio_low = dist_low / r

    # 明暗結合領域の重心
    gy_comb, gx_comb = combined_coords.mean(axis=0)
    dist_comb = np.linalg.norm([gx_comb - cx, gy_comb - cy])
    offset_ratio_comb = dist_comb / r

    if debug:
        print(f"中心: ({cx}, {cy})")
        print(f"[高] 重心: ({gx_high:.1f}, {gy_high:.1f})  距離: {dist_high:.2f} 比率: {offset_ratio_high:.3f}")
        print(f"[低] 重心: ({gx_low:.1f}, {gy_low:.1f})  距離: {dist_low:.2f} 比率: {offset_ratio_low:.3f}")
        print(f"[結合] 重心: ({gx_comb:.1f}, {gy_comb:.1f})  距離: {dist_comb:.2f} 比率: {offset_ratio_comb:.3f}")

        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(img_color, (cx, cy), int(r), (255, 0, 0), 1)
        cv2.circle(img_color, (int(gx_high), int(gy_high)), 3, (0, 0, 255), -1)  # 赤: 高輝度
        cv2.circle(img_color, (int(gx_low), int(gy_low)), 3, (255, 255, 0), -1)  # 水色: 低輝度
        cv2.circle(img_color, (int(gx_comb), int(gy_comb)), 3, (0, 255, 0), -1)  # 緑: 結合

        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title(f"Offset Ratio (Combined): {offset_ratio_comb:.3f}")
        plt.show()

    return (
        cx, cy,
        gx_high, gy_high, dist_high, offset_ratio_high,
        gx_low, gy_low, dist_low, offset_ratio_low,
        gx_comb, gy_comb, dist_comb, offset_ratio_comb
    )


# 元CSVファイルの読み込み
df = pd.read_csv(r"D:\python_working\pollen_statistics\spike_compare_blur_de80_stp20.csv")

# cspike_score を格納するリスト
center_pattern = []

# 画像のあるフォルダ（trainとvalを内包している親フォルダ）
base_image_dir = r"D:\python_working\pollen_statistics\512_sort_dataset" # ←必要に応じてパスを変更

cx_list, cy_list, gx_high_list, gy_high_list,dist_high_list,offset_ratio_high_list, gx_low_list, gy_low_list,dist_low_list,offset_ratio_low_list,gx_comb_list, gy_comb_list, dist_comb_list, offset_ratio_comb_list = [], [], [], [], [], [], [], [], [], [], [],[],[],[]

# tqdmでループを可視化
for _, row in df.iterrows():
    split = row['split'].strip()
    filename = row['filename'].strip()
    pollen_type = row['type'].strip()
    image_path = os.path.join(base_image_dir, split, pollen_type, filename)

    if not os.path.exists(image_path):
        print(f"❗パスが存在しません: {image_path}")
        cx, cy,gx_high, gy_high, dist_high, offset_ratio_high,gx_low, gy_low, dist_low, offset_ratio_low,gx_comb, gy_comb, dist_comb, offset_ratio_comb = [None]*14
    else:
        result = calculate_intensity_offset_refined(image_path, debug=False)
        if result is None:
            print(f"⚠ 処理失敗: {filename}")
            cx, cy,gx_high, gy_high, dist_high, dist_high,offset_ratio_high,gx_low, gy_low, dist_low, offset_ratio_low,gx_comb, gy_comb, dist_comb, offset_ratio_comb = [None]*14
        else:
            cx, cy,gx_high, gy_high, dist_high, offset_ratio_high,gx_low, gy_low, dist_low, offset_ratio_low,gx_comb, gy_comb, dist_comb, offset_ratio_comb = result

    cx_list.append(cx)
    cy_list.append(cy)
    gx_high_list.append(gx_high)
    gy_high_list.append(gy_high)
    dist_high_list.append(dist_high)
    offset_ratio_high_list.append(dist_high)
    gx_low_list.append(gx_low)
    gy_low_list.append(gy_low)
    dist_low_list.append(dist_low)
    offset_ratio_low_list.append(offset_ratio_low)
    gx_comb_list.append(gx_comb)
    gy_comb_list.append(gy_comb)
    dist_comb_list.append(dist_comb)
    offset_ratio_comb_list.append(offset_ratio_comb)


# カラム追加

df['cx'] = cx_list
df['cy'] = cy_list
df['gx_high'] = gx_high_list
df['gy_high'] = gy_high_list
df['dist_high'] = dist_high_list
df['offset_ratio_high'] = offset_ratio_high_list
df['gx_low'] = gx_low_list
df['gy_low'] = gy_low_list
df['dist_low'] = dist_low_list
df['offset_ratio_low'] = offset_ratio_low_list
df['gx_comb'] = gx_comb_list
df['gy_comb'] = gy_comb_list
df['dist_comb'] = dist_comb_list
df['offset_ratio_comb'] = offset_ratio_comb_list

# 新しいCSVとして保存
df.to_csv("radius_features_with_center_pattern_004.csv", index=False)
print(" 新しいCSVファイルを保存しました: radius_features_with_center_pattern_004.csv")

