import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
from  datetime import datetime

# 512x512で中心を揃えて切り出す関数
def crop_centered_square(image, center_x, center_y, size=512):
    """
    round_detect_400_dbscan.pyで花粉をひとつづ粗削りで切り出されたはずです。
    この画像を視認して花粉のtypeをcsvファイルに追加してcsvファイルを完成させてください。

    csvファイルには花粉の中心座標x,yと半径rと視認して手入力した花粉のtypeが入ったはずです。

    csvファイルからデータを読み取り、機械学習に適した大きさの画像を元画像から改めてきれいに切り出します。
    """
    h, w = image.shape[:2]
    half_size = size // 2

    x_start = center_x - half_size
    y_start = center_y - half_size
    x_end = center_x + half_size
    y_end = center_y + half_size

    # 空の黒画像を用意
    result = np.zeros((size, size, 3), dtype=np.uint8)

    # コピー元の範囲を計算
    src_x_start = max(0, x_start)
    src_y_start = max(0, y_start)
    src_x_end = min(w, x_end)
    src_y_end = min(h, y_end)

    # コピー先の範囲を計算
    dst_x_start = src_x_start - x_start if x_start < 0 else 0
    dst_y_start = src_y_start - y_start if y_start < 0 else 0

    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    # コピー実行
    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]
    return result

# メイン処理
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    csv_path = Path(r"D:\python_working\round_detect2025\output_20250427_220717667138\sedar_400_Azumi01_20250324_circle_data.csv") # 作成されたcsvファイルのパスに変更すること
    original_image_folder = Path(r"D:\python_working\round_detect2025\pollen_picture\sedar_400_Azumi01_20250324") # 元画像が入っているフォルダ
    output_folder = Path(r"D:/python_working/round_detect2025/pollen_picture/cropped_centered_512_Azumi01_20250324")  # 出力フォルダ

    folder_name = original_image_folder.name
    suffix_str = "_".join(folder_name.split("_")[-2:])
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    save_name_list = []

    for idx, row in df.iterrows():
        original_filename = row["original_filename"]
        x = int(row["x"])
        y = int(row["y"])
        type_label = row["type"]

        image_path = os.path.join(original_image_folder, original_filename)

        if not os.path.exists(image_path):
            print(f"画像が見つかりません: {image_path}")
            continue

        image = cv2.imread(image_path)
        cropped = crop_centered_square(image, x, y, size=512)

        save_name = f"{type_label}_{suffix_str}_centered_{idx:04d}.jpg" 
        save_name_list.append(save_name)
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, cropped)
        print(f"保存しました: {save_path}")
    
    df["saved_filename"] = save_name_list

    csv_output_path = output_folder / f"centerd_output_log_{suffix_str}.csv"
    df.to_csv(csv_output_path, index= False, encoding= "utf-8-sig")

if __name__ == "__main__":
    main()
