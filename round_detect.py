import cv2
import os
from  datetime import datetime
import numpy as np
from sklearn.cluster import DBSCAN
import csv

#400倍の高倍率での花粉検出コードです。

def round_detect(image):
    """
    プレパラートの画像データから丸いもの（花粉）を検出する関数

    Returns:
        Hogh変換後のnumpy行列になった円情報
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray Image',gray)
    #cv2.imwrite(f"gray.jpg",gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # ノイズの除去
    #　ノイズは（13，13）にすると具合よくノイズ除去してくれる。この部分は奇数のペアで入れてみて調節すること。
    blur = cv2.GaussianBlur(gray, (13, 13), 0)

    # 円の検出
    #ハフ変換したあとはcirclesにNumpy配列で円の情報が返されている。
    #minDist 検出する円同士の最小距離、param1 Cannyエッジ検出の上限値、 param2　円判定のしきい値（小さくすると検出多くなる）
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                param1=50, param2=40, minRadius=50, maxRadius=350)
    
    if circles is None:
        return np.array([])
    circles = np.round(circles[0]).astype("int")
    
    return circles

def cut_square(detected_circles,image):
    print(detected_circles)
    cropped_images = []
    height, width = image.shape[:2]
    # 検出された円を描画
    for (x, y, r) in detected_circles:
        x,y,r = int(x),int(y),int(r)

        x_start = max(x-2*r,0)
        x_end   = min(x+2*r, width)
        y_start = max(y-2*r,0)
        y_end   = min(y+2*r, height)

        if x_end > x_start and y_end > y_start:
            cropped = image[y_start:y_end, x_start:x_end]
            if cropped.size > 0:
                cropped_images.append(cropped)
            else:
                print(f"スキップ：空画像 (x={x}, y={y}, r={r})")
        else:
            print(f"スキップ：不正な範囲 x:{x_start}-{x_end}, y:{y_start}-{y_end}")
            cropped = image[y_start:y_end, x_start:x_end]
        
    return cropped_images

def combine_duplicates_dbscan(circles, eps=100, min_samples=1):
    """
    DBSCANを使って重複円（同じ花粉）を1つにまとめる関数
    クラスター解析のテクニックを使っている

    Parameters:
        circles: ndarray, shape=(N, 3) [x, y, r]
        eps: float, 同一花粉とみなす中心座標間距離（ピクセル）
        min_samples: int, クラスタと認識する最小サンプル数（円数）

    Returns:
        unique_circles: list of [x, y, r] をまとめた円リスト
    """
    if circles is None or len(circles) == 0:
        return []

    # 型変換と丸め処理
    circles = np.round(circles).astype(int)
    print(circles)
    # 中心座標だけを取り出してクラスタリング
    coords = circles[:, :2]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    unique_circles = []

    for label in set(labels):
        #if label == -1:
        #    continue  # ノイズ除外
        group = circles[labels == label]
        avg_x = int(np.mean(group[:, 0]))
        avg_y = int(np.mean(group[:, 1]))
        avg_r = int(np.mean(group[:, 2]))
        unique_circles.append([avg_x, avg_y, avg_r])

    return np.array(unique_circles)

# 切り出し画像を保存
def Image_Strage(image_stories,output_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    for i, croppedimage in enumerate(image_stories):
        filename = f'cropped_image_{timestamp}_{i}.jpg'
        filepath = os.path.join(output_folder, filename)
        if croppedimage.size > 0:
            cv2.imwrite(filepath, croppedimage) 
            print(f"保存: {filepath}")   
        else:
            print(f"スキップ（空画像）：{i}")
        

# 切り出し画像を保存　＆　円の(x,y,r)保存
def save_cropped_and_log_csv(image_stories, original_filename, detected_circles, output_folder, csv_writer):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    for i,(cropped_image, (x,y,r))in enumerate(zip(image_stories, detected_circles)):
        cropped_filename = f'cropped_image_{timestamp}_{i}.jpg'
        filepath = os.path.join(output_folder, cropped_filename)
        if cropped_image.size > 0:
            cv2.imwrite(filepath, cropped_image)
            print(f"保存: {filepath}")
            csv_writer.writerow([original_filename, int(x), int(y), int(r), cropped_filename])
        else:
            print(f"スキップ（空画像）： {i}")

def make_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    output_folder = f"output_{timestamp}"
    os.mkdir(output_folder)    

    return output_folder

def main():
    # 画像読み込み
    input_folder = "pollen_picture/sedar_400_Azumi_20250324_2"
    image_files = [ f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    folder_name = os.path.basename(input_folder)

    output_folder = make_folder()
    csv_path = os.path.join(output_folder, f"{folder_name}_circle_data.csv")

    with open(csv_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["original_filename", "x", "y", "r", "cropped_filename"])

        for file in image_files:
            filepath = os.path.join(input_folder, file)
            image = cv2.imread(filepath)
            

            circles = round_detect(image)
            if circles is None or len(circles) == 0:
                print("円が一つも検出されませんでした。")
            else:
                unique_circles = combine_duplicates_dbscan(circles)
                cropped_images = cut_square(unique_circles,image)

                #切り出して保存だけしたい場合
                #Image_Strage(cropped_images,output_folder)

                #切り出して保存＋csv記録もしたい場合
                save_cropped_and_log_csv(cropped_images, file, unique_circles, output_folder, csv_writer)

if __name__ == "__main__":
    main()


