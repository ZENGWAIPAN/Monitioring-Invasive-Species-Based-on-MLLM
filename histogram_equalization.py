import cv2
import numpy as np


def correct_underwater_image(image_path):
    """
    根據使用者描述的演算法，校正水下圖像的顏色。

    演算法步驟：
    1. 讀取圖像並分離 B, G, R 通道。
    2. 計算每個通道的顏色均值。
    3. 找出均值最大的顏色通道。
    4. 計算校正係數 k（最大均值 / 較小均值）。
    5. 將 k 值乘到對應的顏色通道上。
    6. 將像素值裁剪到 [0, 255] 範圍內。
    7. 合併通道，生成最終圖像。

    :param image_path: 輸入圖像的路徑
    :return: 經過顏色校正的圖像
    """
    # 1. 讀取圖像
    # cv2.imread 讀取進來的顏色通道順序是 BGR (藍, 綠, 紅)
    img = cv2.imread(image_path)

    if img is None:
        print(f"錯誤：無法讀取圖像 '{image_path}'")
        return None

    # 將圖像像素值轉換為浮點數，以便進行精確的除法和乘法計算
    img_float = img.astype(np.float32)

    # 2. 分離通道並計算每個通道的均值
    b, g, r = cv2.split(img_float)
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)

    means = [mean_b, mean_g, mean_r]
    channel_names = ['Blue', 'Green', 'Red']

    # 3. 找出均值最大的顏色通道
    max_mean_index = np.argmax(means)
    max_mean_value = means[max_mean_index]

    print(f"各顏色通道的均值 -> 藍: {mean_b:.2f}, 綠: {mean_g:.2f}, 紅: {mean_r:.2f}")
    print(f"均值最大的顏色是: {channel_names[max_mean_index]} (均值: {max_mean_value:.2f})")

    # 4. & 5. 計算 k 值並對另外兩個通道進行校正
    # 根據哪個通道是最大的，來校正另外兩個
    if max_mean_index == 0:  # 藍色是最大的
        k_g = max_mean_value / mean_g
        k_r = max_mean_value / mean_r
        g = g * k_g
        r = r * k_r
        print(f"校正係數 k -> k_green: {k_g:.2f}, k_red: {k_r:.2f}")
    elif max_mean_index == 1:  # 綠色是最大的
        k_b = max_mean_value / mean_b
        k_r = max_mean_value / mean_r
        b = b * k_b
        r = r * k_r
        print(f"校正係數 k -> k_blue: {k_b:.2f}, k_red: {k_r:.2f}")
    else:  # 紅色是最大的 (在水下照片中較少見)
        k_b = max_mean_value / mean_b
        k_g = max_mean_value / mean_g
        b = b * k_b
        g = g * k_g
        print(f"校正係數 k -> k_blue: {k_b:.2f}, k_green: {k_g:.2f}")

    # 6. 將校正後的像素值裁剪到 [0, 255] 的有效範圍內
    # 乘法可能導致某些像素值超過 255，必須將其限制回來
    b_corrected = np.clip(b, 0, 255)
    g_corrected = np.clip(g, 0, 255)
    r_corrected = np.clip(r, 0, 255)

    # 7. 合併通道，並將圖像從浮點數轉換回 8位元整數 (影像的標準格式)
    corrected_img = cv2.merge([b_corrected, g_corrected, r_corrected]).astype(np.uint8)

    return corrected_img


# --- 主執行區 ---
if __name__ == '__main__':
    # 將 'underwater_image.jpg' 替換為你的照片檔案名
    input_filename = r"C:\Users\acer\Downloads\UIEB\811_img_.png"
    output_filename = r"C:\Users\acer\Downloads\corrected_image4.jpg"

    # 執行顏色校正
    corrected_image = correct_underwater_image(input_filename)

    if corrected_image is not None:
        # 讀取原始圖像以供顯示對比
        original_image = cv2.imread(input_filename)

        # 將原始圖像和校正後圖像並排顯示
        # 如果圖像太大，可以先縮小再顯示
        h, w, _ = original_image.shape
        scale = 600 / h  # 將高度縮放到600像素以便顯示

        original_display = cv2.resize(original_image, (int(w * scale), int(h * scale)))
        corrected_display = cv2.resize(corrected_image, (int(w * scale), int(h * scale)))

        comparison_image = np.hstack([original_display, corrected_display])

        # 顯示對比結果
        cv2.imshow('Original vs Corrected', comparison_image)

        # 將校正後的圖像保存到檔案
        cv2.imwrite(output_filename, corrected_image)
        print(f"\n校正後的圖像已保存為 '{output_filename}'")

        # 等待用戶按任意鍵關閉視窗
        cv2.waitKey(0)
        cv2.destroyAllWindows()
