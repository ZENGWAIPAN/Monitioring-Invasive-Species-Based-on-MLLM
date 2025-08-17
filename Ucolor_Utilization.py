import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os
import sys


# ==============================================================================
# 1. 暗通道先驗 (Dark Channel Prior) 算法
# 根據論文要求，Ucolor 需要一張介質傳輸圖作為輸入。
# 此處我們實現一個簡化的暗通道先驗算法來估算它。
# 參考 He, Kaiming, et al. "Single image haze removal using dark channel prior."
# ==============================================================================
def get_dark_channel(img, block_size=15):
    """
    計算影像的暗通道圖。
    暗通道的物理意義是：在絕大多數非天空的局部區域裡，某一些像素總會有至少一個顏色通道具有很低的值。

    Args:
        img (numpy.ndarray): 輸入的 BGR 格式影像，值範圍 [0, 255]。
        block_size (int): 用於計算局部最小值的區塊大小。

    Returns:
        numpy.ndarray: 暗通道圖。
    """
    b, g, r = cv2.split(img)
    min_channel = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size, block_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def get_atmospheric_light(img, dark_channel, p=0.001):
    """
    估算全球大氣光。
    我們從暗通道中取亮度最高的前 0.1% 的像素，然後在原圖中找到這些像素對應的亮度最高點作為大氣光。

    Args:
        img (numpy.ndarray): 輸入的 BGR 格式影像。
        dark_channel (numpy.ndarray): 對應的暗通道圖。
        p (float): 選取像素的百分比。

    Returns:
        tuple: BGR 格式的大氣光值。
    """
    img_size = img.shape[0] * img.shape[1]
    num_pixels = int(max(math.floor(img_size * p), 1))

    dark_vec = dark_channel.reshape(img_size)
    img_vec = img.reshape(img_size, 3)

    indices = dark_vec.argsort()
    indices = indices[img_size - num_pixels:]

    atmos_light_sum = np.zeros(3)
    for idx in indices:
        atmos_light_sum += img_vec[idx]

    atmos_light = np.mean([img_vec[idx] for idx in indices], axis=0)

    return atmos_light.astype(np.uint8)


def get_transmission_map(img, atmos_light, omega=0.95, block_size=15):
    """
    估算介質傳輸圖 (Transmission Map)。
    傳輸圖 t(x) 描述了光線從場景點到達相機過程中未被散射的比例。

    Args:
        img (numpy.ndarray): 輸入的 BGR 影像。
        atmos_light (tuple): 估算的大氣光值。
        omega (float): 去霧程度的保留係數，保留一部分霧來使影像更自然。
        block_size (int): 區塊大小。

    Returns:
        numpy.ndarray: 估算的傳輸圖。
    """
    img_normalized = img.astype('float') / atmos_light

    transmission = 1 - omega * get_dark_channel(img_normalized, block_size)

    # 進行導向濾波 (Guided Filter) 來平滑傳輸圖，此處為簡化省略，直接返回
    # 在實際應用中，為了獲得更好的邊緣保持效果，通常會使用導向濾波進行優化。
    # transmission = cv2.ximgproc.guidedFilter(img_gray, transmission, radius, eps)

    return transmission


# ==============================================================================
# 2. Ucolor 模型定義與載入
# 根據作者提供的 TensorFlow 程式碼結構，定義模型並載入預訓練權重。
# 由於無法直接執行作者的 TF 1.6 專案，此處為示意性程式碼。
# 實際執行時需要一個預先轉換好的模型檔案（如 .pb 或 .h5）。
# 為了讓此腳本能獨立運行，我們將模擬模型處理過程。
# ==============================================================================
class UcolorModel:
    """
    一個模擬的 Ucolor 模型類別。
    在實際應用中，這裡會載入真實的 TensorFlow/Keras 模型。
    """

    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                # 理想情況下，載入一個轉換好的 Keras/TF2 模型
                self.model = tf.keras.models.load_model(model_path)
                print("預訓練模型載入成功。")
            except Exception as e:
                print(f"模型載入失敗: {e}。將使用模擬處理。")
        else:
            print("未提供模型路徑或路徑不存在，將使用模擬增強效果。")

    def predict(self, original_image, transmission_map):
        """
        執行影像增強。

        Args:
            original_image (numpy.ndarray): BGR 格式的原圖，範圍 [0, 255]。
            transmission_map (numpy.ndarray): 介質傳輸圖，單通道，範圍 [0, 1]。

        Returns:
            numpy.ndarray: 增強後的 BGR 影像。
        """
        if self.model:
            # 實際模型的處理流程：
            # 1. 影像預處理 (縮放、歸一化、色彩空間轉換)
            # 2. 傳輸圖預處理 (逆轉 RMT = 1 - T)
            # 3. 將兩者輸入模型
            # 4. 後處理模型輸出
            # 此處省略複雜的預處理，僅示意
            input_img_tensor = self._preprocess(original_image)
            rmt_tensor = 1.0 - self._preprocess(transmission_map)  # RMT
            enhanced_tensor = self.model.predict([input_img_tensor, rmt_tensor])
            enhanced_image = self._postprocess(enhanced_tensor)
            return enhanced_image
        else:
            # 模擬增強過程 (基於簡化的水下影像模型逆運算)
            return self._simulate_enhancement(original_image, transmission_map)

    def _simulate_enhancement(self, img, tmap):
        """如果沒有模型，使用簡化算法模擬增強效果"""
        print("增強...")
        # 估算大氣光
        dark_channel = get_dark_channel(img)
        atmos_light = get_atmospheric_light(img, dark_channel)

        # 避免除以零
        tmap_refined = np.maximum(tmap, 0.1)

        # J(x) = (I(x) - A) / t(x) + A
        enhanced_img = np.zeros_like(img, dtype=np.float64)
        tmap_broadcast = np.expand_dims(tmap_refined, axis=2)

        # 逆運算
        enhanced_img = (img.astype(np.float64) - atmos_light) / tmap_broadcast + atmos_light

        # 將值限制在 [0, 255] 範圍內
        enhanced_img = np.clip(enhanced_img, 0, 255)

        # 簡單的色彩平衡調整（模擬多色彩空間嵌入的效果）
        # 此處使用白平衡的一個簡單實現
        enhanced_img_lab = cv2.cvtColor(enhanced_img.astype(np.uint8), cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(enhanced_img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_img_lab = cv2.merge((l_enhanced, a, b))
        final_img = cv2.cvtColor(enhanced_img_lab, cv2.COLOR_Lab2BGR)

        return final_img.astype(np.uint8)


# ==============================================================================
# 3. GUI 應用程式
# 使用 tkinter 創建一個使用者友善的圖形介面。
# ==============================================================================
class ImageEnhancerApp:
    def __init__(self, root, ucolor_model):
        """
        初始化應用程式。

        Args:
            root (tk.Tk): tkinter 的主視窗。
            ucolor_model (UcolorModel): Ucolor 模型實例。
        """
        self.root = root
        self.root.title("Ucolor 水下影像增強")
        # 設定一個初始大小
        self.root.geometry("1000x600")

        self.model = ucolor_model

        self.original_img_pil = None
        self.enhanced_img_pil = None

        # --- 創建 UI 元件 ---
        # 主框架，用於容納所有元件
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 按鈕框架
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(pady=10)

        self.load_button = tk.Button(button_frame, text="載入影像", command=self.load_image)
        self.load_button.pack()

        # 影像顯示框架，將用於自適應佈局
        self.image_display_frame = tk.Frame(self.main_frame)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True)

        # 創建兩個標籤用於顯示圖片
        self.panel_a = tk.Label(self.image_display_frame, text="原始影像", font=("Helvetica", 14))
        self.panel_b = tk.Label(self.image_display_frame, text="增強後影像", font=("Helvetica", 14))

        # 綁定視窗大小改變事件
        self.root.bind('<Configure>', self.handle_resize)

    def load_image(self):
        """
        打開檔案對話框讓使用者選擇影像，並進行增強處理。
        """
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return

        try:
            # 載入原始影像
            self.original_img_pil = Image.open(path)

            # --- 執行影像增強 ---
            # 將 PIL Image 轉換為 OpenCV 格式 (BGR)
            img_cv = cv2.cvtColor(np.array(self.original_img_pil), cv2.COLOR_RGB2BGR)

            # 1. 估算傳輸圖
            print("正在估算介質傳輸圖...")
            atmos_light = get_atmospheric_light(img_cv, get_dark_channel(img_cv))
            transmission_map = get_transmission_map(img_cv, atmos_light)

            # 2. 執行 Ucolor 增強
            print("正在執行 Ucolor 增強...")
            enhanced_img_cv = self.model.predict(img_cv, transmission_map)

            # 將 OpenCV 影像轉回 PIL Image
            enhanced_img_rgb = cv2.cvtColor(enhanced_img_cv, cv2.COLOR_BGR2RGB)
            self.enhanced_img_pil = Image.fromarray(enhanced_img_rgb)

            # 首次顯示影像
            self.update_image_display()

        except Exception as e:
            messagebox.showerror("錯誤", f"處理影像時發生錯誤: {e}")
            self.panel_a.pack_forget()
            self.panel_b.pack_forget()

    def update_image_display(self, event=None):
        """
        根據視窗大小更新影像顯示。
        """
        if not self.original_img_pil:
            return

        # 獲取顯示框架的當前尺寸
        frame_width = self.image_display_frame.winfo_width()
        frame_height = self.image_display_frame.winfo_height()

        # --- 自適應佈局邏輯 ---
        # 如果寬度足夠，則左右並排顯示；否則上下堆疊
        is_horizontal = frame_width > frame_height and frame_width > 800

        # 清空舊的佈局
        self.panel_a.pack_forget()
        self.panel_b.pack_forget()

        if is_horizontal:
            # 水平佈局
            self.panel_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            self.panel_b.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
            # 每張圖片大約佔用一半的寬度
            max_width = frame_width // 2 - 15
            max_height = frame_height - 10
        else:
            # 垂直佈局
            self.panel_a.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
            self.panel_b.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=5)
            max_width = frame_width - 10
            max_height = frame_height // 2 - 15

        self._update_single_panel(self.panel_a, self.original_img_pil, max_width, max_height)
        self._update_single_panel(self.panel_b, self.enhanced_img_pil, max_width, max_height)

    def _update_single_panel(self, panel, img_pil, max_w, max_h):
        """輔助函式，更新單個影像面板"""
        if img_pil is None or max_w <= 0 or max_h <= 0:
            return

        # 保持長寬比進行縮放
        img_copy = img_pil.copy()
        img_copy.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img_copy)
        panel.config(image=photo)
        panel.image = photo  # 保持對 photo 的引用，防止被垃圾回收

    def handle_resize(self, event):
        """處理視窗大小變動事件"""
        # 為了避免在拖動過程中頻繁重繪，可以加上延遲 (debounce)
        # 此處為簡化，直接呼叫更新
        if hasattr(self, '_after_id'):
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(100, self.update_image_display, event)


if __name__ == '__main__':
    # 在 Python 3 中需要引入 math
    import math

    # 檢查 TensorFlow 版本
    print(f"TensorFlow 版本: {tf.__version__}")

    # 這裡可以填入你訓練或轉換好的 Ucolor 模型路徑
    # model_path = "path/to/your/ucolor_model.h5"
    model_path = None  # 使用模擬模式

    # 初始化模型
    ucolor_model = UcolorModel(model_path)

    # 創建並運行 GUI
    root = tk.Tk()
    app = ImageEnhancerApp(root, ucolor_model)
    root.mainloop()
