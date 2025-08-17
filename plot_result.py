import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# 由於最終顯示的是英文，理論上不再需要中文字體，但保留也無妨。
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """繪製混淆矩陣。"""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)  # 放大標題字體
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")  # 調整旋轉和對齊方式
    plt.yticks(tick_marks, classes[::-1])

    # 調整數值顯示格式和顏色閾值
    # fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 將小數乘以100，格式化為不帶小數點的整數，並加上 '%' 符號
        percentage_text = f"{cm[i, j] * 100:.0f}%"

        plt.text(j, i, percentage_text,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)  # 保持您喜歡的字體大小

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()


def load_results_and_plot_confusion_matrix(csv_path):
    """
    從 CSV 文件加载测试结果，並繪製帶有英文標籤的混淆矩阵。
    """
    # ==================== 修改開始 ====================

    # 1. 在這裡定義你的中英文名稱對應關係
    #    請將 '中文名X' 替換為你CSV檔案中實際使用的中文魚名
    name_mapping = {
        '鯿魚': 'Carassius Auratus',
        '鱅魚': 'Hypophthalmichthys',
        '吳郭魚': 'Thilapa',
        '鯪魚': 'Abramis Brama',
        '鰂魚': 'Cyprinidae'
        # 如果還有其他類別，請繼續添加
    }

    # ==================== 修改結束 ====================

    true_labels = []
    predicted_labels = []

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过表头

        for row in csvreader:
            _, true_label, predicted_label = row  # 解包時忽略第一列 image_path
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

    # 獲取所有中文類別名稱，並保持一個固定的順序
    class_names_chinese = sorted(list(set(true_labels)))

    # 根據中文類別的順序，生成對應的英文類別名稱列表
    try:
        class_names_english = [name_mapping[name] for name in class_names_chinese]
    except KeyError as e:
        print(f"錯誤：在你的 'name_mapping' 字典中找不到鍵 {e}。")
        print("請確保CSV中的所有中文魚名都已經在字典中定義了對應的英文名。")
        return

    # 使用中文標籤來計算混淆矩陣，確保數據與標籤對應正確
    cm = confusion_matrix(true_labels, predicted_labels, labels=class_names_chinese)

    # 歸一化混淆矩陣
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 處理除以0的情況 (如果某個類別總數為0)

    # 翻轉矩陣以匹配你的顯示邏輯
    cm_flipped = np.flipud(cm_normalized)

    # 繪製混淆矩陣，但傳入的是英文標籤用於顯示
    plt.figure(figsize=(12, 10))  # 稍微增大圖像尺寸以容納標籤
    plot_confusion_matrix(cm_flipped, classes=class_names_english, title='Normalized Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    csv_path = "fish_identification_results.csv"
    load_results_and_plot_confusion_matrix(csv_path)
