import google.generativeai as genai
from PIL import Image
import io
import os

# 请在此处配置您的 Google Gemini API 密钥
# GOOGLE_API_KEY = "YOUR_API_KEY"  #请替换成你自己的key

def img_to_byte(image: Image) -> bytes:
    """将 PIL 图像转换为字节数组。"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')  # 或者 PNG, GIF 等
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def generate_text_from_image_and_text(image_path, prompt_text, api_key, few_shot_examples=None):
    """
    使用 Gemini 模型从图像和文本生成文本。

    Args:
        image_path: 图像文件的路径。
        prompt_text: 提示文本。
        few_shot_examples: 少量示例列表，每个示例应包含 "image_path" 和 "label"。

    Returns:
        生成的文本，如果发生错误则返回 None。
    """
    try:
        # 配置 Gemini API
        genai.configure(api_key=api_key)  # 使用传入的 API 密钥

        # 加载图像
        image = Image.open(image_path)
        image_bytes = img_to_byte(image)

        # 选择 Gemini 模型
        model = genai.GenerativeModel('gemini-2.0-flash')

        # 构建请求内容
        contents = []

        if few_shot_examples:
            for example in few_shot_examples:
                ex_img = Image.open(example["image_path"])
                ex_img_bytes = img_to_byte(ex_img)

                contents.append({
                    "mime_type": "image/jpeg",
                    "data": ex_img_bytes
                })
                contents.append(f"答案:{example['label']}")

        contents.append({
            "mime_type": "image/jpeg",
            "data": image_bytes
        })
        contents.append(prompt_text)

        # 生成文本
        response = model.generate_content(contents)

        # 返回结果
        return response.text.strip()  # 去除首尾空格

    except Exception as e:
        print(f"发生错误: {e}")
        return None


# 默认的few-shot examples
DEFAULT_FEW_SHOT_EXAMPLES = [
    {"image_path": r"C:\Users\acer\OneDrive\文件\WeChat Files\wxid_09h8byqdlu9q22\FileStorage\File\2025-04\鱅魚\0.265167796044461_original.jpg", "label": "鱅魚"},
    {"image_path": r"C:\Users\acer\OneDrive\文件\WeChat Files\wxid_09h8byqdlu9q22\FileStorage\File\2025-04\鯿魚\0.3087807496798944_clear.jpg", "label": "鯿魚"},
    {"image_path": r"C:\Users\acer\OneDrive\文件\WeChat Files\wxid_09h8byqdlu9q22\FileStorage\File\2025-04\吳郭魚\0.4158779144287109_clear.jpg", "label": "吳郭魚"},
    {"image_path": r"C:\Users\acer\OneDrive\文件\WeChat Files\wxid_09h8byqdlu9q22\FileStorage\File\2025-04\鯪魚\0.3245551021515377_clear.jpg", "label": "鯪魚"},
    {"image_path": r"C:\Users\acer\OneDrive\文件\WeChat Files\wxid_09h8byqdlu9q22\FileStorage\File\2025-04\鰂魚\0.2522299003601074_clear.jpg", "label": "鰂魚"}
]
