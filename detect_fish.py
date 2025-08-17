import google.generativeai as genai
from PIL import Image
import io

GOOGLE_API_KEY = "AIzaSyAdd-4oa_R3UnHv1TPlKBbCeTzWnIzaeGk"

genai.configure(api_key=GOOGLE_API_KEY)

def img_to_byte(image: Image) -> bytes:
  img_byte_arr = io.BytesIO()
  image.save(img_byte_arr, format='JPEG')  # 或 PNG, GIF 等
  img_byte_arr = img_byte_arr.getvalue()
  return img_byte_arr


def generate_text_from_image_and_text(image_path, prompt_text, few_shot_example=None):
  try:
    # 加载图像
    image = Image.open(image_path)

    image_bytes = img_to_byte(image)

    # 选择 Gemini 2.0 Flash 模型
    model = genai.GenerativeModel('gemini-2.0-flash')
    # 构建请求内容
    contents = []

    if few_shot_example:
        for example in few_shot_example:
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
    return response.text

  except Exception as e:
    print(f"发生错误: {e}")
    return None


#  使用示例
if __name__ == "__main__":
    image_path = r"C:\Users\acer\OneDrive\文件\WeChat Files\wxid_09h8byqdlu9q22\FileStorage\File\2025-04\T_ans-20241104T112415Z-001\吳郭魚\0.3621562957763672_clear.jpg"
    prompt_text = "根据这张图片，判断这是以下哪种鱼类：鯿魚、大頭魚、金山鰂、鯪魚、鰂魚、白條。如果都不是，请回答'都不是'。请只输出鱼类名称，不要包含其他解释。"
    few_shot_examples = [
        {"image_path": r"C:\Users\acer\Downloads\大头鱼.jpg", "label": "大頭魚"},
        {"image_path": r"C:\Users\acer\Downloads\武昌鱼.jpg", "label": "鯿魚"},
        {"image_path": r"C:\Users\acer\Downloads\罗非.jpg", "label": "金山鰂"},
        {"image_path": r"C:\Users\acer\Downloads\鯪魚.jpg", "label": "鯪魚"},
        {"image_path": r"C:\Users\acer\Downloads\鰂魚.jpg", "label": "鰂魚"},
        {"image_path": r"C:\Users\acer\Downloads\baitiaoyu_fish1.jpg", "label": "白條"}
    ]
    generated_text = generate_text_from_image_and_text(image_path, prompt_text)

    if generated_text:
        print("生成的文本:\n", generated_text, sep='')
    else:
        print("生成文本失败。")
