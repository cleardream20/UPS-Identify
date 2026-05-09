import numpy as np
import cv2
from pathlib import Path

# 定义颜色映射：像素值 -> (R, G, B)
COLOR_MAP = {
    0:  [242, 239, 241],  # background
    1:  [130, 130, 130],  # buildings
    2:  [255, 255, 255],  # roads
    3:  [100, 140, 156],  # water
    4:  [214, 47,  39],   # squares
    5:  [174, 188, 166],  # vegetation
    6:  [255, 182, 179],  # vacant
    7:  [255, 127, 127],  # playground
    8:  [212, 217, 161],  # greenland
    9:  [212, 217, 161],  # park (与 greenland 同色)
    10: [192, 70,  70],   # parking
    11: [221, 218, 220],  # housing
    12: [215, 158, 158],  # workland
    13: [255, 255, 255],  # block (与 roads 同色)
}

def visualize_mask(mask_path, output_path=None):
    """
    将灰度标签掩膜转换为彩色可视化图像。

    Args:
        mask_path (str): 输入灰度 PNG 图像路径。
        output_path (str, optional): 输出彩色图像路径。若不提供，则显示图像。
    """
    # 读取灰度图像（确保为单通道）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取图像: {mask_path}")

    h, w = mask.shape
    # 创建 RGB 图像（3 通道）
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    # 根据像素值填充颜色
    for pixel_val, color in COLOR_MAP.items():
        # 找到对应像素值的位置
        mask_indices = (mask == pixel_val)
        # 注意：COLOR_MAP 中是 (R, G, B)，OpenCV 默认通道顺序为 BGR，需反转
        rgb_image[mask_indices] = color[::-1]  # 转为 BGR 顺序

    # 保存或显示
    if output_path:
        cv2.imwrite(output_path, rgb_image)
        print(f"可视化图像已保存至: {output_path}")
    else:
        cv2.imshow("Mask Visualization", rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    # 替换为你的掩膜图像路径
    input_mask = "D:/User/Desktop/sq_test1/s11.png"
    output_image = "D:/Square/process_seg/A_background/output/s11.jpg"
    visualize_mask(input_mask, output_image)
