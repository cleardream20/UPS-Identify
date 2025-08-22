import os
os.chdir('mmsegmentation')

import numpy as np
import matplotlib.pyplot as plt

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2

import rasterio
from rasterio.transform import from_origin
from osgeo import gdal, ogr, osr
import glob
from PIL import Image
from tqdm import tqdm

from mmseg.apis import init_model, inference_model, show_result_pyplot

Image.MAX_IMAGE_PIXELS = None  # 禁用解压缩炸弹检查

def mmseg_predict(input_patch_path, output_patch_path, model):
    """
    看class_type = x时究竟对应的是哪个种类
    目前没有好方法，只能一个一个尝试
    例如，class_type = 0，运行代码，图片中该类对应的结果会被保存在mmsegmentation/outputs文件夹中
    结果图中蓝色部分即为相应类对应结果的预测结果
    """
    img_bgr = cv2.imread(input_patch_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 创建白色区域掩码
    white_mask = np.all(img_rgb == [254, 254, 254], axis=-1)

    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # 将白色区域强制设为背景(0)
    pred_mask[white_mask] = 0

    unique, counts = np.unique(pred_mask, return_counts=True)
    print(f"预测结果类别分布: {dict(zip(unique, counts))}")

    output = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    class_type = 9
    # 3 water
    # 4 squares
    # 5 vegetation
    # 8 greenland
    # 9 park
    output[pred_mask == class_type] = [0, 0, 200]

    Image.fromarray(output).save(output_patch_path)

# 模型 config 配置文件
config_file = 'MyConfigs/UpsDataset_KNet.py'

# 模型 checkpoint 权重文件
checkpoint_file = './work_dirs/UpsDataset-KNet/iter_10000.pth'

# device = 'cpu'  # 使用CPU进行预测
device = 'cuda:0' # 使用GPU进行预测

model = init_model(config_file, checkpoint_file, device=device)

img = 's50'
img_path = f'Ups_Semantic_Seg_Mask/img_dir/train/{img}.jpg'

out_path = f'./outputs/{img}_pred.jpg'

mmseg_predict(img_path, out_path, model)

