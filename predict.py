import os
os.chdir('mmsegmentation')

import numpy as np
import matplotlib.pyplot as plt

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2

# 模型 config 配置文件
# 文件路径均为相对于mmsegmentation文件夹的相对路径
config_file = 'MyConfigs/UpsDataset_KNet.py'

# 模型 checkpoint 权重文件
checkpoint_file = './work_dirs/UpsDataset-KNet/iter_10000.pth'

# device = 'cpu'  # 使用CPU进行预测
device = 'cuda:0' # 使用GPU进行预测

model = init_model(config_file, checkpoint_file, device=device)

img = 's50'
img_path = f'Ups_Semantic_Seg_Mask/img_dir/train/{img}.jpg'

img_bgr = cv2.imread(img_path)
plt.figure(figsize=(8, 8))
plt.imshow(img_bgr[:,:,::-1])
plt.show()

result = inference_model(model, img_bgr)
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

# 预测结果1
plt.figure(figsize=(8, 8))
plt.imshow(pred_mask)
plt.savefig(f'outputs/{img}-0.jpg')
plt.show()

# 预测结果2
# 显示语义分割结果
plt.figure(figsize=(10, 8))
plt.imshow(img_bgr[:,:,::-1])
plt.imshow(pred_mask, alpha=0.55) # alpha 高亮区域透明度，越小越接近原图
plt.axis('off')
plt.savefig(f'outputs/{img}-1.jpg')
plt.show()

# 预测结果3
plt.figure(figsize=(14, 8))

plt.subplot(1,2,1)
plt.imshow(img_bgr[:,:,::-1])
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_bgr[:,:,::-1])
plt.imshow(pred_mask, alpha=0.6) # alpha 高亮区域透明度，越小越接近原图
plt.axis('off')
plt.savefig(f'outputs/{img}-2.jpg')
plt.show()
