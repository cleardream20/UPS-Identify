# UPS-Identify
The python package for semantic segmentation and recognition of urban public spaces in remote sensing images

## platform
本实验运行在云GPU平台[Featurize](https://featurize.cn)上运行，使用显卡4090(3090应该也可以)

点进work文件夹，点左上角的加号新建一个terminal

## git clone
```sh
git clone https://github.com/cleardream20/UPS-Identify.git
cd UPS-Identify # 如果不在work文件夹下先 cd work
```

## Steps

### Setup all in one
```sh
bash all_in_one.sh
```

### Train / Predict / PostProcessing
```sh
bash run.sh
```

会出现类似下面的"菜单"
```sh
(ups) ➜ UPS-Identify bash run.sh       
=================================
      请选择要执行的操作
=================================
选项:
  train    - 训练模型 (执行 train.sh)
  pred     - 进行预测 (执行 predict.py)
  postp    - 后处理 (执行 postProcessing.py)
  exit     - 退出程序
=================================
请输入您的选择 (train/pred/postp/exit):

# 输入train进行训练，输入pred进行单张图片预测，输入postp进行后处理全流程，输入exit退出
```

最终输出结果保存在`./mmsegmentation/output_shp`中

下载4个和shp相关的文件，和`merge.tif`就OK（右键文件，点击download）


