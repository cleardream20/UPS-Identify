#!/bin/bash

set -e  # 任何命令失败时立即退出

echo "=== 启动conda ==="
# # 检查环境是否已存在
# if conda env list | grep -q "^ups "; then
#     echo "环境 'ups' 已存在，跳过创建"
# else
#     echo "正在创建 conda 环境 'ups'..."
#     conda create -n ups python=3.9 -y
# fi

# eval "$(conda shell.bash hook)"
# conda activate ups

source create_conda.sh

echo "=== 检查环境 ==="

# 设置清华镜像源加速下载
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 1. 安装相应版本PyTorch
echo "安装 PyTorch 2.0.1 + CUDA 11.7..."
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# 2. 安装OpenMMLab工具链
echo "安装 OpenMMLab 工具链..."
pip install -U openmim
pip install mmengine==0.10.7

# 3. 安装 MMCV - 使用 HTTP 而不是 HTTPS 避免 SSL 证书问题
echo "安装 MMCV 2.0.0 (使用HTTP避免SSL问题)..."
# 使用 http:// 而不是 https://
pip install http://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/mmcv-2.0.0-cp39-cp39-manylinux1_x86_64.whl

# 4. 安装其他依赖
echo "安装其他依赖..."
for pkg in opencv-python pillow matplotlib seaborn tqdm ftfy regex rasterio; do
    if ! pip show "$pkg" >/dev/null 2>&1; then
        echo "安装 $pkg..."
        pip install "$pkg"
    else
        echo "$pkg 已安装，跳过"
    fi
done

# 5. 安装 mmdet
echo "安装 MMDetection..."
pip install 'mmdet>=3.1.0,<4.0.0'

# 6. 克隆MMSegmentation
echo "处理 MMSegmentation..."
if [ ! -d "mmsegmentation" ]; then
    git clone https://github.com/open-mmlab/mmsegmentation.git -b v1.2.2
    cd mmsegmentation
else
    echo "mmsegmentation 目录已存在，进入目录"
    cd mmsegmentation
    git checkout v1.2.2 2>/dev/null || echo "已在 v1.2.2 分支或无法切换"
fi

# 7. 安装MMSegmentation
echo "安装 MMSegmentation..."
pip install -v -e .

# 8. 创建目录结构
echo "创建目录结构..."
mkdir_if_not_exist() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "创建目录: $1"
    else
        echo "目录已存在: $1"
    fi
}

mkdir_if_not_exist "checkpoint"
mkdir_if_not_exist "outputs"
mkdir_if_not_exist "MyConfigs"
mkdir_if_not_exist "input_tif"
# mkdir_if_not_exist "Ups_Semantic_Seg_Mask"

# for split in train val; do
#     mkdir_if_not_exist "Ups_Semantic_Seg_Mask/img_dir/$split"
#     mkdir_if_not_exist "Ups_Semantic_Seg_Mask/ann_dir/$split"
# done

# 安装gdal
cd ..
python3 -m pip install "pre_provided/GDAL-3.4.1-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.whl"

# 安装numpy<2.0
pip install "numpy<2.0"

# geopandas
pip install geopandas



echo "=== 进行文件移动和创建 ==="
# 移动 UpsDataset.py，只有文件存在时才移动
[[ -f "./pre_provided/UpsDataset.py" ]] && mv ./pre_provided/UpsDataset.py ./mmsegmentation/mmseg/datasets/

# 移动 UpsDataset_pipeline.py，只有文件存在时才移动
[[ -f "./pre_provided/UpsDataset_pipeline.py" ]] && mv ./pre_provided/UpsDataset_pipeline.py ./mmsegmentation/configs/_base_/datasets/

# __init__.py 应该覆盖原有文件
[[ -f "./pre_provided/__init__.py" ]] && mv ./pre_provided/__init__.py ./mmsegmentation/mmseg/datasets/

# 检测 Ups_Semantic_Seg_Mask 文件夹是否存在，如果存在则整体移动
[[ -d "./Ups_Semantic_Seg_Mask" ]] && mv ./Ups_Semantic_Seg_Mask ./mmsegmentation/

python KNetConfigGenerator.py

echo "=== 所有操作完成 ==="
