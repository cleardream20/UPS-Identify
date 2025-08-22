#!/bin/bash

set -e  # 任何命令失败时立即退出

echo "=== 检查环境 ==="

# 1. 安装相应版本PyTorch
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# 2. 安装OpenMMLab工具链
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0

# 3. 安装其他依赖（仅安装缺失的库）
for pkg in opencv-python pillow matplotlib seaborn tqdm pytorch-lightning ftfy regex rasterio; do
    pip show "$pkg" >/dev/null || pip install "$pkg" -i https://pypi.tuna.tsinghua.edu.cn/simple
done
pip install 'mmdet>=3.1.0' -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 克隆MMSegmentation（跳过已存在的仓库）
if [ ! -d "mmsegmentation" ]; then
    git clone https://github.com/open-mmlab/mmsegmentation.git -b v1.2.2
else
    echo "mmsegmentation 目录已存在，跳过克隆"
fi

# 5. 再次确保 PyTorch 版本（重复安装以覆盖可能的冲突）
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117


# 6. 安装MMSegmentation
cd mmsegmentation
pip install -v -e .


# 7. 安全创建目录（不覆盖已有内容）
mkdir_if_not_exist() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "创建目录: $1"
    else
        echo "目录已存在，跳过创建: $1"
    fi
}

mkdir_if_not_exist "checkpoint"
mkdir_if_not_exist "outputs"
mkdir_if_not_exist "MyConfigs"

for split in train val; do
    mkdir_if_not_exist "Ups_Semantic_Seg_Mask/img_dir/$split"
    mkdir_if_not_exist "Ups_Semantic_Seg_Mask/ann_dir/$split"
done

echo "=== 所有操作完成 ==="
