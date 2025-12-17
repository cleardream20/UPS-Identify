#!/bin/bash

# 检查环境是否已存在
if conda env list | grep -q "^ups "; then
    echo "环境 'ups' 已存在，跳过创建"
else
    echo "正在创建 conda 环境 'ups'..."
    conda create -n ups python=3.9 -y
fi

eval "$(conda shell.bash hook)"
conda activate ups
