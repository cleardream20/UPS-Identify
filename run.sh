#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示菜单函数
show_menu() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${GREEN}      请选择要执行的操作${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo -e "${YELLOW}选项:${NC}"
    echo -e "  ${GREEN}train${NC}    - 训练模型 (执行 train.sh)"
    echo -e "  ${GREEN}pred${NC}     - 进行预测 (执行 predict.py)"
    echo -e "  ${GREEN}postp${NC}    - 后处理 (执行 postProcessing.py)"
    echo -e "  ${RED}exit${NC}     - 退出程序"
    echo -e "${BLUE}=================================${NC}"
}

show_menu

# 读取用户输入
read -p "请输入您的选择 (train/pred/postp/exit): " user_input

# 将输入转换为小写
user_input=$(echo "$user_input" | tr '[:upper:]' '[:lower:]')

case $user_input in
    "train")
        echo -e "${GREEN}开始训练...${NC}"
        if [[ -f "train.sh" ]]; then
            bash train.sh
        else
            echo -e "${RED}错误: 找不到 train.sh 文件${NC}"
        fi
        ;;
        
    "pred")
        echo -e "${GREEN}开始预测...${NC}"
        if [[ -f "predict.py" ]]; then
            python predict.py
        else
            echo -e "${RED}错误: 找不到 predict.py 文件${NC}"
        fi
        ;;
        
    "postp")
        echo -e "${GREEN}开始后处理...${NC}"
        if [[ -f "postProcessing.py" ]]; then
            python postProcessing.py
        else
            echo -e "${RED}错误: 找不到 postProcessing.py 文件${NC}"
        fi
        ;;
        
    "exit")
        echo -e "${BLUE}退出程序，再见！${NC}"
        ;;
        
    *)
        echo -e "${RED}无效的输入！请输入 train, pred, postp 或 exit${NC}"
        ;;
esac
