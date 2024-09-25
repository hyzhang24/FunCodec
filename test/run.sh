#!/bin/bash

# 设置Python环境（如果需要）
# source /path/to/your/virtualenv/bin/activate

# 运行训练脚本
python train.py

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed."
fi