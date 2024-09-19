export CUDA_VISIBLE_DEVICES=6
YOUR_DATASET_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data/
DATASET_NAME=yang_silence
YOUR_MODEL_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data//${DATASET_NAME}/model

# 检查目录是否存在
if [ -d "$YOUR_MODEL_DIR" ]; then
    # 删除目录及其内容
    rm -rf "$YOUR_MODEL_DIR"
fi

# 重新创建目录
mkdir -p "$YOUR_MODEL_DIR"

python ../train.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
    --model_path ${YOUR_MODEL_DIR} \
    --configs /data0/tangt/zpingan-Intern/GaussianTalker/arguments/64_dim_2_transformer.py