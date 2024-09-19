YOUR_DATASET_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data
DATASET_NAME=obama
YOUR_MODEL_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data/obama/model
AUDIO_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data/audio/

export CUDA_VISIBLE_DEVICES=7
python ../render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
    --model_path ${YOUR_MODEL_DIR} \
    --configs /data0/tangt/zpingan-Intern/GaussianTalker/arguments/64_dim_1_transformer.py \
    --iteration 10000 \
    --batch 128