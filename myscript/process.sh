export CUDA_VISIBLE_DEVICES=7

YOUR_DATASET_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data
DATASET_NAME=yang_silence

python ./data_utils/process.py ${YOUR_DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.mp4 