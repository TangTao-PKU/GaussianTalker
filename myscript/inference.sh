
YOUR_DATASET_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data
DATASET_NAME=yang_talk
YOUR_MODEL_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data/${DATASET_NAME}/model
AUDIO_DIR=/data0/tangt/zpingan-Intern/GaussianTalker/data/audio/

export CUDA_VISIBLE_DEVICES=7
python ../render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
    --model_path ${YOUR_MODEL_DIR} \
    --configs /data0/tangt/zpingan-Intern/GaussianTalker/arguments/64_dim_2_transformer.py \
    --iteration 10000 \
    --batch 32 \
    --custom_aud ${AUDIO_DIR}/daihuo_female4.npy \
    --custom_wav ${AUDIO_DIR}/daihuo_female4.wav \
    --skip_train \
    --skip_test
    # --render_attention