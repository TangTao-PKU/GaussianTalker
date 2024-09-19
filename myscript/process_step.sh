export CUDA_VISIBLE_DEVICES=6

dataset=hehe0521
workspace=models/trial_obama0802_head
# asr_model=hubert
asr_model=deepspeech

#python data_utils/dynamic_roi_extract.py data/$dataset --vname chenli_full.mp4
#python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --train_percent 0.55 --longhair 0  && wait
##python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 1 && wait
##python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 2 && wait
###python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 3 && wait
##python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 4 --longhair 1 && wait
###python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 5 && wait
#python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 6 && wait
#python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 7 && wait
python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 8 && wait
python data_utils/process.py data/$dataset/${dataset}.mp4 --asr $asr_model --task 9 
