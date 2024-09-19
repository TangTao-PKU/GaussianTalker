import os

path = "/data0/tangt/zpingan-Intern/GaussianTalker/data/audio/daihuo_female2.wav"

if __name__ == '__main__':
    print(f'[INFO] ===== extract audio labels for {path} =====')
    # cmd = f'python nerf/asr.py --wav {path} --save_feats'
    # os.system(cmd)
    cmd = f'python /data0/tangt/zpingan-Intern/GaussianTalker/data_utils/deepspeech_features/extract_ds_features.py --input {path}'
    os.system(cmd)
    import shutil
    shutil.copy(path.replace('.wav', '.npy'), path.replace('.wav', '_ds.npy'))
    print(f'[INFO] ===== extracted audio labels =====')


# python data_utils/deepspeech_features/extract_ds_features.py --input new_for_inference.wav

# def extract_audio_features(path, mode='wav2vec'):

#     print(f'[INFO] ===== extract audio labels for {path} =====')
#     if mode == 'wav2vec':
#         cmd = f'python nerf/asr.py --wav {path} --save_feats'
#     else: # deepspeech
#         cmd = f'python data_utils/deepspeech_features/extract_ds_features.py --input {path}'
#     os.system(cmd)
#     import shutil
#     shutil.copy(path.replace('.wav', '.npy'), path.replace('.wav', '_ds.npy'))
#     print(f'[INFO] ===== extracted audio labels =====')
