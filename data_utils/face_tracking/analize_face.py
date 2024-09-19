import os
import sys
import argparse
import cv2
import numpy as np
from data_loader import load_dir
from facemodel import Face_3DMM
from util import *
from render_3dmm import Render_3DMM
from tqdm import trange


def imgs_to_video(img_dir, video_path, audio_path=None, verbose=False):
    cmd = f"ffmpeg -i {img_dir}/%5d.png "
    if audio_path is not None:
        cmd += f"-i {audio_path} "
        cmd += "-strict -2 "
    cmd += "-c:v libx264 -pix_fmt yuv420p -b:v 2000k -y "
    if verbose is False:
        cmd += " -v quiet "
    cmd += f"{video_path} "

    os.system(cmd)

def load_mask_dir(path, start, end):
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, f'{format(i, "05d")}.png')):
            imgs_paths.append(os.path.join(path, f'{format(i, "05d")}.png'))
    return imgs_paths


def draw_lmk(img, l_lmk, lmk_type='2d'):
    assert lmk_type in ['2d', '3d'], 'lmk_type must be 2d or 3d!'

    if lmk_type == '3d':
        l_lmk = l_lmk[:, :2]  # [68, 2]

    for i in range(len(l_lmk)):
        x, y = list(map(int, l_lmk[i]))
        if i in eye_idx:
            color = (0, 0, 255)
        elif i in mouth_idx:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        if lmk_type == '3d':
            img = cv2.rectangle(img, pt1=(x - 1, y - 1), pt2=(x + 1, y + 1), color=color, thickness=1)
        else:
            img = cv2.circle(img, center=(x, y), radius=3, color=color, thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(l_lmk)):
        x, y = list(map(int, l_lmk[i]))
        img = cv2.putText(img, f"{i}", org=(x, y), fontFace=font, fontScale=0.3, color=(255, 0, 0))


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="../../data/obama", help="idname of target person"
    )
    parser.add_argument("--img_h", type=int, default=512, help="image height")
    parser.add_argument("--img_w", type=int, default=512, help="image width")
    parser.add_argument("--frame_num", type=int, default=7500, help="image number")
    parser.add_argument("--show_3dface", type=bool, default=True, help='enable show_3dface')
    parser.add_argument("--show_2dlmk", type=bool, default=True, help='enable show_2dlmk')
    parser.add_argument("--show_3dlmk", type=bool, default=True, help='enable show_3dlmk')
    parser.add_argument("--use_cache", type=bool, default=False, help='use mask cache')
    args = parser.parse_args()

    show_3dface = args.show_3dface
    show_2dlmk = args.show_2dlmk
    show_3dlmk = args.show_3dlmk
    use_cache = args.use_cache

    start_id = 0
    end_id = args.frame_num

    lms, img_paths = load_dir(os.path.join(args.path, 'ori_imgs'), start_id, end_id)

    print(len(img_paths))

    if use_cache:
        mask_paths = load_mask_dir(os.path.join(args.path, 'tmp_track_mask'), start_id, end_id)

    num_frames = lms.shape[0]
    h, w = args.img_h, args.img_w
    cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).cuda()
    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    model_3dmm = Face_3DMM(
        os.path.join(dir_path, "3DMM"), id_dim, exp_dim, tex_dim, point_num
    )

    params_dict = torch.load(os.path.join(args.path, 'track_params.pt'))

    id_para, exp_para = params_dict['id'].cuda(), params_dict['exp'].cuda()
    euler_angle, trans = params_dict['euler'].cuda(), params_dict['trans'].cuda()
    # tex_para, light_para = params_dict['tex'].cuda(), params_dict['light'].cuda()

    focal = params_dict['focal'].cuda()

    light_para = lms.new_zeros((num_frames, 27), requires_grad=True).cuda()
    tex_para = lms.new_zeros((1, tex_dim), requires_grad=True).cuda()

    batch_size = 32
    device_default = torch.device("cuda:7")
    device_render = torch.device("cuda:7")
    renderer = Render_3DMM(focal.detach().cpu(), h, w, batch_size, device_render)

    eye_idx = list(range(36, 48))
    mouth_idx = list(range(48, 68))

    tmp_img_dir = os.path.join(args.path, 'tmp_track_img')
    tmp_mask_dir = os.path.join(args.path, 'tmp_track_mask')
    os.makedirs(tmp_img_dir, exist_ok=True)
    os.makedirs(tmp_mask_dir, exist_ok=True)

    for i in trange(int((num_frames - 1) / batch_size + 1)):
        if (i + 1) * batch_size > num_frames:
            pad_num = (i + 1) * batch_size - num_frames
            sel_pad_ids = np.ones(pad_num, dtype='int') * (num_frames - 1)

            start_n = i * batch_size
            sel_ids = np.hstack([np.arange(start_n, num_frames), sel_pad_ids])
        else:
            start_n = i * batch_size
            sel_ids = np.arange(i * batch_size, i * batch_size + batch_size)

        imgs = []
        for sel_id in sel_ids:
            imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
        imgs = np.stack(imgs)
        sel_imgs = torch.as_tensor(imgs).cuda()
        sel_lms = lms[sel_ids]

        if use_cache:
            render_imgs = []
            for sel_id in sel_ids:
                render_imgs.append(cv2.imread(mask_paths[sel_id]))

        sel_exp_para = exp_para.new_zeros((batch_size, exp_dim), requires_grad=True)
        sel_exp_para.data = exp_para[sel_ids].clone()

        sel_euler = euler_angle.new_zeros((batch_size, 3), requires_grad=True)
        sel_euler.data = euler_angle[sel_ids].clone()

        sel_trans = trans.new_zeros((batch_size, 3), requires_grad=True)
        sel_trans.data = trans[sel_ids].clone()

        sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
        sel_light.data = light_para[sel_ids].clone()

        sel_id_para = id_para.expand(batch_size, -1).detach()
        sel_tex_para = tex_para.expand(batch_size, -1).detach()

        focal_length = lms.new_zeros(1, requires_grad=True)
        focal_length.data += focal

        geometry = model_3dmm.get_3dlandmarks(
            sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy
        )

        # calculate 3dlmk
        sel_3dlmk = forward_transform(geometry, sel_euler, sel_trans, focal_length, cxy)

        # render 3dmm face
        if show_3dface:
            sel_geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
            sel_texture = model_3dmm.forward_tex(sel_tex_para)

            rott_geo = forward_rott(sel_geometry, sel_euler, sel_trans)
            render_imgs = renderer(
                rott_geo.to(device_render),
                sel_texture.to(device_render),
                sel_light.to(device_render),
            )
            render_imgs = render_imgs.to(device_default)

        for j, img in enumerate(imgs):
            index = i * batch_size + j
            if index > num_frames - 1:
                break

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if show_3dface:
                if use_cache:
                    render_img = render_imgs[j][:, :, :3]
                else:
                    render_img = render_imgs[j].cpu().detach()[:, :, :3].numpy().astype('uint8')
                    render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(tmp_mask_dir, f'{format(index, "05d")}.png'), render_img)
                # cv2.imwrite(os.path.join(tmp_mask_dir, f'{format(index, "05d")}.jpg'), img)

                alpha = 0.6  # 第一张图像的权重
                beta = 0.4  # 第二张图像的权重
                gamma = 0  # 亮度调整参数，设为0表示不进行调整
                img = cv2.addWeighted(img, alpha, render_img, beta, gamma)

            if show_2dlmk:
                _2dlmk = sel_lms[j]
                draw_lmk(img, _2dlmk, lmk_type='2d')

            if show_3dlmk:
                _3dlmk = sel_3dlmk[j]
                draw_lmk(img, _3dlmk, lmk_type='3d')

            cv2.imwrite(os.path.join(tmp_img_dir, f'{format(index, "05d")}.png'), img)

    if show_3dface:
        imgs_to_video(tmp_mask_dir, os.path.join(args.path, 'aud_3dmm.mp4'), os.path.join(args.path, 'aud.wav'))

    imgs_to_video(tmp_img_dir, os.path.join(args.path, 'aud_with_3dmm.mp4'), os.path.join(args.path, 'aud.wav'))
    # os.system(f"rm -r {tmp_img_dir}")
    # break

    print('ok')
