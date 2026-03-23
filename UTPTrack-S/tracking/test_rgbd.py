import cv2
import os
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
import importlib
import multiprocessing
import torch
import time
from lib.test.tracker.sutrack import SUTRACK
from lib.test.tracker.sutrackcmp import SUTRACKCMP
from lib.train.dataset.depth_utils import get_x_frame


def get_parameters(script_name, yaml_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(script_name))
    params = param_module.parameters(yaml_name)
    return params


def getImgAndGT(seq_path):
    RGB_img_list = sorted([seq_path + '/color/' + p for p in os.listdir(seq_path + '/color') if p.endswith(".jpg")])
    D_img_list = sorted([seq_path + '/depth/' + p for p in os.listdir(seq_path + '/depth') if p.endswith(".png")])
    RGB_gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
    RGB_gt[np.isnan(RGB_gt)] = 0.0
    if RGB_gt.shape[1] > 4:
        gt_x_all = RGB_gt[:, [0, 2, 4, 6]]
        gt_y_all = RGB_gt[:, [1, 3, 5, 7]]

        x1 = np.amin(gt_x_all, 1).reshape(-1, 1)
        y1 = np.amin(gt_y_all, 1).reshape(-1, 1)
        x2 = np.amax(gt_x_all, 1).reshape(-1, 1)
        y2 = np.amax(gt_y_all, 1).reshape(-1, 1)

        RGB_gt = np.concatenate((x1, y1, x2 - x1, y2 - y1), 1)


    return RGB_img_list, D_img_list, RGB_gt

def save_result(save_folder, result_arr, track_times, scores, seq_name):
    result_file = join(save_folder, seq_name + '_001.txt')
    result_time_file = join(save_folder, seq_name + '_001_time.value')
    result_conf_file = join(save_folder, seq_name + '_001_confidence.value')
    result_list = result_arr.tolist()
    result_list[0] = [1]
    with open(result_file, 'w') as f:
        for x in result_list:
            f.write(','.join([str(i) for i in x]) + '\n')

    with open(result_time_file, 'w') as f:
        for x in track_times:
            f.write("{:.15f}\n".format(x))

    with open(result_conf_file, 'w') as f:
        for x in scores:
            f.write('\n') if x is None else f.write("{:.15f}\n".format(x))

def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=60, debug=0, script_name='sutrack'):

    seq_txt = seq_name + '_001'
    # save_name = '{}_{}'.format(script_name, yaml_name)
    save_name = f'{yaml_name}'
    # save_path = f'./Depthtrack_workspace/results/' + save_name +  '/rgbd-unsupervised/' + seq_name +  '/' + seq_txt + '.txt'
    save_path = f'results/' + save_name +  '/rgbd-unsupervised/' + seq_name +  '/' + seq_txt + '.txt'

    # save_folder = f'./Depthtrack_workspace/results/' + save_name + '/rgbd-unsupervised/' +  seq_name
    save_folder = f'results/' + save_name + '/rgbd-unsupervised/' +  seq_name

    if not debug and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    params = get_parameters(script_name, yaml_name)
    debug_ = debug
    if debug is None:
        debug_ = getattr(params, 'debug', 0)
    params.debug = debug_
    if script_name == 'sutrack':
        tracker = SUTRACK(params, dataset_name)
    elif script_name == 'sutrackcmp':
        tracker = SUTRACKCMP(params, dataset_name)

    seq_path = seq_home + '/' + seq_name
    print('—————————— Process sequence: '+ seq_name +' ——————————————')
    RGB_img_list, D_img_list, RGB_gt = getImgAndGT(seq_path)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    scores = []
    track_times = []
    for frame_idx, (rgb_path, D_path) in enumerate(zip(RGB_img_list, D_img_list)):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            image = get_x_frame(rgb_path, D_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbcolormap'))
            init_info = {'init_bbox': RGB_gt[0].tolist()}
            tracker.initialize(image, init_info)  # xywh
            scores.append(None)
        elif frame_idx > 0:
            image = get_x_frame(rgb_path, D_path, dtype=getattr(params.cfg.DATA,'XTYPE','rgbcolormap'))
            info = {'gt_bbox': RGB_gt[frame_idx]}
            outputs = tracker.track(image, info)
            pred_bbox = outputs['target_bbox']
            conf_score = outputs["best_score"].item() if 'best_score' in outputs else 1.0
            result[frame_idx] = np.array(pred_bbox)
            scores.append(conf_score)
        toc += cv2.getTickCount() - tic
        track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
    toc /= cv2.getTickFrequency()
    if not debug:
        save_result(save_folder, result, track_times, scores, seq_name)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBD dataset.')
    parser.add_argument('--script_name', type=str, default='sutrackcmp', help='Name of tracking method.')
    parser.add_argument('--yaml_name', type=str, default='ce_b224', help='Name of parameter file.')
    parser.add_argument('--dataset_name', type=str, default='DepthTrack', help='Name of dataset.')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=None, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='sequential', type=str, help='running mode: [sequential , parallel]')
    parser.add_argument('--debug', default=1, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    cur_dir = abspath(dirname(__file__))
    ## path initialization
    seq_list = None
    if dataset_name == 'DepthTrack':
        seq_home = '/home/wuhao/workspace/SUTrack/Depthtrack_workspace/sequences'
        with open(join(seq_home, 'list.txt'), 'r') as f:
            seq_list = f.read().splitlines()
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")


    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")