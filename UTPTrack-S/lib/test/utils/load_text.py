import numpy as np
import pandas as pd
import json

def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str

def load_list(sequence_name):
    with open("/home/hk-project-p0022189/tum_yvc3016/haowu/UTPTrack/UTPTrack-S-0812/qwen_tnl2k_text.json", "r", encoding="utf-8") as infile:
        data = [json.loads(line) for line in infile]

    # 过滤出image_path包含sequence的条目，并提取对应的new_description
    text_list = [item["new_description"] for item in data if f"/{sequence_name}/" in item["image_path"]]

    return text_list
