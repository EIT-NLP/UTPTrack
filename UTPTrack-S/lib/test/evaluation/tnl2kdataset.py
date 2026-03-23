import os
import json
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text, load_str, load_list


class TNL2kDataset(BaseDataset):
    """
    TNL2k test set
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.text_data = self.load_json_data()
        self.sequence_list = self._get_sequence_list()
    
    def load_json_data(self):
        with open("/home/hk-project-p0022189/tum_yvc3016/haowu/UTPTrack/UTPTrack-S-0812/qwen_tnl2k_text_manual.json", "r", encoding="utf-8") as infile:
            data = [json.loads(line) for line in infile]
        return data

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        text_dsp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        text_dsp = load_str(text_dsp_path)

        # text_dsp = ""

        # 代码添加在这里，最后生成text_list
        text_list = [item["new_description"] for item in self.text_data if f"/{sequence_name}/" in item["image_path"]]
        lens = ground_truth_rect.shape[0]
        if len(text_list) != lens:
            text_list = [text_dsp for _ in range(lens)]
        
        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        # target_class = class_name
        # return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4), text_dsp=text_dsp)
        # return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4), language_query=text_dsp)
        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4), language_query=text_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        for seq in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, seq)):
                sequence_list.append(seq)

        return sequence_list
