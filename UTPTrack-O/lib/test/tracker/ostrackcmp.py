import math

from lib.models.ostrackcmp import build_ostrackcmp
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization, multi_img_set_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrackCMP(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrackCMP, self).__init__(params)
        network = build_ostrackcmp(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor, output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        self.template_list = [template.tensors] * self.num_template
        if self.num_template > 1:
            self.z_patch_arr_list = [self.z_patch_arr] * self.num_template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device)
            if self.cfg.DATA.TEMPLATE.NUMBER > 1:
                self.box_mask_z = self.box_mask_z.repeat(1, self.cfg.DATA.TEMPLATE.NUMBER)

        # save states
        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.tensors.device).unsqueeze(0)]

        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

        # print(f'frame: {self.frame_id}')
        
    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # print(f'frame: {self.frame_id}')
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor, output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        search_list = [search.tensors]

        with torch.no_grad():
            out_dict = self.network.forward(self.template_list, search_list, self.template_anno_list, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        conf_score = out_dict['score']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
                self.z_patch_arr_list[-1] = z_patch_arr  # 因为现在只有1张动态模态，只需要更新它就可以了
                template = self.preprocessor.process(z_patch_arr, z_amask_arr)
                self.template_list.append(template.tensors)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor([self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.template_anno_list.append(prev_box_crop.to(template.tensors.device).unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                if self.num_template == 1:
                    z_patch_arr_show = self.z_patch_arr
                    self.visdom.register(torch.from_numpy(z_patch_arr_show).permute(2, 0, 1), 'image', 1, 'template')
                else:
                    z_patch_arr_list_show = self.z_patch_arr_list
                    z_patch_arr_all_show = multi_img_set_visualization(z_patch_arr_list_show)
                    self.visdom.register(torch.from_numpy(z_patch_arr_all_show).permute(2, 0, 1), 'image', 1, 'template')

                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                # if 'removed_indexes_x' in out_dict and out_dict['removed_indexes_x']:
                if ('removed_indexes_x' in out_dict and out_dict['removed_indexes_x'] is not None and
                        any(removed_idx_x is not None for removed_idx_x in out_dict['removed_indexes_x'])):
                    removed_indexes_x = out_dict['removed_indexes_x']
                    removed_indexes_x = [removed_indexes_x_i.cpu().numpy() for removed_indexes_x_i in removed_indexes_x]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_x)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrackCMP
