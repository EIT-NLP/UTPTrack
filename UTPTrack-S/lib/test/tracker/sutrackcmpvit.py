from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d
from lib.models.sutrackcmpvit import build_sutrackcmpvit
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
import clip
import numpy as np
import os
from lib.train.admin import env_settings
from lib.test.tracker.vis_utils import gen_visualization, multi_img_set_visualization

class SUTRACKCMPVIT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SUTRACKCMPVIT, self).__init__(params)
        network = build_sutrackcmpvit(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0

        if self.debug == 1:
            self._init_visdom(None, 1)
        if self.debug == 2:
            root = env_settings().workspace_dir
            self.save_dir = os.path.join(root, "test/debug")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

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

        # mapping similar datasets
        if 'GOT10K' in DATASET_NAME:
            DATASET_NAME = 'GOT10K'
        if 'LASOT' in DATASET_NAME:
            DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME:
            DATASET_NAME = 'TNL2K'

        # multi modal vision
        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT
        print("MULTI_MODAL_VISION is: ", self.multi_modal_vision)

        #multi modal language
        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT
        print("MULTI_MODAL_LANGUAGE is: ", self.multi_modal_language)

        #using nlp information
        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT
        print("USE_NLP is: ", self.use_nlp)

        self.task_index_batch = None


    def initialize(self, image, info: dict):
        # print(f"frame id: {self.frame_id}")
        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor, output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and (template.size(1) == 3):
            template = torch.cat((template, template), axis=1)
        self.template_list = [template] * self.num_template
        if self.num_template > 1:
            self.z_patch_arr_list = [self.z_patch_arr] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)]
        self.frame_id = 0

        # language information
        if self.multi_modal_language:
            if self.use_nlp:
                init_nlp = info.get("init_nlp")
            else:
                init_nlp = None
            text_data, _ = self.extract_token_from_nlp_clip(init_nlp)
            text_data = text_data.unsqueeze(0).to(template.device)
            with torch.no_grad():
                self.text_src = self.network.forward_textencoder(text_data=text_data)
        else:
            self.text_src = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE   # 临时增加的，为了可视化


    def track(self, image, info: dict = None):
        # print(f"frame id: {self.frame_id}")
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor, output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        if self.multi_modal_vision and (search.size(1) == 3):
            search = torch.cat((search, search), axis=1)
        search_list = [search]

        # run the encoder
        with torch.no_grad():
            enc_opt, aux_dict = self.network.forward_encoder(self.template_list, search_list, self.template_anno_list, self.text_src, self.task_index_batch)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=enc_opt)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True: # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['offset_map'], return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
                self.z_patch_arr_list[-1] = z_patch_arr # 因为现在只有1张动态模态，只需要更新它就可以了
                template = self.preprocessor.process(z_patch_arr)
                if self.multi_modal_vision and (template.size(1) == 3):
                    template = torch.cat((template, template), axis=1)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        # for debug
        x_dte_patch_arr_show = None
        if image.shape[-1] == 6:
            image_show = image[:,:,:3]
            x_rgb_patch_arr_show = x_patch_arr[:,:,:3]
            x_dte_patch_arr_show = x_patch_arr[:,:,3:]
            x_patch_arr_show = multi_img_set_visualization([x_rgb_patch_arr_show, x_dte_patch_arr_show])
            if self.num_template == 1:
                z_rgb_patch_arr_show = self.z_patch_arr[:,:,:3]
                z_dte_patch_arr_show = self.z_patch_arr[:, :, 3:]
                z_patch_arr_show = multi_img_set_visualization([z_rgb_patch_arr_show, z_dte_patch_arr_show])
            else:
                z_patch_arr_list_show = []
                for i in range(self.num_template):
                    temp_z_patch_arr = self.z_patch_arr_list[i]
                    z_rgb_atch_arr_show = temp_z_patch_arr[:, :, :3]
                    z_dte_atch_arr_show = temp_z_patch_arr[:, :, 3:]
                    z_patch_arr_list_show.append(z_rgb_atch_arr_show)
                    z_patch_arr_list_show.append(z_dte_atch_arr_show)
        else:
            image_show = image
            x_patch_arr_show = x_patch_arr
            if self.num_template == 1:
                z_patch_arr_show = self.z_patch_arr
            else:
                z_patch_arr_list_show = self.z_patch_arr_list

        if self.debug == 1:
            # print(f"frame {self.frame_id}'s conf score: {conf_score.item():.15f}")
            self.visdom.register((image_show, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
            self.visdom.register(torch.from_numpy(x_patch_arr_show).permute(2, 0, 1), 'image', 1, 'search_region')
            if self.num_template == 1:
                self.visdom.register(torch.from_numpy(z_patch_arr_show).permute(2, 0, 1), 'image', 1, 'template')
            else:
                z_patch_arr_all_show = multi_img_set_visualization(z_patch_arr_list_show)
                self.visdom.register(torch.from_numpy(z_patch_arr_all_show).permute(2, 0, 1), 'image', 1, 'template')

            # if 'removed_indexes_x' in aux_dict and aux_dict['removed_indexes_x'] is not None:
            if ('removed_indexes_x' in aux_dict and aux_dict['removed_indexes_x'] is not None and
                    any(removed_idx_x is not None for removed_idx_x in aux_dict['removed_indexes_x'])):
                removed_indexes_x = aux_dict['removed_indexes_x']
                removed_indexes_x = [removed_indexes_x_i.cpu().numpy() for removed_indexes_x_i in removed_indexes_x]
                if x_dte_patch_arr_show is not None:
                    masked_search = gen_visualization(image=x_rgb_patch_arr_show,
                                                      mask_indices=removed_indexes_x,
                                                      dte=True,
                                                      image_dte=x_dte_patch_arr_show)
                else:
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_x)
                self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

            if self.use_nlp == True:
                self.visdom.register(info['init_nlp'], 'text', 1, 'text')

            self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
            self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

            while self.pause_mode:
                if self.step:
                    self.step = False
                    break

        elif self.debug == 2:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)

        return {"target_bbox": self.state,
                "best_score": conf_score}

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

    def extract_token_from_nlp_clip(self, nlp):
        if nlp is None:
            nlp_ids = torch.zeros(77, dtype=torch.long)
            nlp_masks = torch.zeros(77, dtype=torch.long)
        else:
            nlp_ids = clip.tokenize(nlp).squeeze(0)
            nlp_masks = (nlp_ids == 0).long()
        return nlp_ids, nlp_masks

def get_tracker_class():
    return SUTRACKCMPVIT
