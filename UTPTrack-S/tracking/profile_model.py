import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
from lib.models.sutrack.sutrack import build_sutrack
from lib.models.sutrackcmp.sutrackcmp import build_sutrackcmp
from lib.models.sutrackcmpvit.sutrackcmpvit import build_sutrackcmpvit
from lib.models.sutrack.compression.ce import generate_mask_cond
import clip
from lib.models.sutrack.clip import TextEncoder


def get_num_channels(cfg):
    if "itpnb" in cfg.MODEL.ENCODER.TYPE:
        num_channels = 512
    elif "itpnl" in cfg.MODEL.ENCODER.TYPE:
        num_channels = 768
    elif "itpns" in cfg.MODEL.ENCODER.TYPE or "itpnt" in cfg.MODEL.ENCODER.TYPE:
        num_channels = 384
    else:
        num_channels = 512
    return num_channels

def extract_token_from_nlp_clip(nlp):
    if nlp is None:
        nlp_ids = torch.zeros(77, dtype=torch.long)
        nlp_masks = torch.zeros(77, dtype=torch.long)
    else:
        nlp_ids = clip.tokenize(nlp).squeeze(0)
        nlp_masks = (nlp_ids == 0).long()
    return nlp_ids, nlp_masks

def evaluate_sutarck(cfg, model, template_list, search_list, template_anno_list, text_src, task_index):
    class PofileSutrack(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
            text_src = self.model.forward_textencoder(text_data=text_src)
            enc_opt, aux_dict = self.model.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
            out_dict = self.model.forward_decoder(feature=enc_opt)
            return aux_dict

    class ProfileTextEncoder(TextEncoder):
        def forward(self, text_data):
            text_data = text_data.unsqueeze(0)
            text_src = self.clip.encode_text(text_data).type(self.dtype)
            text_src = self.text_proj(text_src)
            text_src = text_src.unsqueeze(1)
            return text_src


    sutrack_model = PofileSutrack(model)
    text_encoder = ProfileTextEncoder(type=cfg.MODEL.TEXT_ENCODER.TYPE, out_channel=get_num_channels(cfg)).to(device)
    sutrack_model.text_encoder = text_encoder
    macs1, params1 = profile(text_encoder, inputs=text_src)
    macs2, params2 = profile(sutrack_model, inputs=(template_list, search_list, template_anno_list, text_src, task_index))
    macs, params = clever_format([macs2-macs1, params2-params1], "%.3f")
    print("MACs:", macs)
    print("Params:", params)

    T_w = 500
    T_t = 1000
    print("Testing speed ...")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(T_w):
            aux_dict = sutrack_model.forward(template_list, search_list, template_anno_list, text_src, task_index)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(T_t):
            aux_dict = sutrack_model.forward(template_list, search_list, template_anno_list, text_src, task_index)
        torch.cuda.synchronize()
        end = time.time()

    avg_time = (end - start) / T_t
    fps = 1. / avg_time
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    print("Latency (avg): %.2f ms" % (avg_time * 1000))
    print("FPS: %.2f" % fps)
    print("Max CUDA Memory Usage: %.2f MB" % max_mem)

    token_counts = aux_dict.get("token_count_per_block", None)
    if token_counts:
        print("Token count per Stage 3 transformer block:")
        tokens_sum = 0
        for idx, token_count in enumerate(token_counts):
            if idx == 0:
                print(f"  Input Visual Tokens = {token_count - 2:.2f}")  # 需要减去cls_token，text好像初始化不对，没传进去
            else:
                print(f"  Attn Block {idx-1:02d}: Tokens = {token_count-2:.2f}") # 需要减去cls_token，text好像初始化不对，没传进去
                tokens_sum += (token_count-2)
        print("Avg Vis Tok: ", tokens_sum / (len(token_counts)-1))
        print("Cmp Vis Tok: ", token_counts[-1]-2)  # 最后一层
    else:
        print("Token statistics unavailable.")

def evaluate_sutrackcmp(cfg, model, template_list, search_list, template_anno_list, text_src, task_index):
    bs = template_list[0].shape[0]
    device = template_list[0].device
    ce_template_mask = generate_mask_cond(cfg, bs, device)

    class PofileSutrackCMP(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
            text_src = self.model.forward_textencoder(text_data=text_src)
            enc_opt, aux_dict = self.model.forward_encoder(template_list=template_list,
                                                           search_list=search_list,
                                                           template_anno_list=template_anno_list,
                                                           text_src=text_src,
                                                           task_index=task_index,
                                                           ce_template_mask=ce_template_mask,
                                                           ce_keep_rate=None,
                                                           cm_back_rate=None,
                                                           te_keep_rate=None,
                                                           tm_back_rate=None,
                                                           )
            out_dict = self.model.forward_decoder(feature=enc_opt)
            return aux_dict

    class ProfileTextEncoder(TextEncoder):
        def forward(self, text_data):
            text_data = text_data.unsqueeze(0)
            text_src = self.clip.encode_text(text_data).type(self.dtype)
            text_src = self.text_proj(text_src)
            text_src = text_src.unsqueeze(1)
            return text_src

    sutrack_model = PofileSutrackCMP(model)
    text_encoder = ProfileTextEncoder(type=cfg.MODEL.TEXT_ENCODER.TYPE, out_channel=get_num_channels(cfg)).to(device)
    sutrack_model.text_encoder = text_encoder
    macs1, params1 = profile(text_encoder, inputs=text_src)
    macs2, params2 = profile(sutrack_model, inputs=(template_list, search_list, template_anno_list, text_src, task_index))
    macs, params = clever_format([macs2-macs1, params2-params1], "%.3f")
    print("MACs:", macs)
    print("Params:", params)

    T_w = 500
    T_t = 1000
    print("Testing speed ...")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(T_w):
            aux_dict = sutrack_model.forward(template_list, search_list, template_anno_list, text_src, task_index)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(T_t):
            aux_dict = sutrack_model.forward(template_list, search_list, template_anno_list, text_src, task_index)
        torch.cuda.synchronize()
        end = time.time()

    avg_time = (end - start) / T_t
    fps = 1. / avg_time
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    print("Latency (avg): %.2f ms" % (avg_time * 1000))
    print("FPS: %.2f" % fps)
    print("Max CUDA Memory Usage: %.2f MB" % max_mem)

    token_counts = aux_dict.get("token_count_per_block", None)
    if token_counts:
        print("Token count per Stage 3 transformer block:")
        tokens_sum = 0
        for idx, token_count in enumerate(token_counts):
            if idx == 0:
                print(f"  Input Visual Tokens = {token_count - 2:.2f}")  # 需要减去cls_token，text好像初始化不对，没传进去
            else:
                print(f"  Attn Block {idx-1:02d}: Tokens = {token_count-2:.2f}") # 需要减去cls_token，text好像初始化不对，没传进去
                tokens_sum += (token_count-2)
        print("Avg Vis Tok: ", tokens_sum / (len(token_counts)-1))
        print("Cmp Vis Tok: ", token_counts[-1]-2)  # 最后一层
    else:
        print("Token statistics unavailable.")

def evaluate_sutrackcmpvit(cfg, model, template_list, search_list, template_anno_list, text_src, task_index):
    class PofileSutrackCMPVIT(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
            text_src = self.model.forward_textencoder(text_data=text_src)
            enc_opt, aux_dict = self.model.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
            out_dict = self.model.forward_decoder(feature=enc_opt)
            return aux_dict

    class ProfileTextEncoder(TextEncoder):
        def forward(self, text_data):
            text_data = text_data.unsqueeze(0)
            text_src = self.clip.encode_text(text_data).type(self.dtype)
            text_src = self.text_proj(text_src)
            text_src = text_src.unsqueeze(1)
            return text_src

    sutrack_model = PofileSutrackCMPVIT(model)
    text_encoder = ProfileTextEncoder(type=cfg.MODEL.TEXT_ENCODER.TYPE, out_channel=get_num_channels(cfg)).to(device)
    sutrack_model.text_encoder = text_encoder
    macs1, params1 = profile(text_encoder, inputs=text_src)
    macs2, params2 = profile(sutrack_model, inputs=(template_list, search_list, template_anno_list, text_src, task_index))
    macs, params = clever_format([macs2-macs1, params2-params1], "%.3f")
    print("MACs:", macs)
    print("Params:", params)

    T_w = 5 # 500
    T_t = 10    # 1000
    print("Testing speed ...")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(T_w):
            aux_dict = sutrack_model.forward(template_list, search_list, template_anno_list, text_src, task_index)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(T_t):
            aux_dict = sutrack_model.forward(template_list, search_list, template_anno_list, text_src, task_index)
        torch.cuda.synchronize()
        end = time.time()

    avg_time = (end - start) / T_t
    fps = 1. / avg_time
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    print("Latency (avg): %.2f ms" % (avg_time * 1000))
    print("FPS: %.2f" % fps)
    print("Max CUDA Memory Usage: %.2f MB" % max_mem)

    token_counts = aux_dict.get("token_count_per_block", None)
    if token_counts:
        print("Token count per Stage 3 transformer block:")
        tokens_sum = 0
        for idx, token_count in enumerate(token_counts):
            if idx == 0:
                print(f"  Input Visual Tokens = {token_count - 2:.2f}")  # 需要减去cls_token，text好像初始化不对，没传进去
            else:
                print(f"  Attn Block {idx-1:02d}: Tokens = {token_count-2:.2f}") # 需要减去cls_token，text好像初始化不对，没传进去
                tokens_sum += (token_count-2)
        print("Avg Vis Tok: ", tokens_sum / (len(token_counts)-1))
        print("Cmp Vis Tok: ", token_counts[-1]-2)  # 最后一层
    else:
        print("Token statistics unavailable.")

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='sutrackcmpvit', choices=['sutrack, sutrackcmp, sutrackcmpvit'], help='training script name')
    parser.add_argument('--config', type=str, default='dyvit_b384', help='yaml configure file name')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)

    args = parse_args()
    # yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    yaml_fname = '/home/wuhao/workspace/SUTrack/experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)

    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    # template / search: (B, 3, H, W)
    template = torch.randn(bs, 6, z_sz, z_sz).to(device)
    search = torch.randn(bs, 6, x_sz, x_sz).to(device)
    init_nlp = None
    text_data, _ = extract_token_from_nlp_clip(init_nlp)
    text_data = text_data.unsqueeze(0).to(device)
    task_index = None

    if cfg.TEST.MULTI_MODAL_VISION and template.shape[1] == 3:
        template = torch.cat([template, template], dim=1)
        search = torch.cat([search, search], dim=1)

    # 模拟 batch 和 annotation
    template_list = [template] * cfg.DATA.TEMPLATE.NUMBER
    search_list = [search] * cfg.DATA.SEARCH.NUMBER
    template_anno_list = [torch.tensor([[0.4, 0.4, 0.2, 0.2]], dtype=torch.float32).to(device)] * cfg.DATA.TEMPLATE.NUMBER

    if args.script == "sutrack":
        model = build_sutrack(cfg)
        model = model.to(device)
        model.eval()
        evaluate_sutarck(cfg, model, template_list, search_list, template_anno_list, text_data, task_index)
    elif args.script == "sutrackcmp":
        model = build_sutrackcmp(cfg)
        model = model.to(device)
        model.eval()
        evaluate_sutrackcmp(cfg, model, template_list, search_list, template_anno_list, text_data, task_index)
    elif args.script == "sutrackcmpvit":
        model = build_sutrackcmpvit(cfg)
        model = model.to(device)
        model.eval()
        evaluate_sutrackcmpvit(cfg, model, template_list, search_list, template_anno_list, text_data, task_index)
    else:
        raise NotImplementedError