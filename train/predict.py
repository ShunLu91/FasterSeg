from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils

import numpy as np
from thop import profile

from config_train import config
from datasets import Cityscapes

from utils.init_func import init_weight
from eval import SegEvaluator

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_seg import Network_Multi_Path_Infer as Network

config.save = '../snapshots/predict-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))


def main():
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # config network and criterion ################
    min_kept = int(config.batch_size * config.image_height * config.image_width // (16 * config.gt_down_sampling ** 2))

    # data loader ###########################
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'down_sampling': config.down_sampling}

    # Model #######################################
    models = []
    evaluators = []
    lasts = []
    for idx, arch_idx in enumerate(config.arch_idx):
        if config.load_epoch == "last":
            state = torch.load(os.path.join(config.load_path, "arch_%d.pt" % arch_idx))
        else:
            state = torch.load(os.path.join(config.load_path, "arch_%d_%d.pt" % (arch_idx, int(config.load_epoch))))

        model = Network(
            [state["alpha_%d_0" % arch_idx].detach(), state["alpha_%d_1" % arch_idx].detach(),
             state["alpha_%d_2" % arch_idx].detach()],
            [None, state["beta_%d_1" % arch_idx].detach(), state["beta_%d_2" % arch_idx].detach()],
            [state["ratio_%d_0" % arch_idx].detach(), state["ratio_%d_1" % arch_idx].detach(),
             state["ratio_%d_2" % arch_idx].detach()],
            num_classes=config.num_classes, layers=config.layers, Fch=config.Fch,
            width_mult_list=config.width_mult_list, stem_head_width=config.stem_head_width[idx],
            ignore_skip=arch_idx == 0
        )

        mIoU02 = state["mIoU02"]
        latency02 = state["latency02"]
        obj02 = objective_acc_lat(mIoU02, latency02)
        mIoU12 = state["mIoU12"]
        latency12 = state["latency12"]
        obj12 = objective_acc_lat(mIoU12, latency12)
        if obj02 > obj12:
            last = [2, 0]
        else:
            last = [2, 1]
        lasts.append(last)
        model.build_structure(last)
        # logging.info("net: " + str(model))
        for b in last:
            if len(config.width_mult_list) > 1:
                plot_op(getattr(model, "ops%d" % b), getattr(model, "path%d" % b), width=getattr(model, "widths%d" % b),
                        head_width=config.stem_head_width[idx][1], F_base=config.Fch).savefig(
                    os.path.join(config.save, "ops_%d_%d.png" % (arch_idx, b)), bbox_inches="tight")
            else:
                plot_op(getattr(model, "ops%d" % b), getattr(model, "path%d" % b), F_base=config.Fch).savefig(
                    os.path.join(config.save, "ops_%d_%d.png" % (arch_idx, b)), bbox_inches="tight")
        plot_path_width(model.lasts, model.paths, model.widths).savefig(
            os.path.join(config.save, "path_width%d.png" % arch_idx))
        plot_path_width([2, 1, 0], [model.path2, model.path1, model.path0],
                        [model.widths2, model.widths1, model.widths0]).savefig(
            os.path.join(config.save, "path_width_all%d.png" % arch_idx))
        flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
        logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
        logging.info("ops:" + str(model.ops))
        logging.info("path:" + str(model.paths))
        logging.info("last:" + str(model.lasts))
        model = model.cuda()
        init_weight(model, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

        if arch_idx == 0 and len(config.arch_idx) > 1:
            partial = torch.load(os.path.join(config.teacher_path, "weights%d.pt" % arch_idx))
            state = model.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            model.load_state_dict(state)
        elif config.is_eval:
            partial = torch.load(os.path.join(config.eval_path, "weights%d.pt" % arch_idx))
            state = model.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            model.load_state_dict(state)

        evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                                 config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0,
                                 config=config,
                                 verbose=False, save_path=None, show_image=False, show_prediction=False)
        evaluators.append(evaluator)

        # Optimizer ###################################
        base_lr = config.lr
        if arch_idx == 1 or len(config.arch_idx) == 1:
            # optimize teacher solo OR student (w. distill from teacher)
            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        models.append(model)

    # Cityscapes ###########################################
    if config.is_eval:
        logging.info(config.load_path)
        logging.info(config.eval_path)
        logging.info(config.save)
        with torch.no_grad():
            # validation
            print("[validation...]")
            valid_mIoUs = infer(models, evaluators, logger=None)
            for idx, arch_idx in enumerate(config.arch_idx):
                if arch_idx == 0:
                    logging.info("teacher's valid_mIoU %.3f" % (valid_mIoUs[idx]))
                else:
                    logging.info("student's valid_mIoU %.3f" % (valid_mIoUs[idx]))
        exit(0)


def infer(models, evaluators, logger):
    mIoUs = []
    for model, evaluator in zip(models, evaluators):
        model.eval()
        _, mIoU = evaluator.run_online_multiprocess()
        mIoUs.append(mIoU)
    return mIoUs


if __name__ == '__main__':
    main()
