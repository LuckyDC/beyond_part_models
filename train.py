import torch
import logging
import random
import yaml
import pprint
import os

import numpy as np

from torch import optim
from torch import nn

from models import PCBModel
from engine import create_train_engine
from data import get_train_loader

from configs.default import cfg

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.benchmark = True

    # load configuration
    cfg_file = "configs/config.yml"
    cfg.merge_from_file(cfg_file)
    customized_cfg = yaml.load(open(cfg_file, "r"))

    data_cfg = cfg.get(cfg.dataset)
    cfg.root = data_cfg.root
    cfg.train = data_cfg.train
    cfg.num_id = data_cfg.num_id
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    # set logger
    log_dir = "logs/" + cfg.dataset
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + cfg.prefix + ".txt",
                        filemode="a")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.info(pprint.pformat(customized_cfg))

    # data loader
    loader = get_train_loader(root=os.path.join(cfg.root, cfg.train),
                              batch_size=cfg.batch_size,
                              image_size=cfg.image_size,
                              random_crop=cfg.random_crop,
                              random_erase=cfg.random_erase,
                              random_mirror=cfg.random_mirror)

    # model
    model = PCBModel(num_class=cfg.num_id,
                     num_parts=6,
                     bottleneck_dims=256,
                     pool_type="avg")

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    param_groups = [{'params': model.backbone.parameters(), 'lr': cfg.ft_lr},
                    {'params': model.embed.parameters(), 'lr': cfg.new_params_lr},
                    {'params': model.classifier.parameters(), 'lr': cfg.new_params_lr}]

    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.wd)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_step, gamma=0.1)

    # engine
    engine = create_train_engine(model=model,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 lr_scheduler=lr_scheduler,
                                 logger=logger,
                                 device=torch.device("cuda"),
                                 non_blocking=True,
                                 log_period=cfg.log_period,
                                 save_interval=10,
                                 save_dir="checkpoints/" + cfg.dataset,
                                 prefix=cfg.prefix)

    # training
    engine.run(loader, max_epochs=cfg.num_epoch)
