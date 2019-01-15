import torch
import logging
import random
import yaml
import pprint
import os

import numpy as np

from torch import optim
from torch import nn
from torch.nn import init

from models import PCBModel
from engine import get_trainer
from data import get_test_loader
from data import get_train_loader

from utils.initializer import Initializer

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
    for k, v in data_cfg.items():
        cfg[k] = v

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
    train_loader = get_train_loader(root=os.path.join(cfg.root, cfg.train),
                                    batch_size=cfg.batch_size,
                                    image_size=cfg.image_size,
                                    random_crop=cfg.random_crop,
                                    random_erase=cfg.random_erase,
                                    random_mirror=cfg.random_mirror,
                                    num_workers=4)

    query_loader = None
    gallery_loader = None
    if cfg.validate_interval > 0:
        query_loader = get_test_loader(root=os.path.join(cfg.root, cfg.query),
                                       batch_size=512,
                                       image_size=cfg.image_size,
                                       num_workers=4)

        gallery_loader = get_test_loader(root=os.path.join(cfg.root, cfg.gallery),
                                         batch_size=512,
                                         image_size=cfg.image_size,
                                         num_workers=4)

    # model
    model = PCBModel(num_class=cfg.num_id,
                     num_parts=cfg.num_parts,
                     bottleneck_dims=cfg.bottleneck_dims,
                     pool_type=cfg.pool_type,
                     share_embed=cfg.share_embed)

    # initialize
    init = Initializer(init.normal_, {"std": 0.001})
    model.embed.apply(init)
    model.classifier.apply(init)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    param_groups = [{'params': model.backbone.parameters(), 'lr': cfg.ft_lr},
                    {'params': model.embed.parameters(), 'lr': cfg.new_params_lr},
                    {'params': model.classifier.parameters(), 'lr': cfg.new_params_lr}]

    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.wd)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_step, gamma=0.1)

    # engine
    engine = get_trainer(model=model,
                         optimizer=optimizer,
                         criterion=criterion,
                         lr_scheduler=lr_scheduler,
                         logger=logger,
                         device=torch.device("cuda"),
                         non_blocking=True,
                         log_period=cfg.log_period,
                         save_interval=10,
                         save_dir="checkpoints/" + cfg.dataset,
                         prefix=cfg.prefix,
                         validate_interval=cfg.validate_interval,
                         query_loader=query_loader,
                         gallery_loader=gallery_loader)

    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)

    print("Model saved at checkpoints/{}/{}_model_{}.pth".format(cfg.dataset, cfg.prefix, cfg.num_epoch))
