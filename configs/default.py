from yacs.config import CfgNode

cfg = CfgNode()

cfg.prefix = "baseline"

# setting for loader
cfg.batch_size = 64
cfg.gpu = 0

# settings for optimizer
cfg.ft_lr = 0.01
cfg.new_params_lr = 0.02
cfg.wd = 5e-4
cfg.lr_step = [25, 50]
cfg.num_epoch = 60

# settings for loader
cfg.dataset = "duke"
cfg.image_size = (384, 128)

# augmentation
cfg.random_mirror = True
cfg.random_crop = False
cfg.random_erase = False

# settings for base architecture
cfg.num_parts = 6
cfg.bottleneck_dims = 256
cfg.pool_type = "avg"
cfg.share_embed = False

cfg.dataset = "duke"

# logging
cfg.validate_interval = -1
cfg.log_period = 50

# config for dataset
cfg.market = CfgNode()
cfg.market.num_id = 751
cfg.market.root = "/home/chuanchen_luo/data/Market-1501-v15.09.15"
cfg.market.train = "bounding_box_train"
cfg.market.query = "query"
cfg.market.gallery = "bounding_box_test"

cfg.duke = CfgNode()
cfg.duke.num_id = 702
cfg.duke.root = "/home/chuanchen_luo/data/DukeMTMC-reID"
cfg.duke.train = "bounding_box_train"
cfg.duke.query = "query"
cfg.duke.gallery = "bounding_box_test"

cfg.cuhk = CfgNode()
cfg.cuhk.num_id = 767
cfg.cuhk.root = " /home/chuanchen_luo/data/cuhk03-np/labeled"
cfg.cuhk.train = "bounding_box_train"
cfg.cuhk.query = "query"
cfg.cuhk.gallery = "bounding_box_test"

cfg.msmt = CfgNode()
cfg.msmt.num_id = 1041
cfg.msmt.root = "/home/chuanchen_luo/data/MSMT17_V1"
cfg.msmt.train = "train"
cfg.msmt.query = "list_query.txt"
cfg.msmt.gallery = "list_gallery.txt"
