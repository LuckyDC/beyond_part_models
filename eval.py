import argparse
import os
import sys
import subprocess
import torch

import scipy.io as sio

from utils.evaluation import eval_feature

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("model_path", type=str)

    args = parser.parse_args()
    dataset, fname = args.model_path.split("/")[1:]
    prefix = os.path.splitext(fname)[0]

    # extract feature
    cmd = "python{} extract.py {} {} ".format(sys.version[0], args.gpu, args.model_path)

    subprocess.check_call(cmd.strip().split(" "))

    # evaluation
    query_features_path = 'features/%s/query-%s.mat' % (dataset, prefix)
    gallery_features_path = "features/%s/gallery-%s.mat" % (dataset, prefix)

    assert os.path.exists(query_features_path) and os.path.exists(gallery_features_path)

    query_mat = sio.loadmat(query_features_path)
    gallery_mat = sio.loadmat(gallery_features_path)

    query_features = query_mat["feat"]
    query_ids = query_mat["ids"].squeeze()
    query_cam_ids = query_mat["cam_ids"].squeeze()

    gallery_features = gallery_mat["feat"]
    gallery_ids = gallery_mat["ids"].squeeze()
    gallery_cam_ids = gallery_mat["cam_ids"].squeeze()

    eval_feature(query_features, gallery_features, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids,
                 torch.device("cuda", args.gpu))
