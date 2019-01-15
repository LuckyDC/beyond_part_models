import os
import argparse
import torch

import numpy as np
import scipy.io as sio

from configs.default import cfg
from data import get_test_loader
from models import PCBModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("model_path", type=str)

    args = parser.parse_args()
    model_path = args.model_path
    dataset, fname = model_path.split("/")[1:]
    prefix = os.path.splitext(fname)[0]

    dataset_config = cfg.get(dataset)
    image_size = (384, 128)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    state_dict = torch.load(model_path, map_location=torch.device("cuda"))

    num_parts = 0
    for k in state_dict.keys():
        if "embed" in k:
            num_parts = max(num_parts, int(k.split(".")[1]))
    bottleneck_dims = state_dict["embed.0.0.weight"].size(0)
    model = PCBModel(num_parts=num_parts + 1, bottleneck_dims=bottleneck_dims)
    model.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        model.cuda()

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # extract query feature
    query = get_test_loader(root=os.path.join(dataset_config.root, dataset_config.query),
                            batch_size=512,
                            image_size=image_size,
                            num_workers=16)

    query_feat = []
    query_label = []
    query_cam_id = []
    for data, label, cam_id in query:
        feat = model(data.cuda(non_blocking=True))

        query_feat.append(feat.data.cpu().numpy())
        query_label.append(label.data.cpu().numpy())
        query_cam_id.append(cam_id.data.cpu().numpy())

    query_feat = np.concatenate(query_feat, axis=0)
    query_label = np.concatenate(query_label, axis=0)
    query_cam_id = np.concatenate(query_cam_id, axis=0)
    print(query_feat.shape)

    save_name = "features/{}/query-{}.mat".format(dataset, prefix)
    sio.savemat(save_name, {"feat": query_feat, "ids": query_label, "cam_ids": query_cam_id})

    # extract gallery feature
    gallery = get_test_loader(root=os.path.join(dataset_config.root, dataset_config.gallery),
                              batch_size=512,
                              image_size=image_size,
                              num_workers=16)

    gallery_feat = []
    gallery_label = []
    gallery_cam_id = []
    for data, label, cam_id in gallery:
        feat = model(data.cuda(non_blocking=True))

        gallery_feat.append(feat.data.cpu().numpy())
        gallery_label.append(label)
        gallery_cam_id.append(cam_id)

    gallery_feat = np.concatenate(gallery_feat, axis=0)
    gallery_label = np.concatenate(gallery_label, axis=0)
    gallery_cam_id = np.concatenate(gallery_cam_id, axis=0)
    print(gallery_feat.shape)

    save_name = "features/{}/gallery-{}.mat".format(dataset, prefix)
    sio.savemat(save_name, {"feat": gallery_feat, "ids": gallery_label, "cam_ids": gallery_cam_id})
