import torch


def process_feat(feat, bs):
    new_feat = []
    for data in feat:
        if isinstance(data, torch.Tensor):
            new_feat.append(data.repeat(bs, 1, 1, 1))
        else:
            new_data_tuple = [
                tensor_data.repeat(bs, 1, 1, 1).clone().cuda() for tensor_data in data
            ]
            new_feat.append(new_data_tuple)
    return tuple(new_feat)


def process_feat_batch(feats, bs):
    new_feats = []
    for feat in feats:
        new_feats.append(process_feat(feat, bs))
    return new_feats
