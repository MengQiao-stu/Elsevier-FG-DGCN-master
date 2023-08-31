import os, math
import numpy as np
import torch
import torch.optim as optim

from lib.utils.meter import Meter
from LSGModel import LSGModel
from lib.dataset import augmentation
from thop import profile, clever_format

@torch.no_grad()
def eval(model, img, gt, device):
    def achievable_segmentation_accuracy(superpixel, label):
        TP = 0
        unique_id = np.unique(superpixel)
        for uid in unique_id:
            mask = superpixel == uid
            label_hist = np.histogram(label[mask])
            maximum_regionsize = label_hist[0].max()
            TP += maximum_regionsize
        return TP / label.size

    model.eval()
    sum_asa = 0

    inputs = img
    labels = gt
    inputs = torch.tensor(inputs)
    inputs = inputs.to(device)
    labels = torch.tensor(labels)
    labels = labels.to(device)

    height, width = inputs.shape[-2:]

    Q, H, feat = model(inputs)

    H = H.reshape(height, width)
    labels = labels.argmax(1).reshape(height, width)

    asa = achievable_segmentation_accuracy(H.to("cpu").detach().numpy(), labels.to("cpu").numpy())
    sum_asa += asa
    model.train()
    return sum_asa


def update_param(img, model, device):
    inputs = img
    inputs = torch.tensor(inputs)
    inputs = inputs.to(device)

    flops, params = profile(model.cuda(), inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.3f")
    print("fops, params:",flops,params)

    Q, H, feat = model(inputs)

    return Q, feat, H


def train(cfg, data, labels):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data = data
    train_samples_gt_onehot = labels

    model = LSGModel(cfg.fdim, cfg.nspix, cfg.bands, cfg.niter).to(device)

    optimizer = optim.Adam(model.parameters(), cfg.lr)

    augment = augmentation.Compose(
        [augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])

    meter = Meter()

    iterations = 0
    while iterations < cfg.train_iter:
        iterations += 1
        img1 = data
        gt1 = train_samples_gt_onehot
        img1 = torch.from_numpy(img1)
        gt1 = torch.from_numpy(gt1)
        img1 = img1.permute(2, 0, 1)
        gt1 = gt1.permute(1, 0)
        HSIgt = np.zeros((1, gt1.shape[0], gt1.shape[1])).astype(np.float32)
        HSIgt[0] = gt1
        HSIimg = np.zeros((1, img1.shape[0], img1.shape[1], img1.shape[2])).astype(np.float32)
        HSIimg[0] = img1

        Q, H, seg = update_param(HSIimg, model, device)

        state = meter.state(f"[{iterations}/{cfg.train_iter}]")
        print(state)

        if iterations == cfg.train_iter:
            break

    return Q, H, seg

def get_A(segments, S, scale, sigma: float):
    segments = segments
    S = S
    superpixel_count = scale
    A = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
    (h, w) = segments.shape
    for i in range(h - 2):
        for j in range(w - 2):
            sub = segments[i:i + 2, j:j + 2]
            sub_max = torch.max(sub)
            sub_max = sub_max.int()
            sub_min = torch.min(sub)
            sub_min = sub_min.int()
            if sub_max != sub_min:
                idx1 = sub_max
                idx2 = sub_min
                if A[idx1, idx2] != 0:
                    continue

                pix1 = S[idx1]
                pix2 = S[idx2]
                diss = torch.exp(-torch.sum(torch.square(pix1 - pix2)) / sigma ** 2)  # 高斯相似度
                A[idx1, idx2] = A[idx2, idx1] = diss

    return A
class LSG(object):
    def __init__(self, data, labels):
        self.data =data
        self.train_samples_gt_onehot =labels
        self.height, self.width, self.bands = data.shape
        self.bands = data.shape[2]
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--root", type=str, help="/path/to/BSR")
        parser.add_argument("--out_dir", default="./log", type=str, help="/path/to/output directory")
        parser.add_argument("--batchsize", default=6, type=int)
        parser.add_argument("--nworkers", default=4, type=int, help="number of threads for CPU parallel")
        parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
        parser.add_argument("--train_iter", default=5, type=int)
        parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
        parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
        parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
        parser.add_argument("--color_scale", default=0.26, type=float)
        parser.add_argument("--pos_scale", default=2.5, type=float)
        parser.add_argument("--compactness", default=1e-5, type=float)
        parser.add_argument("--test_interval", default=10, type=int, help="每多少次输出一下")
        parser.add_argument("--bands", default=self.bands, type=int, help="数据通道数")

        self.args = parser.parse_args()

        os.makedirs(self.args.out_dir, exist_ok=True)



    def simple_superpixel(self,scale):
        self.args.nspix = scale
        Q, S, Seg = train(self.args, self.data, self.train_samples_gt_onehot)
        Q = Q[0]
        Q = Q.permute(1, 0)
        S = S[0]
        S = S.permute(1, 0)
        Seg = Seg[0]
        Seg = torch.reshape(Seg, [self.height, self.width])
        A = get_A(Seg, S, S.shape[0], sigma=10)
        return Q, S, A, Seg