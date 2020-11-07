import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn import preprocessing
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import pretrain_vgg
import resnet
import argparse
import pickle
import cv2
from visualize import errorMap
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
from utils.tools import one_hot, one_hot_forMap
import sys
import dataloaders
import model_weightSample
from utils.tools import draw_errorMap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 20

""" set parameters """
parser = argparse.ArgumentParser()
parser.add_argument('--kmeans', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--data', type=str, default='bottle')
parser.add_argument('--type', type=str, default='good')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--model', type=str, default='vgg19')
parser.add_argument('--train_batch', type=int, default=32)
args = parser.parse_args()

out = args.kmeans
test_good_path = "./dataset/{}/test/good_resize".format(args.data)

if (args.type == 'good'):
    test_path = test_good_path
    test_label_name = "label/{}/{}/test/good_{}_{}.pth".format(
        str(args.model),
        args.data,
        str(out),
        str(args.batch)
    )
    defect_gt_path = "dataset/{}/ground_truth/broken_small_resize/".format(args.data)


else:
    test_path = "./dataset/{}/test/{}_resize".format(args.data, args.type)
    test_label_name = "label/{}/{}/test/{}_{}_{}.pth".format(
        str(args.model),
        args.data,
        args.type,
        str(out),
        str(args.batch)
    )
    defect_gt_path = "dataset/{}/ground_truth/{}_resize/".format(args.data, args.type)
    test_label = torch.tensor(torch.load(test_label_name))


# def eval_feature(pretrain_model, epoch, model, test_loader, mask_loader):
#     global kmeans
#     global mask_dir

#     model.eval()
#     pretrain_model.eval()
#     split_size = 8

#     with torch.no_grad():
        
#         img_feature = []
#         total_gt = []
#         total_idx = []

#         for ((idx, img), (idx2, img2)) in zip(test_loader, mask_loader):
#             img = img.to(device)
#             idx = idx[0].item()

#             print(f'eval phase: img idx={idx}')

#             value_feature = []
#             value_label = []
#             label_idx = []
#             label_gt = []

#             for i in range(16):
#                 xs = []
#                 ys = []

#                 crop_list = []
#                 loss = 0.0
#                 loss_alpha = 0.0

#                 for j in range(16):
#                     crop_img = img[:, :, i*64:i*64+64, j*64:j*64+64].to(device)
#                     crop_output = pretrain_model(crop_img)
#                     """ flatten the dimension of H and W """
#                     out_ = crop_output.flatten(1,2).flatten(1,2)
#                     out = pca.transform(out_.detach().cpu().numpy())
#                     out = pil_to_tensor(out).squeeze().to(device)
#                     crop_list.append(out)

#                     mask = torch.ones(1, 1, 1024, 1024)
#                     mask[:, :, i*64:i*64+64, j*64:j*64+64] = 0
#                     mask = mask.to(device)
#                     x = img * mask
#                     x = torch.cat((x, mask), 1)
#                     label = test_label[idx][i*16+j].to(device)
                        
#                     # label = test_label[idx*1024 + i*32+j].to(device, dtype=torch.float)
#                     # print(label)
                   
#                     xs.append(x)
#                     ys.append(label)
            
#                 x = torch.cat(xs, 0)
#                 y = torch.stack(ys).squeeze().to(device)

#                 output = model(x)
#                 # y_ = output.argmax(-1).detach()
#                 y_ = output.argmax(-1).detach().cpu().numpy()

#                 for k in range(16):
#                     label_idx.append(y_[k])
#                     label_gt.append(y[k].item())
#                     output_center = kmeans.cluster_centers_[y_[k]]
#                     output_center = np.reshape(output_center, (1, -1))
#                     output_center = pil_to_tensor(output_center).to(device)
#                     output_center = torch.squeeze(output_center)

#                     if y_[k] == y[k].item():
#                         isWrongLabel = 0
#                     else:
#                         isWrongLabel = 1

#                     diff = isWrongLabel * nn.MSELoss()(output_center, crop_list[k])
#                     value_feature.append(diff.item())

#                     writer.add_scalar('test_loss', diff.item(), eval_fea_count)
#                     # print(f'Testing i={i} j={k} loss={diff.item()}')
                    
#                     eval_fea_count += 1

#             total_gt.append(label_gt)
#             total_idx.append(label_idx)
#             img_feature.append(value_feature)


#     img_feature = np.array(img_feature).reshape((len(test_loader), -1))
#     total_gt = np.array(total_gt).reshape((len(test_loader), -1))
#     total_idx = np.array(total_idx).reshape((len(test_loader), -1))

    # return img_feature, total_gt, total_idx


if __name__ == "__main__":

    test_dataset = dataloaders.MvtecLoader(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    mask_dataset = dataloaders.MaskLoader(defect_gt_path)
    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)

    model = model_weightSample.scratch_model
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('models/{}/{}/exp9_128_10.ckpt'.format(
        args.model, 
        args.data
        )   
    ))
        
    model = model.to(device)

    img_feature, total_gt, total_idx = model_weightSample.eval_feature(9, model, test_loader, mask_loader)
    draw_errorMap(img_feature, total_gt, total_idx, 9, model, test_loader, mask_loader, args.data, args.type)

