import os, sys, pdb
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from torch import nn
import torchvision.models as models
from utils.visual_mask_TFV import visualize_mask

def eval_test(args, model, dset_loaders, dset_size, phase="test"):
    if args.device == 0:
        device = torch.device('cuda:0')
    elif args.device == 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    labels_all = [] * dset_size[phase]
    preds_all = [] * dset_size[phase]
    model.train(False)
    criterion = nn.CrossEntropyLoss()
    running_loss = .0
    for data in dset_loaders[phase]:

        v00_inputs, v12_inputs, v24_inputs, labels = data
        v00_inputs, v12_inputs, v24_inputs, labels = Variable(v00_inputs.to(device=device)), Variable(v12_inputs.to(device=device)), Variable(v24_inputs.to(device=device)), Variable(labels.to(device=device))

        if args.model_type == 'vit_token_fusion' and args.visit_num == 'v00':
            outputs = model(v00_inputs, args.base_keep_rate)
        elif args.model_type == 'vit_token_fusion' and args.visit_num == 'v12':
            outputs = model(v12_inputs, args.base_keep_rate)
        elif args.model_type == 'vit_token_fusion' and args.visit_num == 'v24':
            outputs = model(v24_inputs, args.base_keep_rate)
        else:
            outputs = model(v00_inputs, v12_inputs, v24_inputs)

        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)
        running_loss += loss.data

        labels_np = labels.cpu().numpy()
        labels = labels_np.tolist()
        labels_all.extend(labels)
        preds_cpu = preds.cpu()
        preds_np = preds_cpu.numpy()
        preds = preds_np.tolist()
        preds_all.extend(preds)

    balanced_acc = balanced_accuracy_score(labels_all, preds_all)
    conf_matrix = confusion_matrix(labels_all, preds_all)
    acc = 1.0*np.trace(conf_matrix)/np.sum(conf_matrix)
    mse = 1.0 * running_loss.cpu().tolist() / dset_size[phase]
    print("In {}: confusion matrix is:\n {}".format(phase, conf_matrix))

    return balanced_acc, acc, mse


def visualise_TFV(args, model, dset_loaders, phase="test"):
    if args.device == 0:
        device = torch.device('cuda:0')
    elif args.device == 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    output_dir = os.path.join(os.path.join(args.model_dir, args.best_model_name), 'visual_TFV')
    visualize_mask(dset_loaders[phase], model, device, output_dir, args.fuse_token, args)


