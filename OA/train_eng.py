import os
import shutil
import torch
from torch import nn, optim
from torch.autograd import Variable
import time
from utils.torch_util import LRScheduler
from eval_eng import eval_test
import pandas as pd

def train_model(args, model, dset_loaders, dset_size):
    if args.device == 0:
        device = torch.device('cuda:0')
    elif args.device == 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    best_model, best_val_acc, best_test_acc, best_num_epoch = None, .0, .0, 0
    best_model_path = os.path.join(args.model_dir, args.best_model_name)
    if os.path.exists(best_model_path):
        shutil.rmtree(best_model_path)
    os.makedirs(best_model_path)

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, momentum=0.9)
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    # print('-'*10+'Training'+'-'*10)
    print('Device id:{} Initial lr:{}  Optimizer:{} num_class:{}'.format(
        args.device, args.lr, args.optim, args.num_class))

    lr_scheduler = LRScheduler(args.lr, args.lr_decay_epoch)
    results = {'val_loss': [], 'val_acc': [], 'test_acc': [], 'macro_test_acc': []}

    for epoch in range(args.num_epoch):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, args.num_epoch))
        # scaler = torch.cuda.amp.GradScaler()
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)
            else:
                model.train(False)

            running_loss, running_corrects = .0, .0

            for data in dset_loaders[phase]:
                v00_inputs, v12_inputs, v24_inputs, labels = data
                v00_inputs, v12_inputs, v24_inputs, labels = Variable(v00_inputs.to(device=device)), Variable(v12_inputs.to(device=device)), Variable(v24_inputs.to(device=device)), Variable(labels.to(device=device))
                optimizer.zero_grad()
                if args.model_type == 'vit_token_fusion' and args.visit_num == 'v00':
                    outputs = model(v00_inputs, args.base_keep_rate)
                elif args.model_type == 'vit_token_fusion' and args.visit_num == 'v12':
                    outputs = model(v12_inputs, args.base_keep_rate)
                elif args.model_type == 'vit_token_fusion' and args.visit_num == 'v24':
                    outputs = model(v24_inputs, args.base_keep_rate)
                else:

                    outputs = model(v00_inputs, v12_inputs, v24_inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if phase == 'train':
                    # scaler.scale(loss).backward()
                    # scaler.step(optimizer)
                    # scaler.update()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = 1.0 * running_loss.cpu().tolist() / dset_size[phase]
            epoch_acc = 1.0 * running_corrects.cpu().tolist() / dset_size[phase]
            elapse_time = time.time() - since
            print("In {}, Number case:{} Loss:{:.4f} Acc:{:.4f} Time:{}".format(
                phase, dset_size[phase], epoch_loss, epoch_acc, elapse_time))

            if phase == "val":
                test_balance_acc, test_acc, test_mse = eval_test(args, model, dset_loaders, dset_size, "test")
                print("---On test_set: acc is {:.3f}, balance acc is {:.3f}, mse is {:.3f}".format(test_acc, test_balance_acc, test_mse))
                results['val_loss'].append(epoch_loss)
                results['val_acc'].append(epoch_acc)
                results['test_acc'].append(test_acc)
                results['macro_test_acc'].append(test_balance_acc)
                data_frame = pd.DataFrame(data=results, index=range(0, epoch + 1))
                data_frame.to_csv(os.path.join(best_model_path, 'log.csv'), index_label='epoch')
                if epoch_acc > best_val_acc or test_acc > best_test_acc:
                    if epoch_acc > best_val_acc:
                        best_val_acc = epoch_acc
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                    best_num_epoch = epoch
                    val_metric_str = str(epoch).zfill(3) + '-' + str(round(epoch_acc, 3))
                    test_metric_str = "-" + str(round(test_acc, 3)) + "-" + str(round(test_mse, 3)) + ".pth"
                    args.best_model_path = os.path.join(best_model_path, val_metric_str + test_metric_str)
                    torch.save(model.state_dict(), args.best_model_path)
                    # model.to(device=device)
                maybe_visualize(model, dset_loaders["val"], args, epoch)

    print('=' * 80)
    print('Validation best_acc: {}  best_num_epoch: {}'.format(best_val_acc, best_num_epoch))
# this code for attention map
def maybe_visualize(model, dataloader, args, epoch, output_dir="vis_results"):
    if epoch != 0 and epoch % 10 != 0:
        return
    # if epoch % 2 != 0:
    #     return

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from torchvision.utils import save_image
    import numpy as np, os, cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    cnn_model = model.cnn_model
    cnn_model = cnn_model.cuda() if torch.cuda.is_available() else cnn_model.cpu()

    target_layer = cnn_model.features[35]  # VGG19 last conv
    os.makedirs(output_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        v00, v12, v24, labels = [x.to(device) for x in batch]
        v00.requires_grad_()
        v12.requires_grad_()
        v24.requires_grad_()

        # for get cnn feature
        v00_f = cnn_model(v00)
        v12_f = cnn_model(v12)
        v24_f = cnn_model(v24)
        sequence = torch.stack([v00_f, v12_f, v24_f], dim=1)
        weights = model.time_attn(sequence)
        weights = torch.softmax(weights, dim=1)

        # only attention last batch
        for j in range(v00.size(0)):
            sample_idx = i * dataloader.batch_size + j
            att = weights[j].squeeze().detach().cpu().numpy()

            for t_idx, (v_tensor, tag) in enumerate(zip([v00, v12, v24], ['v00', 'v12', 'v24'])):
                input_img = v_tensor[j].unsqueeze(0)

                with GradCAM(model=cnn_model, target_layers=[target_layer]) as cam:
                    grayscale_cam = cam(input_tensor=input_img, targets=None)[0]

                rgb_img = input_img.squeeze().detach().cpu().permute(1, 2, 0).numpy()
                rgb_img = np.clip((rgb_img * 0.229 + 0.485), 0, 1)
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                # save formation
                epoch_dir = os.path.join(output_dir, f"epoch_{epoch}", f"sample_{sample_idx}")
                os.makedirs(epoch_dir, exist_ok=True)

                save_image(input_img, f"{epoch_dir}/{tag}_input.jpg")
                cv2.imwrite(f"{epoch_dir}/{tag}_cam.jpg", cam_bgr)

            # save attention
            np.save(f"{epoch_dir}/attention_weights.npy", att)

        break

