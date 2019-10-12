import torch
# import torch.nn.functional as F
from unet import UNet
from dataloader import ImgPoseDataset, ToTensor, Rescale

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
# import torch.autograd.Function as Function
import numpy as np

import torchvision
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import os
from network_modules import UNetInnerProd, innerProdLoss
import time

from options import LossOptions, UnetOptions

from log import visualize_to_tensorboard, scale_to_tensorboard

import json

def split_train_val(img_pose_dataset, validation_split, batch_size=1):
    ### Create train and val set using a fixed seed
    shuffle_dataset = True
    validation_split = .1
    random_seed= 42
    dataset_size = len(img_pose_dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers from the indices:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    data_loader_train = DataLoader(img_pose_dataset, batch_size=batch_size, sampler=train_sampler)
    data_loader_val = DataLoader(img_pose_dataset, batch_size=batch_size, sampler=valid_sampler)

    return data_loader_train, data_loader_val

def topk_coord_and_feat(feature_norm, k=3000):
    ### feature_norm shape: b*h*w
    ### mask: b*h*w
    sample_aug = 5

    b = feature_norm.shape[0]

    feature_norm[:,:5] = 0
    feature_norm[:,-5:] = 0
    feature_norm[:,:,:5] = 0
    feature_norm[:,:,-5:] = 0    

    feature_n_flat = feature_norm.reshape(feature_norm.shape[0], -1)
    topk = torch.topk(feature_n_flat, sample_aug*k)

    mask = torch.zeros_like(feature_n_flat).to(dtype=torch.bool)
    for i in range(b):
        perm = torch.randperm(sample_aug*k)
        idx_sample = perm[:k]
        topk_sample = topk.indices[i][idx_sample]
        mask[i, topk_sample] = True
    mask = mask.reshape(feature_norm.shape[0], feature_norm.shape[1], feature_norm.shape[2])

    return mask

def feat_to_np(feature):
    ### feature: b*c*h*w
    b = feature.shape[0]
    feat_np = [None]*b
    for i in range(b):
        feat_np[i] = feature[i].permute(1,2,0).cpu().numpy()
    return feat_np

def mask_to_np(mask):
    ### mask: b*h*w
    b = mask.shape[0]
    mask_np = [None]*b
    for i in range(b):
        mask_np[i] = mask[i].cpu().numpy()
    return mask_np

def reorganize_inputs(sample_batch):
    inputs = {}
    inputs[0] = {}
    inputs[1] = {}
    for i in range(2):
        inputs[i]['img'] = sample_batch['image {}'.format(i+1)]
        inputs[i]['img_raw'] = sample_batch['image {} raw'.format(i+1)]
        inputs[i]['depth'] = sample_batch['depth {}'.format(i+1)]
        inputs[i]['idepth'] = sample_batch['idepth {}'.format(i+1)]
        inputs[i]['gray'] = sample_batch['gray {}'.format(i+1)]
    inputs['rela_pose'] = sample_batch['rela_pose']
    inputs['rela_euler'] = sample_batch['rela_euler']
    inputs['imgname 1'] = sample_batch['imgname 1']
    return inputs

def main():

    print('Cuda available?', torch.cuda.is_available())
    device = torch.device('cuda:1' if torch.cuda.is_available()else 'cpu')

    unet_options = UnetOptions()
    unet_options.setto()
    loss_options = LossOptions(unet_options)
    
    # from segmentation_models_pytorch.encoders import get_preprocessing_fn
    # preprocess_input_fn = get_preprocessing_fn('resnet34', pretrained='imagenet')
    preprocess_input_fn = None # using the above preprocessing function will make the rgb not in range [0, 1] and cause the rgb_to_hsv function to fail

    model_overall = UNetInnerProd(unet_options=unet_options, device=device, loss_options=loss_options )
    lr = loss_options.lr
    optim = torch.optim.Adam(model_overall.model_UNet.parameters(), lr=lr) #cefault 1e-3

    ### Create dataset
    ### for TUM dataset, you need to run associate.py on a folder of unzipped folders of TUM data sequences
    img_pose_dataset = ImgPoseDataset(
        root_dir = loss_options.root_dir, 
        transform=transforms.Compose([Rescale(output_size=(loss_options.height, loss_options.width), post_fn=preprocess_input_fn), ToTensor(device=device) ]), 
        folders=loss_options.folders )
    
    if not unet_options.run_eval:
        data_loader_train, data_loader_val = split_train_val(img_pose_dataset, 0.1, batch_size=loss_options.batch_size)
    else:
        data_loader_train = DataLoader(img_pose_dataset, batch_size=loss_options.batch_size, shuffle=False)

    if unet_options.run_eval:
        checkpoint = torch.load('saved_models/epoch00_6000.pth')
        model_overall.model_UNet.load_state_dict(checkpoint['model_state_dict'])
        model_overall.model_loss.load_state_dict(checkpoint['loss_model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        model_overall.eval()
        model_overall.model_UNet.eval()
        model_overall.model_loss.eval()

    if unet_options.continue_train: 
        save_model_folder = os.path.join('saved_models', 'wo_color_Mon Sep 30 15:21:22 2019')
        checkpoint = torch.load(os.path.join(save_model_folder, 'epoch00_3000.pth') )
        model_overall.model_UNet.load_state_dict(checkpoint['model_state_dict'])
        model_overall.model_loss.load_state_dict(checkpoint['loss_model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])


    epochs = 1

    print('going into loop')
    iter_overall = 0
    lr_change_time = 0
    start_time = time.time()
    ctime = time.ctime()
    overall_time = 0

    ## Configuring path for outputs
    if loss_options.color_in_cost:
        mode_folder = 'with_color'
    else:
        mode_folder = 'wo_color'
    ## SummaryWriter
    writers = {}
    for mode in ["train", "val"]:
        writers[mode] = SummaryWriter(os.path.join('runs', mode_folder + '_' + ctime + '_' + mode ))
    ## Save model if training, output features if evaluating
    if not unet_options.continue_train: 
        if unet_options.run_eval:
            output_folder = os.path.join('feature_output', mode_folder + '_' + ctime )
            os.mkdir(output_folder)
        else:
            save_model_folder = os.path.join('saved_models', mode_folder + '_' + ctime  )
            os.mkdir(save_model_folder)
    ## Save the options to a json file
    options = {}
    dict_unet = dict(unet_options.__dict__)
    dict_loss = dict(loss_options.__dict__)
    del dict_loss['opt_unet']
    options['unet_options'] = dict_unet
    options['loss_options'] = dict_loss

    option_file_path = os.path.join('runs', mode_folder + '_' + ctime + '.json' )
    with open(option_file_path, 'w') as option_file:
        json.dump(options, option_file, indent=2)
            
    for i_epoch in range(epochs):
        print('entering epoch', i_epoch) 
        for i_batch, sample_batch in enumerate(data_loader_train):
            if iter_overall > 20000:
                break

            ## Create a dictionary of inputs
            inputs = reorganize_inputs(sample_batch)
            
            ## Mode selection
            if unet_options.run_eval:
                visualize_mode = True
            else:
                visualize_mode = (iter_overall <= 1000 and iter_overall % 100 == 0) or (iter_overall > 1000 and iter_overall % 500 == 0)
            eval_mode = iter_overall % 100 == 0

            if visualize_mode and loss_options.width <= 128:
                model_overall.opt_loss.visualize_pca_chnl = True
                model_overall.model_loss.opt.visualize_pca_chnl = True
            else:
                model_overall.opt_loss.visualize_pca_chnl = False
                model_overall.model_loss.opt.visualize_pca_chnl = False

            ## Run through the model
            if unet_options.run_eval:
                with torch.no_grad():
                    losses, output = model_overall(inputs, iter_overall)
            else:
                losses, output = model_overall(inputs, iter_overall)

            # if iter_overall == 0:
            #     writer.add_graph(model_overall, input_to_model=(img1,img2,dep1,dep2,idep1, idep2, pose1_2, img1_raw, img2_raw) )

            ## Generate feature map and point selection for images 
            if unet_options.run_eval:
                feature1_norm = output[0]['feature_norm']
                feature1 = output[0]['feature']  ## maybe should use feature_normalized, which is not calculated in eval_mode, so the code structure should be adjusted 
                imgname = inputs['imgname 1']

                mask1_topk = topk_coord_and_feat(feature1_norm, k=3000)
                # mask1_top5k = topk_coord_and_feat(feature1_norm, k=5000)
                # mask1_top8k = topk_coord_and_feat(feature1_norm, k=8000)
                # mask1_top10k = topk_coord_and_feat(feature1_norm, k=10000)
                output[0]['mask_topk'] = mask1_topk
                feat_np1 = feat_to_np(feature1)
                mask_np1 = mask_to_np(mask1_topk)
                # mask_np5k = mask_to_np(mask1_top5k)
                # mask_np8k = mask_to_np(mask1_top8k)
                # mask_np10k = mask_to_np(mask1_top10k)

                for i in range(len(feat_np1)):
                    feat_np1[i].tofile(os.path.join(output_folder, 'feat_'+imgname[i] +'.bin') )
                    mask_np1[i].tofile(os.path.join(output_folder, 'mask_'+imgname[i] +'.bin') )
                    # mask_np5k[i].tofile(os.path.join(output_folder, 'mask5k_'+imgname[i] +'.bin') )
                    # mask_np8k[i].tofile(os.path.join(output_folder, 'mask8k_'+imgname[i] +'.bin') )
                    # mask_np10k[i].tofile(os.path.join(output_folder, 'mask10k_'+imgname[i] +'.bin') )

                    print(iter_overall, 'imgname', i, ':', imgname[i])

            ## Log to tensorboard
            if unet_options.run_eval:
                if visualize_mode:
                    visualize_to_tensorboard(sample_batch, output, writers['val'], unet_options, loss_options, iter_overall)
                scale_to_tensorboard(losses, writers['val'], unet_options, loss_options, iter_overall, output=output)
            else:
                if visualize_mode:
                    visualize_to_tensorboard(sample_batch, output, writers['train'], unet_options, loss_options, iter_overall)
                scale_to_tensorboard(losses, writers['train'], unet_options, loss_options, iter_overall, output=output)

            
            ### Optimize
            loss = losses['final'] / loss_options.iter_between_update
            if not unet_options.run_eval:
                # optim.zero_grad()
                # loss.backward()
                # optim.step()

                if iter_overall > 0 and iter_overall % loss_options.iter_between_update == 0:
                    optim.step()
                    optim.zero_grad()
                    # print('optimizer update at', iter_overall)

                loss.backward()
                

            if iter_overall %10 == 0:
                time_now = time.time()
                time_duration = time_now - start_time
                overall_time += time_duration
                start_time = time_now
                print('Epoch', i_epoch, 'Batch', i_batch, 'Loss', float(loss), 'Time: {:.1f}, {:.1f}.'.format(time_duration, overall_time), 'Max mem:', torch.cuda.max_memory_allocated() )
                # if unet_options.pose_predict_mode:
                #     euler1_2 = sample_batch['rela_euler']

                #     print('euler:')
                #     print(euler1_2)
                #     print('pred_euler:')
                #     print(euler_pred)

            ### adjust learning rate
            if (not unet_options.run_eval) & (not loss_options.subset_in_loss) :
                if iter_overall > 0 and iter_overall % loss_options.lr_decay_iter == 0:
                # if lr_change_time==0 and (iter_overall ==6000 or loss < -1e7):
                    # lr_change_time += 1
                    lr = lr * 0.1
                    for g in optim.param_groups:
                        g['lr'] = g['lr'] * 0.1
                    print('learning rate 0.1x at', iter_overall, ', now:', lr)


            ### validation
            if not unet_options.run_eval:
                if eval_mode:
                    sample_val = next(iter(data_loader_val))
                    inputs_val = reorganize_inputs(sample_val)

                    model_overall.eval()
                    with torch.no_grad():
                        losses, _ = model_overall(inputs_val)
                                
                    model_overall.train()
                    
                    scale_to_tensorboard(losses, writers['val'], unet_options, loss_options, iter_overall)

                if iter_overall % 2000 == 0:
                    if unet_options.continue_train:
                        model_path_save = os.path.join(save_model_folder, 'epoch{:0>2}_{:0>2}_continued.pth'.format(i_epoch, i_batch ) )
                    else:
                        model_path_save = os.path.join(save_model_folder, 'epoch{:0>2}_{:0>2}.pth'.format(i_epoch, i_batch ) )
                    torch.save({
                        'epoch': i_epoch,
                        'model_state_dict': model_overall.model_UNet.state_dict(),
                        'loss_model_state_dict': model_overall.model_loss.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': loss
                        }, model_path_save)
                    print('Model saved to:', model_path_save)    

            iter_overall += 1

    for mode in writers:
        writers[mode].close()

if __name__ == "__main__":
    main()