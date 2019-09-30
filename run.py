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

def vis_feat(feature, neg=False):
    # only keep the positive or negative part of the feature map, normalize the max to 1 
    # (removing border part because they sometimes are too large)

    if neg:
        feat1_pos = -feature.clone().detach()
        feat1_pos[feature > 0] = 0
    else:
        feat1_pos = feature.clone().detach()
        feat1_pos[feature < 0] = 0
    # feat1_pos[:,:,0:5,:] = 0
    # feat1_pos[:,:,feat1_pos.shape[2]-5:feat1_pos.shape[2],:] = 0
    # feat1_pos[:,:,:,0:5] = 0
    # feat1_pos[:,:,:,feat1_pos.shape[3]-5:feat1_pos.shape[3]] = 0
    
    feat1_pos_max = torch.max(feat1_pos)
    if feat1_pos_max > 0:
        feat1_pos = feat1_pos / feat1_pos_max

    return feat1_pos

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

def main():

    print('Cuda available?', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

    unet_options = UnetOptions()
    unet_options.setto()
    loss_options = LossOptions(unet_options)
    
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    preprocess_input_fn = get_preprocessing_fn('resnet34', pretrained='imagenet')

    model_overall = UNetInnerProd(unet_options=unet_options, device=device, loss_options=loss_options )
    lr = 1e-4
    optim = torch.optim.Adam(model_overall.model_UNet.parameters(), lr=lr) #cefault 1e-3

    ### Create dataset
    ### for TUM dataset, you need to run associate.py on a folder of unzipped folders of TUM data sequences
    img_pose_dataset = ImgPoseDataset(
        root_dir = loss_options.root_dir, 
        transform=transforms.Compose([Rescale(output_size=(loss_options.height, loss_options.width), post_fn=preprocess_input_fn), ToTensor(device=device) ]), 
        folders=loss_options.folders )
    
    if not unet_options.run_eval:
        data_loader_train, data_loader_val = split_train_val(img_pose_dataset, 0.1)
    else:
        data_loader_train = DataLoader(img_pose_dataset, batch_size=1, shuffle=False)

    if unet_options.run_eval:
        checkpoint = torch.load('saved_models/epoch00_6000.pth')
        model_overall.model_UNet.load_state_dict(checkpoint['model_state_dict'])
        model_overall.model_loss.load_state_dict(checkpoint['loss_model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        model_overall.eval()
        model_overall.model_UNet.eval()
        model_overall.model_loss.eval()

    epochs = 1

    writer = SummaryWriter()

    print('going into loop')
    iter_overall = 0
    lr_change_time = 0
    start_time = time.time()
    ctime = time.ctime()
    if loss_options.color_in_cost:
        mode_folder = 'with_color_'
    else:
        mode_folder = 'wo_color_'
    if unet_options.run_eval:
        output_folder = os.path.join('feature_output', mode_folder + ctime )
        os.mkdir(output_folder)
    else:
        save_model_folder = os.path.join('saved_models', mode_folder + ctime )
        os.mkdir(save_model_folder)
    overall_time = 0
    for i_epoch in range(epochs):
        print('entering epoch', i_epoch) 
        for i_batch, sample_batch in enumerate(data_loader_train):
            if i_batch > 10000:
                break
            img1 = sample_batch['image 1']
            img2 = sample_batch['image 2']
            img1_raw = sample_batch['image 1 raw']
            img2_raw = sample_batch['image 2 raw']
            
            dep1 = sample_batch['depth 1']
            dep2 = sample_batch['depth 2']
            idep1 = sample_batch['idepth 1']
            idep2 = sample_batch['idepth 2']
            pose1_2 = sample_batch['rela_pose']
            euler1_2 = sample_batch['rela_euler']
            

            model_overall.set_norm_level(i_batch)

            if unet_options.run_eval:
                with torch.no_grad():
                    losses, output = model_overall(sample_batch)
            else:
                losses, output = model_overall(sample_batch)
            
            loss = losses['final']
            innerp_loss = losses['CVO']
            feat_norm = losses['norm']

            if unet_options.pose_predict_mode:
                innerp_loss_pred = losses['CVO_pred']
                euler_pred = output['euler_pred']

            if unet_options.weight_map_mode:
                feat_w_1 = output[0]['feature_w']
                feat_w_2 = output[1]['feature_w']
            
            feature1_norm = output[0]['feature_norm']
            feature2_norm = output[1]['feature_norm']
            feature1 = output[0]['feature']
            feature2 = output[1]['feature']

            if loss_options.pca_in_loss or loss_options.visualize_pca_chnl:
                feature1_chnl3 = output[0]['feature_chnl3']
                feature2_chnl3 = output[1]['feature_chnl3']
                # feature1_norm_pca = output[0]['feature_norm_pca']
                # feature2_norm_pca = output[1]['feature_norm_pca']
                
            if loss_options.subset_in_loss:
                mask_1 = output[0]['norm_mask']
                mask_2 = output[1]['norm_mask']

            # if iter_overall == 0:
            #     writer.add_graph(model_overall, input_to_model=(img1,img2,dep1,dep2,idep1, idep2, pose1_2, img1_raw, img2_raw) )

            if unet_options.run_eval:
                mask1_topk = topk_coord_and_feat(feature1_norm, k=3000)
                # mask1_top5k = topk_coord_and_feat(feature1_norm, k=5000)
                # mask1_top8k = topk_coord_and_feat(feature1_norm, k=8000)
                # mask1_top10k = topk_coord_and_feat(feature1_norm, k=10000)
                feat_np1 = feat_to_np(feature1)
                mask_np1 = mask_to_np(mask1_topk)
                mask_np5k = mask_to_np(mask1_top5k)
                mask_np8k = mask_to_np(mask1_top8k)
                mask_np10k = mask_to_np(mask1_top10k)

                imgname = sample_batch['imgname 1']
                for i in range(len(feat_np1)):
                    feat_np1[i].tofile(os.path.join(output_folder, 'feat_'+imgname[i] +'.bin') )
                    mask_np1[i].tofile(os.path.join(output_folder, 'mask_'+imgname[i] +'.bin') )
                    # mask_np5k[i].tofile(os.path.join(output_folder, 'mask5k_'+imgname[i] +'.bin') )
                    # mask_np8k[i].tofile(os.path.join(output_folder, 'mask8k_'+imgname[i] +'.bin') )
                    # mask_np10k[i].tofile(os.path.join(output_folder, 'mask10k_'+imgname[i] +'.bin') )

                    print(iter_overall, 'imgname', i, ':', imgname[i])

            if unet_options.run_eval:
                visualize_mode = True
            else:
                visualize_mode = i_batch % 50 == 0

            eval_mode = i_batch % 50 == 0
            if visualize_mode:
                grid1 = torchvision.utils.make_grid(img1_raw)
                grid2 = torchvision.utils.make_grid(img2_raw)

                writer.add_image('img1',grid1, iter_overall)
                writer.add_image('img2',grid2, iter_overall)

                grid1fea = torchvision.utils.make_grid(feature1_norm)
                grid2fea = torchvision.utils.make_grid(feature2_norm)

                writer.add_image('feature1',grid1fea, iter_overall)
                writer.add_image('feature2',grid2fea, iter_overall)

                if unet_options.run_eval:
                    grid1_mask = torchvision.utils.make_grid(mask1_topk)
                    writer.add_image('mask1_topk',grid1_mask, iter_overall)
                    # grid1_mask = torchvision.utils.make_grid(mask1_top5k)
                    # writer.add_image('mask1_top5k',grid1_mask, iter_overall)
                    # grid1_mask = torchvision.utils.make_grid(mask1_top8k)
                    # writer.add_image('mask1_top8k',grid1_mask, iter_overall)
                    # grid1_mask = torchvision.utils.make_grid(mask1_top10k)
                    # writer.add_image('mask1_top10k',grid1_mask, iter_overall)

                if unet_options.weight_map_mode:
                    # feat_w_1 = feat_w_1 /  torch.max(feat_w_1)
                    # feat_w_2 = feat_w_2 /  torch.max(feat_w_2)
                    feat_w_1 = ( (feat_w_1 + torch.min(feat_w_1)) / (torch.max(feat_w_1) - torch.min(feat_w_1))*255 ).to(torch.uint8)
                    feat_w_2 = ( (feat_w_2 + torch.min(feat_w_2)) / (torch.max(feat_w_2) - torch.min(feat_w_2))*255 ).to(torch.uint8)
                    grid3 = torchvision.utils.make_grid(feat_w_1)
                    grid4 = torchvision.utils.make_grid(feat_w_2)
                    # print(feat_w_1)
                    # print(torch.sum(feat_w_1))
                    writer.add_image('weight1',grid3, iter_overall)
                    writer.add_image('weight2',grid4, iter_overall)

                if loss_options.subset_in_loss:
                    grid1_mask = torchvision.utils.make_grid(mask_1)
                    grid2_mask = torchvision.utils.make_grid(mask_2)

                    writer.add_image('mask1',grid1_mask, iter_overall)
                    writer.add_image('mask2',grid2_mask, iter_overall)

                if loss_options.pca_in_loss or loss_options.visualize_pca_chnl:
                    feat1_abs = torch.abs(feature1_chnl3)
                    feat2_abs = torch.abs(feature2_chnl3)
                    min_fea_1_abs = torch.min(feat1_abs)
                    min_fea_2_abs = torch.min(feat2_abs)
                    max_fea_1_abs = torch.max(feat1_abs)
                    max_fea_2_abs = torch.max(feat2_abs)
                    max_fea_1 = max_fea_1_abs
                    max_fea_2 = max_fea_2_abs

                    writer.add_scalar('min_fea_1_abs', min_fea_1_abs, iter_overall)
                    writer.add_scalar('min_fea_2_abs', min_fea_2_abs, iter_overall)
                    writer.add_scalar('max_fea_1_abs', max_fea_1_abs, iter_overall)
                    writer.add_scalar('max_fea_2_abs', max_fea_2_abs, iter_overall)

                    # The tensorboard visualize value in (-1,0) the same as in (0, 1), e.g. -1.9 = -0.9 = 0.1 = 1.1, 1 is the brightest
                    feat1_pos = vis_feat(feature1_chnl3)
                    feat1_neg = vis_feat(feature1_chnl3, neg=True)
                    feat2_pos = vis_feat(feature2_chnl3)
                    feat2_neg = vis_feat(feature2_chnl3, neg=True)

                    grid1pos = torchvision.utils.make_grid(feat1_pos)
                    grid1neg = torchvision.utils.make_grid(feat1_neg)
                    grid2pos = torchvision.utils.make_grid(feat2_pos)
                    grid2neg = torchvision.utils.make_grid(feat2_neg)

                    writer.add_image('feature1_pos',grid1pos, iter_overall)
                    writer.add_image('feature1_neg',grid1neg, iter_overall)
                    writer.add_image('feature2_pos',grid2pos, iter_overall)
                    writer.add_image('feature2_neg',grid2neg, iter_overall)
                

            ###############################################################

            
            # feat_test=feature1_chnl3.clone().detach()
            # feat_test[:,:,:,:] = 0
            # feat_test[:,:,0:10, :] = 0.2
            # feat_test[:,:,10:20, :] = 0.1
            # # feat_test[:,:,20:30, :] = 1.8
            # # feat_test[:,:,30:40, :] = -1.1
            # grid_test= torchvision.utils.make_grid(feat_test)
            # writer.add_image('feature_test',grid_test, iter_overall)
            if unet_options.run_eval:
                writer.add_scalar('loss', loss, iter_overall)
            else:
                writer.add_scalars('loss', {'train': loss}, iter_overall)

            if loss_options.sparsify_mode != 6 and loss_options.sparsify_mode != 1:
                # 6 and 1 are the only modes where norm is not calculated 
                if unet_options.run_eval:
                    writer.add_scalar('innerp_loss', innerp_loss, iter_overall)
                else:
                    writer.add_scalars('innerp_loss', {'train': innerp_loss}, iter_overall)
                writer.add_scalar('feat_norm', feat_norm, iter_overall)
            if unet_options.pose_predict_mode:
                writer.add_scalar('innerp_loss_pred', innerp_loss_pred, iter_overall)

            ### optimize
            if not unet_options.run_eval:
                optim.zero_grad()
                loss.backward()
                optim.step()

            if i_batch %10 == 0:
                time_now = time.time()
                time_duration = time_now - start_time
                overall_time += time_duration
                start_time = time_now
                print('batch', i_batch, 'finished. Time: {:.1f}, {:.1f}.'.format(time_duration, overall_time), 'Max mem:', torch.cuda.max_memory_allocated() )
                if unet_options.pose_predict_mode:
                    print('euler:')
                    print(euler1_2)
                    print('pred_euler:')
                    print(euler_pred)

            # ### adjust learning rate
            # if lr_change_time==0 and (i_batch ==2000 or loss < -1e7):
            #     lr_change_time += 1
            #     lr = lr * 0.1
            #     for g in optim.param_groups:
            #         g['lr'] = g['lr'] * 0.1
            #     print('learning rate 0.1x at', i_batch, ', now:', lr)
            # if lr_change_time==1 and (i_batch ==4000 or loss < -1e10):
            #     lr_change_time += 1
            #     lr = lr * 0.1
            #     for g in optim.param_groups:
            #         g['lr'] = g['lr'] * 0.1
            #     print('learning rate 0.1x at', i_batch, ', now:', lr)


            ### validation
            if not unet_options.run_eval:
                if eval_mode:
                    sample_val = next(iter(data_loader_val))
                    img1 = sample_val['image 1']
                    img2 = sample_val['image 2']
                    img1_raw = sample_val['image 1 raw']
                    img2_raw = sample_val['image 2 raw']
                    
                    dep1 = sample_val['depth 1']
                    dep2 = sample_val['depth 2']
                    idep1 = sample_val['idepth 1']
                    idep2 = sample_val['idepth 2']
                    pose1_2 = sample_val['rela_pose']
                    euler1_2 = sample_val['rela_euler']

                    model_overall.eval()
                    with torch.no_grad():
                        losses, output = model_overall(sample_val)                    
                        loss = losses['final']
                                
                    model_overall.train()
                    
                    writer.add_scalars('loss', {'val': loss}, iter_overall)
                    # writer.add_scalars('innerp_loss', {'val': innerp_loss}, iter_overall)


                if iter_overall % 1000 == 0:
                    model_path_save = os.path.join(save_model_folder, 'epoch{:0>2}_{:0>2}.pth'.format(i_epoch, iter_overall ) )
                    torch.save({
                        'epoch': i_epoch,
                        'model_state_dict': model_overall.model_UNet.state_dict(),
                        'loss_model_state_dict': model_overall.model_loss.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': loss
                        }, model_path_save)
                    print('Model saved to:', model_path_save)    

            iter_overall += 1

    writer.close()

if __name__ == "__main__":
    main()