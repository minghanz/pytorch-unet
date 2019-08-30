import torch
# import torch.nn.functional as F
from unet import UNet
from dataloader import ImgPoseDataset, ToTensor, Rescale

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import torch.autograd.Function as Function

import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
from network_modules import UNetInnerProd, innerProdLoss

def vis_feat(feature, neg=False):
    # only keep the positive or negative part of the feature map, normalize the max to 1 
    # (removing border part because they sometimes are too large)

    if neg:
        feat1_pos = -feature.clone().detach()
        feat1_pos[feature > 0] = 0
    else:
        feat1_pos = feature.clone().detach()
        feat1_pos[feature < 0] = 0
    feat1_pos[:,:,0:5,:] = 0
    feat1_pos[:,:,feat1_pos.shape[2]-5:feat1_pos.shape[2],:] = 0
    feat1_pos[:,:,:,0:5] = 0
    feat1_pos[:,:,:,feat1_pos.shape[3]-5:feat1_pos.shape[3]] = 0
    
    feat1_pos_max = torch.max(feat1_pos)
    if feat1_pos_max > 0:
        feat1_pos = feat1_pos / feat1_pos_max

    return feat1_pos

def feat_svd(feature):
    # input feature is b*c*h*w, need to change to b*c*n
    b = feature.shape[0]
    c = feature.shape[1]
    h = feature.shape[2]
    w = feature.shape[3]
    feat_flat = feature.reshape(b, c, h*w)
    
    # feat_mean = torch.mean(feature, dim=(2,3), keepdim=False)
    # feat_flat = feat_flat - feat_mean.unsqueeze(2).expand_as(feat_flat)
    # print(feat_flat.device)
    # print(feat_flat.shape)
    feat_mean = feat_flat.mean(dim=2, keepdim=True)
    feat_flat = feat_flat - feat_mean.expand_as(feat_flat)
    u, s, v = torch.svd(feat_flat)
    feat_new = torch.bmm( u[:,:,0:3].transpose(1,2), feat_flat) # b*3*n
    feat_img = feat_new.reshape(b,3,h,w)
    return feat_img

def main():

    print('Cuda available?', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

    # model = UNet(in_channels=3, n_classes=3, depth=3, wf=4, padding=True).to(device)
    # loss_model = innerProdLoss(device=device).to(device)
    # optim = torch.optim.Adam(model.parameters())

    diff_mode = True
    sparse_mode = True
    kernalize = True
    color_in_cost = True
    L2_norm = False
    width = 96 # (72*96)
    height = 72
    model_overall = UNetInnerProd(in_channels=3, n_classes=64, depth=3, wf=4, padding=True, up_mode='upsample', device=device, 
                                    diff_mode=diff_mode, sparse_mode=sparse_mode, kernalize=kernalize, color_in_cost=color_in_cost, L2_norm=L2_norm, 
                                    fx=int(width/2), fy=int(width/2), cx=int(width/2), cy=int(height/2) )
    lr = 1e-4
    optim = torch.optim.Adam(model_overall.model_UNet.parameters(), lr=lr) #cefault 1e-3

    img_pose_dataset = ImgPoseDataset(transform=transforms.Compose([Rescale(output_size=(height,width)), ToTensor(device=device) ]) )
    data_to_load = DataLoader(img_pose_dataset, batch_size=2, shuffle=True)

    epochs = 10

    writer = SummaryWriter()

    print('going into loop')
    iter_overall = 0
    lr_change_time = 0
    for i_epoch in range(epochs):
        print('entering epoch', i_epoch) 
        for i_batch, sample_batch in enumerate(data_to_load):
            img1 = sample_batch['image 1']
            img2 = sample_batch['image 2']
            dep1 = sample_batch['depth 1']
            dep2 = sample_batch['depth 2']
            idep1 = sample_batch['idepth 1']
            idep2 = sample_batch['idepth 2']
            pose1_2 = sample_batch['rela_pose']
            euler1_2 = sample_batch['rela_euler']
            
            # feature1 = model(img1)
            # feature2 = model(img2)

            # dep1.requires_grad = False
            # dep2.requires_grad = False

            # loss = loss_model(feature1, feature2, dep1, dep2, pose1_2)
            if diff_mode:
                model_overall.model_loss.gen_rand_pose()
            feature1_full, feature2_full, loss, innerp_loss, feat_norm, innerp_loss_pred, euler_pred = \
                model_overall(img1, img2, dep1, dep2, idep1, idep2, pose1_2)

            if iter_overall == 0:
                writer.add_graph(model_overall, input_to_model=(img1,img2,dep1,dep2,idep1, idep2, pose1_2) )
            
            grid1 = torchvision.utils.make_grid(img1)
            grid2 = torchvision.utils.make_grid(img2)

            writer.add_image('img1',grid1, iter_overall)
            writer.add_image('img2',grid2, iter_overall)

            # feature1 = feature1_full[:,0:3,:,:]
            # feature2 = feature2_full[:,0:3,:,:]

            feature1 = feat_svd(feature1_full)
            feature2 = feat_svd(feature2_full)

            min_fea_1 = torch.min(feature1)
            min_fea_2 = torch.min(feature2)
            max_fea_1 = torch.max(feature1)
            max_fea_2 = torch.max(feature2)
            writer.add_scalar('min_fea_1', min_fea_1, iter_overall)
            writer.add_scalar('min_fea_2', min_fea_2, iter_overall)
            writer.add_scalar('max_fea_1', max_fea_1, iter_overall)
            writer.add_scalar('max_fea_2', max_fea_2, iter_overall)

            
            # to normalize feature map to max 1 for better visualization
            # if torch.abs(min_fea_1) > torch.abs(max_fea_1):
            #     max_abs_1 = torch.abs(min_fea_1)
            # else:
            #     max_abs_1 = torch.abs(max_fea_1)
            # if torch.abs(min_fea_2) > torch.abs(max_fea_2):
            #     max_abs_2 = torch.abs(min_fea_2)
            # else:
            #     max_abs_2 = torch.abs(max_fea_2) 
            # feature1  = feature1 / max_abs_1
            # feature2  = feature2 / max_abs_2

            # The tensorboard visualize value in (-1,0) the same as in (0, 1), e.g. -1.9 = -0.9 = 0.1 = 1.1, 1 is the brightest
            feat1_pos = vis_feat(feature1)
            feat1_neg = vis_feat(feature1, neg=True)
            feat2_pos = vis_feat(feature2)
            feat2_neg = vis_feat(feature2, neg=True)

            grid1pos = torchvision.utils.make_grid(feat1_pos)
            grid1neg = torchvision.utils.make_grid(feat1_neg)
            grid2pos = torchvision.utils.make_grid(feat2_pos)
            grid2neg = torchvision.utils.make_grid(feat2_neg)

            writer.add_image('feature1_pos',grid1pos, iter_overall)
            writer.add_image('feature1_neg',grid1neg, iter_overall)
            writer.add_image('feature2_pos',grid2pos, iter_overall)
            writer.add_image('feature2_neg',grid2neg, iter_overall)

            feature1  = feature1 / max_fea_1
            feature2  = feature2 / max_fea_2

            grid1fea = torchvision.utils.make_grid(feature1)
            grid2fea = torchvision.utils.make_grid(feature2)

            writer.add_image('feature1',grid1fea, iter_overall)
            writer.add_image('feature2',grid2fea, iter_overall)

            
            # feat_test=feature1.clone().detach()
            # feat_test[:,:,:,:] = 0
            # feat_test[:,:,0:10, :] = 0.2
            # feat_test[:,:,10:20, :] = 0.1
            # # feat_test[:,:,20:30, :] = 1.8
            # # feat_test[:,:,30:40, :] = -1.1
            # grid_test= torchvision.utils.make_grid(feat_test)
            # writer.add_image('feature_test',grid_test, iter_overall)

            writer.add_scalar('loss', loss, iter_overall)
            writer.add_scalar('innerp_loss', innerp_loss, iter_overall)
            writer.add_scalar('feat_norm', feat_norm, iter_overall)
            writer.add_scalar('innerp_loss_pred', innerp_loss_pred, iter_overall)

            optim.zero_grad()
            loss.backward()
            optim.step()
            if i_batch %10 == 0:
                print('batch', i_batch, 'finished')
                print('euler:')
                print(euler1_2)
                print('pred_euler:')
                print(euler_pred)
            if lr_change_time==0 and (i_batch ==2000 or loss < -1e7):
                lr_change_time += 1
                lr = lr * 0.1
                for g in optim.param_groups:
                    g['lr'] = g['lr'] * 0.1
                print('learning rate 0.1x at', i_batch, ', now:', lr)
            if lr_change_time==1 and (i_batch ==4000 or loss < -1e10):
                lr_change_time += 1
                lr = lr * 0.1
                for g in optim.param_groups:
                    g['lr'] = g['lr'] * 0.1
                print('learning rate 0.1x at', i_batch, ', now:', lr)
            iter_overall += 1

        model_path_save = os.path.join('saved_models', 'epoch{:0>2}_{:0>2}.pth'.format(1, i_epoch) )
        torch.save({
            'epoch': i_epoch,
            'model_state_dict': model_overall.model_UNet.state_dict(),
            'loss_model_state_dict': model_overall.model_loss.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss
            }, model_path_save)
        print('Model saved to:', model_path_save) 
    writer.close()

if __name__ == "__main__":
    main()