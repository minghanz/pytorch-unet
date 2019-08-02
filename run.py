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
from geometry import UNetInnerProd, innerProdLoss

def main():

    print('Cuda available?', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

    # model = UNet(in_channels=3, n_classes=3, depth=3, wf=4, padding=True).to(device)
    # loss_model = innerProdLoss(device=device).to(device)
    # optim = torch.optim.Adam(model.parameters())

    diff_mode = True
    sparse_mode = True
    model_overall = UNetInnerProd(in_channels=3, n_classes=3, depth=3, wf=4, padding=True, device=device, diff_mode=diff_mode, sparse_mode=sparse_mode)
    lr = 1e-4
    optim = torch.optim.Adam(model_overall.model_UNet.parameters(), lr=lr) #cefault 1e-3

    img_pose_dataset = ImgPoseDataset(transform=transforms.Compose([Rescale(), ToTensor(device=device) ]) )
    data_to_load = DataLoader(img_pose_dataset, batch_size=3, shuffle=True)

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
            pose1_2 = sample_batch['rela_pose']
            
            # feature1 = model(img1)
            # feature2 = model(img2)

            # dep1.requires_grad = False
            # dep2.requires_grad = False

            # loss = loss_model(feature1, feature2, dep1, dep2, pose1_2)
            if diff_mode:
                model_overall.model_loss.gen_rand_pose()
            feature1, feature2, loss, innerp_loss, feat_norm = model_overall(img1, img2, dep1, dep2, pose1_2)

            if iter_overall == 0:
                writer.add_graph(model_overall, input_to_model=(img1,img2,dep1,dep2,pose1_2) )
            
            grid1 = torchvision.utils.make_grid(img1)
            grid2 = torchvision.utils.make_grid(img2)

            writer.add_image('img1',grid1, iter_overall)
            writer.add_image('img2',grid2, iter_overall)

            min_fea_1 = torch.min(feature1)
            min_fea_2 = torch.min(feature2)
            max_fea_1 = torch.max(feature1)
            max_fea_2 = torch.max(feature2)
            writer.add_scalar('min_fea_1', min_fea_1, iter_overall)
            writer.add_scalar('min_fea_2', min_fea_2, iter_overall)
            writer.add_scalar('max_fea_1', max_fea_1, iter_overall)
            writer.add_scalar('max_fea_2', max_fea_2, iter_overall)

            feature1  = feature1 / max_fea_1
            feature2  = feature2 / max_fea_2
            
            grid1fea = torchvision.utils.make_grid(feature1)
            grid2fea = torchvision.utils.make_grid(feature2)

            writer.add_image('feature1',grid1fea, iter_overall)
            writer.add_image('feature2',grid2fea, iter_overall)

            writer.add_scalar('loss', loss, iter_overall)
            writer.add_scalar('innerp_loss', innerp_loss, iter_overall)
            writer.add_scalar('feat_norm', feat_norm, iter_overall)

            optim.zero_grad()
            loss.backward()
            optim.step()
            if i_batch %100 == 0:
                print('batch', i_batch, 'finished')
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