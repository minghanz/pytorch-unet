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

    
# class innerProdLossFunc(Function):

#     @staticmethod
#     def forward(ctx, feature1, feature2, depth1, depth2, pose1_2, xy1_grid):
    


def main():

    print('Cuda available?', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

    # model = UNet(in_channels=3, n_classes=3, depth=3, wf=4, padding=True).to(device)
    # loss_model = innerProdLoss(device=device).to(device)
    # optim = torch.optim.Adam(model.parameters())

    model_overall = UNetInnerProd(in_channels=3, n_classes=3, depth=3, wf=4, padding=True, device=device)
    optim = torch.optim.Adam(model_overall.model_UNet.parameters(), lr=1e-5) #cefault 1e-3

    img_pose_dataset = ImgPoseDataset(transform=transforms.Compose([Rescale(), ToTensor(device=device) ]) )
    data_to_load = DataLoader(img_pose_dataset, batch_size=3, shuffle=True)

    epochs = 10

    writer = SummaryWriter()

    print('going into loop')
    iter_overall = 0
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

            feature1, feature2, loss = model_overall(img1, img2, dep1, dep2, pose1_2)

            if iter_overall == 0:
                writer.add_graph(model_overall, input_to_model=(img1,img2,dep1,dep2,pose1_2) )
            
            grid1 = torchvision.utils.make_grid(img1)
            grid2 = torchvision.utils.make_grid(img2)

            writer.add_image('img1',grid1, iter_overall)
            writer.add_image('img2',grid2, iter_overall)

            grid1fea = torchvision.utils.make_grid(feature1)
            grid2fea = torchvision.utils.make_grid(feature2)

            writer.add_image('feature1',grid1fea, iter_overall)
            writer.add_image('feature2',grid2fea, iter_overall)

            writer.add_scalar('loss', loss, iter_overall)

            optim.zero_grad()
            loss.backward()
            optim.step()
            if i_batch %100 == 0:
                print('batch', i_batch, 'finished')
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