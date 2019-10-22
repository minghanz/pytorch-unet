import torch
# import torch.nn.functional as F
from unet import UNet
from dataloader import ImgPoseDataset, ToTensor, Rescale, SplitBlocks, RotateOrNot

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

from log import visualize_to_tensorboard, scale_to_tensorboard, pt_sel_log, pt_sel

from hist_and_thresh import create_hist

import json
import copy

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

class Trainer:
    def __init__(self):
        ################################# options
        self.device = torch.device('cuda:1' if torch.cuda.is_available()else 'cpu')

        self.unet_options = UnetOptions()
        self.unet_options.setto()
        self.loss_options = LossOptions(self.unet_options)

        ################################# network
        preprocess_input_fn = None # using the above preprocessing function will make the rgb not in range [0, 1] and cause the rgb_to_hsv function to fail

        self.model_overall = UNetInnerProd(unet_options=self.unet_options, device=self.device, loss_options=self.loss_options )
        self.lr = self.loss_options.lr
        self.optim = torch.optim.Adam(self.model_overall.model_UNet.parameters(), lr=self.lr) #cefault 1e-3

        ################################# dataset
        ### for TUM dataset, you need to run associate.py on a folder of unzipped folders of TUM data sequences
        if self.loss_options.keep_scale_consistent and (not self.loss_options.run_eval):
            img_pose_dataset = ImgPoseDataset(
                root_dir = self.loss_options.root_dir, 
                transform=transforms.Compose([
                    Rescale(output_size=(self.loss_options.height, self.loss_options.width), post_fn=preprocess_input_fn), ToTensor(device=self.device), 
                    SplitBlocks(self.loss_options.height_split, self.loss_options.width_split, self.loss_options.effective_height, self.loss_options.effective_width) ]), 
                folders=self.loss_options.folders )
        elif self.loss_options.data_aug_rotate180 and (not self.loss_options.run_eval):
            img_pose_dataset = ImgPoseDataset(
                root_dir = self.loss_options.root_dir, 
                transform=transforms.Compose([
                    Rescale(output_size=(self.loss_options.height, self.loss_options.width), post_fn=preprocess_input_fn), ToTensor(device=self.device), 
                    RotateOrNot(device=self.device) ]), 
                folders=self.loss_options.folders )
        else:
            img_pose_dataset = ImgPoseDataset(
                root_dir = self.loss_options.root_dir, 
                transform=transforms.Compose([Rescale(output_size=(self.loss_options.height, self.loss_options.width), post_fn=preprocess_input_fn), ToTensor(device=self.device) ]), 
                folders=self.loss_options.folders )

        if self.loss_options.run_eval:
            self.data_loader_train = DataLoader(img_pose_dataset, batch_size=self.loss_options.batch_size, shuffle=False)
        else:
            self.data_loader_train, self.data_loader_val = split_train_val(img_pose_dataset, 0.1, batch_size=self.loss_options.batch_size)

        ################################# pretrained weights
        if self.loss_options.run_eval:
            checkpoint = torch.load('saved_models/with_color_Tue Oct 15 10:36:18 2019/epoch00_10000.pth')
            self.model_overall.model_UNet.load_state_dict(checkpoint['model_state_dict'])
            self.model_overall.model_loss.load_state_dict(checkpoint['loss_model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model_overall.eval()
            self.model_overall.model_UNet.eval()
            self.model_overall.model_loss.eval()

        if self.unet_options.continue_train: 
            save_model_folder = os.path.join('saved_models', 'wo_color_Mon Sep 30 15:21:22 2019')
            checkpoint = torch.load(os.path.join(save_model_folder, 'epoch00_3000.pth') )
            self.model_overall.model_UNet.load_state_dict(checkpoint['model_state_dict'])
            self.model_overall.model_loss.load_state_dict(checkpoint['loss_model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

        ################################ logger
        ctime = time.ctime()
        if self.loss_options.color_in_cost:
            mode_folder = 'with_color'
        else:
            mode_folder = 'wo_color'
        ## SummaryWriter
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join('runs', mode_folder + '_' + ctime + '_' + mode ))
        ## Save model if training, output features if evaluating
        if not self.unet_options.continue_train: 
            if self.loss_options.run_eval:
                self.output_folder = os.path.join('feature_output', mode_folder + '_' + ctime )
                os.mkdir(self.output_folder)
                os.makedirs( os.path.join(self.output_folder, 'feature_map') )
                for k in self.loss_options.top_k_list:
                    for g in self.loss_options.grid_list:
                        for sample_aug in self.loss_options.sample_aug_list:
                            os.makedirs(os.path.join( self.output_folder, 'mask_top_{}/grid_{}/sample_{}'.format(k, g, sample_aug)) )
            else:
                self.save_model_folder = os.path.join('saved_models', mode_folder + '_' + ctime  )
                os.mkdir(self.save_model_folder)
        ## Save the options to a json file
        options = {}
        dict_unet = copy.deepcopy(self.unet_options.__dict__) # https://www.peterbe.com/plog/be-careful-with-using-dict-to-create-a-copy
        dict_loss = copy.deepcopy(self.loss_options.__dict__) # compared with dict(self.loss_options.__dict__), this makes sure the copy is detached from the original
        del dict_loss['opt_unet']
        options['unet_options'] = dict_unet
        options['loss_options'] = dict_loss

        option_file_path = os.path.join('runs', mode_folder + '_' + ctime + '.json' )
        with open(option_file_path, 'w') as option_file:
            json.dump(options, option_file, indent=2)

        
        if self.loss_options.run_eval:
            self.eval()
        else:
            self.train()

    def mode_selection_in_each_iter(self, iter_overall):
        if self.loss_options.run_eval:
            visualize_mode = iter_overall % 10 == 0
            eval_mode = False
        else:
            visualize_mode = (iter_overall <= 1000 and iter_overall % 100 == 0) or (iter_overall > 1000 and iter_overall % 500 == 0)
            eval_mode = iter_overall % 100 == 0

        if visualize_mode and self.loss_options.effective_width <= 128:
            self.model_overall.opt_loss.visualize_pca_chnl = True
            self.model_overall.model_loss.opt.visualize_pca_chnl = True
        else:
            self.model_overall.opt_loss.visualize_pca_chnl = False
            self.model_overall.model_loss.opt.visualize_pca_chnl = False

        return visualize_mode, eval_mode

    def mode_selection_before_eval(self):
        self.model_overall.opt_loss.set_eval_full_size()
        self.model_overall.model_loss.opt.set_eval_full_size()
        return

    def mode_selection_after_eval(self):
        self.model_overall.opt_loss.unset_eval_full_size()
        self.model_overall.model_loss.opt.unset_eval_full_size()
        return

    def timing_and_print_in_each_iter(self, iter_overall,i_epoch, i_batch, loss, start_time, overall_time ):
        if iter_overall %10 == 0:
            time_now = time.time()
            time_duration = time_now - start_time
            overall_time += time_duration
            start_time = time_now
            print('Epoch', i_epoch, 'Batch', i_batch, 'Loss', float(loss), 'Time: {:.1f}, {:.1f}.'.format(time_duration, overall_time), 'Max mem:', torch.cuda.max_memory_allocated() )
            # if self.unet_options.pose_predict_mode:
            #     euler1_2 = sample_batch['rela_euler']
            #     print('euler:')
            #     print(euler1_2)
            #     print('pred_euler:')
            #     print(euler_pred)
        return start_time, overall_time

    def eval(self):
        print('Start evaluation')
        iter_overall = 0
        start_time = time.time()
        overall_time = 0

        for i_epoch in range(self.loss_options.epochs):
            print('entering epoch', i_epoch) 
            for i_batch, sample_batch in enumerate(self.data_loader_train):
                if iter_overall > self.loss_options.total_iters:
                    break
                
                ### Mode selection
                visualize_mode, _ = self.mode_selection_in_each_iter(iter_overall)

                ### Run through the model
                with torch.no_grad():
                    losses, output = self.model_overall(sample_batch, iter_overall)

                ### Plot a histogram of feature norm
                # create_hist(sample_batch, output)

                ### Generate feature map and point selection for images 
                pt_sel(output, self.loss_options.top_k_list, self.loss_options.grid_list, self.loss_options.sample_aug_list)
                pt_sel_log(sample_batch, output, self.output_folder, iter_overall, self.loss_options.top_k_list, self.loss_options.grid_list, self.loss_options.sample_aug_list)

                ### Log to tensorboard
                if visualize_mode:
                    visualize_to_tensorboard(sample_batch, output, self.writers['val'], self.unet_options, self.loss_options, iter_overall)
                scale_to_tensorboard(losses, self.writers['val'], self.unet_options, self.loss_options, iter_overall, output=output)
                
                ### Optimize
                loss = losses['final']
                
                ### Print and timing
                start_time, overall_time = self.timing_and_print_in_each_iter(iter_overall, i_epoch, i_batch, loss, start_time, overall_time )

                iter_overall += 1

        for mode in self.writers:
            self.writers[mode].close()
        

    def train(self):
        print('Start training')
        iter_overall = 0
        start_time = time.time()
        overall_time = 0
        
        iter_in_update_loop = 0
        for i_epoch in range(self.loss_options.epochs):
            print('entering epoch', i_epoch) 
            for i_batch, sample_batch_ in enumerate(self.data_loader_train):
                if iter_overall > self.loss_options.total_iters:
                    break
                
                ### Mode selection
                visualize_mode, eval_mode = self.mode_selection_in_each_iter(iter_overall)

                ### Run through the model
                if self.loss_options.keep_scale_consistent:
                    i_tile = np.random.randint(self.loss_options.height_split)
                    j_tile = np.random.randint(self.loss_options.width_split)
                    sample_batch = sample_batch_[ (i_tile, j_tile) ]
                else:
                    sample_batch = sample_batch_

                losses, output = self.model_overall(sample_batch, iter_overall)

                # if iter_overall == 0:
                #     writer.add_graph(self.model_overall, input_to_model=(img1,img2,dep1,dep2,idep1, idep2, pose1_2, img1_raw, img2_raw) )

                ### Log to tensorboard
                if visualize_mode:
                    visualize_to_tensorboard(sample_batch, output, self.writers['train'], self.unet_options, self.loss_options, iter_overall)
                scale_to_tensorboard(losses, self.writers['train'], self.unet_options, self.loss_options, iter_overall, output=output)
                
                ### Optimize
                # optim.zero_grad()
                # loss.backward()
                # optim.step()
                if iter_overall > 0 and iter_in_update_loop % self.loss_options.iter_between_update == 0:
                    self.optim.step()
                    self.optim.zero_grad()
                    iter_in_update_loop = 0
                loss = losses['final'] / self.loss_options.iter_between_update
                if loss != 0:
                    loss.backward()
                    iter_in_update_loop += 1

                ### Print and timing
                start_time, overall_time = self.timing_and_print_in_each_iter(iter_overall, i_epoch, i_batch, loss, start_time, overall_time )

                ### adjust learning rate
                if not self.loss_options.subset_in_loss :
                    if iter_overall > 0 and iter_overall % self.loss_options.lr_decay_iter == 0:
                        self.lr = self.lr * 0.1
                        for g in self.optim.param_groups:
                            g['lr'] = g['lr'] * 0.1
                        print('learning rate 0.1x at', iter_overall, ', now:', self.lr)


                ### validation
                if eval_mode:
                    sample_val_ = next(iter(self.data_loader_val))

                    if self.loss_options.keep_scale_consistent:
                        i_tile = np.random.randint(self.loss_options.height_split)
                        j_tile = np.random.randint(self.loss_options.width_split)
                        sample_val = sample_val_[ (i_tile, j_tile) ]
                    else:
                        sample_val = sample_val_

                    ## Run through the model
                    self.model_overall.eval()
                    with torch.no_grad():
                        losses, _ = self.model_overall(sample_val)    
                    self.model_overall.train()

                    ### Log to tensorboard
                    scale_to_tensorboard(losses, self.writers['val'], self.unet_options, self.loss_options, iter_overall)

                    ### run eval_in_full_size
                    if self.loss_options.keep_scale_consistent:
                        if visualize_mode:
                            self.mode_selection_before_eval()
                            sample_val = sample_batch_['original']

                            ## Run through the model
                            self.model_overall.eval()
                            with torch.no_grad():
                                _, output = self.model_overall(sample_val)    
                            self.model_overall.train()

                            visualize_to_tensorboard(sample_val, output, self.writers['val'], self.unet_options, self.loss_options, iter_overall)
                            self.mode_selection_after_eval()

                    

                ### save trained model
                if iter_overall % 2000 == 0:
                    if self.unet_options.continue_train:
                        model_path_save = os.path.join(self.save_model_folder, 'epoch{:0>2}_{:0>2}_continued.pth'.format(i_epoch, i_batch ) )
                    else:
                        model_path_save = os.path.join(self.save_model_folder, 'epoch{:0>2}_{:0>2}.pth'.format(i_epoch, i_batch ) )
                    torch.save({
                        'epoch': i_epoch,
                        'model_state_dict': self.model_overall.model_UNet.state_dict(),
                        'loss_model_state_dict': self.model_overall.model_loss.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'loss': loss
                        }, model_path_save)
                    print('Model saved to:', model_path_save)    

                iter_overall += 1

        for mode in self.writers:
            self.writers[mode].close()



if __name__ == "__main__":
    trainer = Trainer()
    
