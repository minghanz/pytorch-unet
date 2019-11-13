import torchvision
import torch
import os 

import matplotlib.pyplot as plt
import numpy as np
import torchsnooper

def scale_to_tensorboard(losses, writer, unet_options, loss_options, iter_overall, output=None):
    for item in losses:
        if item == 'final':
            if loss_options.min_grad_mode:
                writer.add_scalar('loss/final_grad', losses[item], iter_overall)
            if loss_options.min_dist_mode:
                writer.add_scalar('loss/final_dist', losses[item], iter_overall)
        else:
            writer.add_scalar('loss/{}'.format(item), losses[item], iter_overall)

    if output is not None:
        feature1 = output[0]['feature_normalized']
        feature2 = output[1]['feature_normalized']

        feat1_abs = torch.abs(feature1)
        feat2_abs = torch.abs(feature2)
        min_fea_1_abs = torch.min(feat1_abs)
        min_fea_2_abs = torch.min(feat2_abs)
        max_fea_1_abs = torch.max(feat1_abs)
        max_fea_2_abs = torch.max(feat2_abs)
        # max_fea_1 = max_fea_1_abs
        # max_fea_2 = max_fea_2_abs

        writer.add_scalar('fea_stats/abs_min_1', min_fea_1_abs, iter_overall)
        writer.add_scalar('fea_stats/abs_min_2', min_fea_2_abs, iter_overall)
        writer.add_scalar('fea_stats/abs_max_1', max_fea_1_abs, iter_overall)
        writer.add_scalar('fea_stats/abs_max_2', max_fea_2_abs, iter_overall)

def visualize_to_tensorboard(sample_batch, output, writer, unet_options, loss_options, iter_overall):
    
    img1_raw = sample_batch[0]['img_raw']
    img2_raw = sample_batch[1]['img_raw']
    
    grid1 = torchvision.utils.make_grid(img1_raw)
    grid2 = torchvision.utils.make_grid(img2_raw)

    writer.add_image('img1/rgb',grid1, iter_overall)
    writer.add_image('img2/rgb',grid2, iter_overall)

    feature1_norm = output[0]['feature_norm']
    feature2_norm = output[1]['feature_norm']

    grid1fea = torchvision.utils.make_grid(feature1_norm)
    grid2fea = torchvision.utils.make_grid(feature2_norm)

    writer.add_image('img1/feature',grid1fea, iter_overall)
    writer.add_image('img2/feature',grid2fea, iter_overall)

    # feature1_norm_noedge = remove_edge(feature1_norm)
    # feature2_norm_noedge = remove_edge(feature2_norm)
    
    # grid1fea = torchvision.utils.make_grid(feature1_norm_noedge)
    # grid2fea = torchvision.utils.make_grid(feature2_norm_noedge)

    # writer.add_image('img1/feature_noedge',grid1fea, iter_overall)
    # writer.add_image('img2/feature_noedge',grid2fea, iter_overall)


    depth1 = sample_batch[0]['depth']
    depth2 = sample_batch[1]['depth']

    depth1 = torch.where(depth1 > 0, 1/depth1, torch.zeros_like(depth1))
    depth2 = torch.where(depth2 > 0, 1/depth2, torch.zeros_like(depth2))
    depth1 = depth1 / torch.max(depth1)
    depth2 = depth2 / torch.max(depth2)
    
    grid1 = torchvision.utils.make_grid(depth1)
    grid2 = torchvision.utils.make_grid(depth2)

    writer.add_image('img1/depth',grid1, iter_overall)
    writer.add_image('img2/depth',grid2, iter_overall)


    if unet_options.weight_map_mode:
        feat_w_1 = output[0]['feature_w']
        feat_w_2 = output[1]['feature_w']
        # feat_w_1 = feat_w_1 /  torch.max(feat_w_1)
        # feat_w_2 = feat_w_2 /  torch.max(feat_w_2)
        feat_w_1 = ( (feat_w_1 + torch.min(feat_w_1)) / (torch.max(feat_w_1) - torch.min(feat_w_1))*255 ).to(torch.uint8)
        feat_w_2 = ( (feat_w_2 + torch.min(feat_w_2)) / (torch.max(feat_w_2) - torch.min(feat_w_2))*255 ).to(torch.uint8)
        grid3 = torchvision.utils.make_grid(feat_w_1)
        grid4 = torchvision.utils.make_grid(feat_w_2)
        # print(feat_w_1)
        # print(torch.sum(feat_w_1))
        writer.add_image('img1/weight',grid3, iter_overall)
        writer.add_image('img2/weight',grid4, iter_overall)

    if loss_options.subset_in_loss:
        mask_1 = output[0]['norm_mask']
        mask_2 = output[1]['norm_mask']

        grid1_mask = torchvision.utils.make_grid(mask_1)
        grid2_mask = torchvision.utils.make_grid(mask_2)

        writer.add_image('img1/mask',grid1_mask, iter_overall)
        writer.add_image('img2/mask',grid2_mask, iter_overall)
    
    if loss_options.samp_pt:
        mask_1 = output[0]['mask']
        mask_2 = output[1]['mask']

        grid1_mask = torchvision.utils.make_grid(mask_1)
        grid2_mask = torchvision.utils.make_grid(mask_2)

        writer.add_image('img1/mask',grid1_mask, iter_overall)
        writer.add_image('img2/mask',grid2_mask, iter_overall)

    if loss_options.run_eval:
        for item in output[0]:
            if 'mask_top_' in item:
                mask1_topk = output[0][item]
                grid1_mask = torchvision.utils.make_grid(mask1_topk)
                writer.add_image(item, grid1_mask, iter_overall)

    if loss_options.pca_in_loss or loss_options.visualize_pca_chnl:
        if unet_options.non_neg:
            list_cnl = [3, 6]
        else:
            list_cnl = [3]
        for cnl in list_cnl:
            name_cnl = 'feature_chnl{}'.format(cnl)
            
            feature1_chnl3 = output[0][name_cnl]
            feature2_chnl3 = output[1][name_cnl]

            # The tensorboard visualize value in (-1,0) the same as in (0, 1), e.g. -1.9 = -0.9 = 0.1 = 1.1, 1 is the brightest
            feat1_pos = vis_feat(feature1_chnl3)
            feat2_pos = vis_feat(feature2_chnl3)

            grid1pos = torchvision.utils.make_grid(feat1_pos)
            grid2pos = torchvision.utils.make_grid(feat2_pos)

            if unet_options.non_neg:
                writer.add_image('img1/'+name_cnl, grid1pos, iter_overall)
                writer.add_image('img2/'+name_cnl, grid2pos, iter_overall)
            else:
                writer.add_image('img1/feat_pos',grid1pos, iter_overall)
                writer.add_image('img2/feat_pos',grid2pos, iter_overall)

                feat1_neg = vis_feat(feature1_chnl3, neg=True)
                feat2_neg = vis_feat(feature2_chnl3, neg=True)
                grid1neg = torchvision.utils.make_grid(feat1_neg)
                grid2neg = torchvision.utils.make_grid(feat2_neg)
                writer.add_image('img1/feat_neg',grid1neg, iter_overall)
                writer.add_image('img2/feat_neg',grid2neg, iter_overall)
                
        # if iter_overall == 0:
        #     writer.add_graph(model_overall, input_to_model=(img1,img2,dep1,dep2,idep1, idep2, pose1_2, img1_raw, img2_raw) )


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

def remove_edge(feature, byte_mode=False):
    marg = 2
    if byte_mode:
        feature[:,:,0:marg,:] = False
        feature[:,:,feature.shape[2]-marg:feature.shape[2],:] = False
        feature[:,:,:,0:marg] = False
        feature[:,:,:,feature.shape[3]-marg:feature.shape[3]] = False
    else:
        feature[:,:,0:marg,:] = 0
        feature[:,:,feature.shape[2]-marg:feature.shape[2],:] = 0
        feature[:,:,:,0:marg] = 0
        feature[:,:,:,feature.shape[3]-marg:feature.shape[3]] = 0

    return

# @torchsnooper.snoop()
def topk_by_otsu(feature_norm, sample_aug, k, visualize=False):
    bin_num = 100
    hist_min = 0
    hist_max = 1
    masks = []
    for ib in range(feature_norm.shape[0]):
        ## create the histogram
        norm_hist = torch.histc(feature_norm[ib], bins=bin_num, min=hist_min, max=hist_max)

        x_grid = np.linspace(hist_min, hist_max, bin_num+1)[0:-1]
        bar_width_half = (hist_max - hist_min)/bin_num/2
        norm_grid = x_grid + bar_width_half
        norm_grid = torch.from_numpy(norm_grid).to(feature_norm.device, feature_norm.dtype)

        ## OTSU algorithm https://blog.csdn.net/baimafujinji/article/details/50629103
        wtotal = norm_hist.sum()
        w0 = torch.tensor(0, device=feature_norm.device)
        sumtotal = torch.sum(norm_hist * norm_grid)
        sum0 = torch.tensor(0, device=feature_norm.device)
        
        maximum = 0
        level = 0
        for i in range(bin_num):
            w0 = w0 + norm_hist[i]
            if w0 == 0:
                continue
            w1 = wtotal - w0
            if w1 == 0:
                break
            sum0 = sum0 + norm_grid[i] * norm_hist[i]
            m0 = sum0 / w0
            m1 = (sumtotal - sum0) / w1
            icv = w0 * w1 * (m0 - m1) * (m0 - m1)
            if icv > maximum:
                level = i
                maximum = icv

        threshold = norm_grid[level] + bar_width_half
        # print("threshold:", threshold)

        ## sample pixels given the threshold
        if k == -1:
            mask = feature_norm[ib] >= threshold
            mask = mask.squeeze(1)
        elif sample_aug == -1:
            valid_idx = (feature_norm[ib] >= threshold).nonzero()
            valid_num = valid_idx.shape[0]

            perm = torch.randperm(valid_num)
            idx_sample = perm[:k]
            valid_idx_sample = valid_idx[idx_sample].split(1, dim=1)

            mask = torch.zeros_like(feature_norm[ib]).to(dtype=torch.bool)
            mask[valid_idx_sample] = True
        else:
            raise ValueError("k or sample_aug must be -1 here")
        masks.append(mask)

        if visualize:            
            plt.figure()
            norm_hist_np = norm_hist.cpu().numpy()
            plt.bar(x_grid, norm_hist_np, width=bar_width_half*2 )
            plt.plot([threshold, threshold], [0, float(torch.max(norm_hist).cpu())])
            plt.show()
    
    masks = torch.stack(masks, dim=0).squeeze(1)

    return masks

def topk_coord_and_feat_single_grid(feature_norm, sample_aug, k):

    if k == -1 or sample_aug == -1:
        mask = topk_by_otsu(feature_norm, sample_aug, k, visualize=False)
    else:
        feature_n_flat = feature_norm.reshape(feature_norm.shape[0], -1)

        topk = torch.topk(feature_n_flat, sample_aug*k)
        # print('topk shape', topk.values.shape)
        # print("cur thresh at {}:".format(sample_aug * k), topk.values[0,-1])
        # thresh = topk.values[:,-1]

        mask = torch.zeros_like(feature_n_flat).to(dtype=torch.bool)
        if sample_aug > 1:
            for i in range(feature_norm.shape[0]):
                perm = torch.randperm(sample_aug*k)
                idx_sample = perm[:k]
                topk_sample = topk.indices[i][idx_sample]
                mask[i, topk_sample] = True
        else:
            for i in range(feature_norm.shape[0]):
                mask[i, topk.indices[i]] = True

        mask = mask.reshape(feature_norm.shape[0], feature_norm.shape[2], feature_norm.shape[3])

    return mask

def topk_coord_and_feat(feature_norm, k=3000, grid_h=1, grid_w=1, sample_aug=5):
    ### feature_norm shape: b*h*w
    ### mask: b*h*w

    b = feature_norm.shape[0]

    feature_norm[:,:,:5] = 0
    feature_norm[:,:,-5:] = 0
    feature_norm[:,:,:,:5] = 0
    feature_norm[:,:,:,-5:] = 0    

    mask = torch.zeros_like(feature_norm).to(dtype=torch.bool)
    mask = mask.squeeze(0)
    height = feature_norm.shape[2]
    width = feature_norm.shape[3]
    grid_height = int(height / grid_h)
    grid_width = int(width / grid_w)
    if k != -1:
        k_in_grid = round(k / (grid_h*grid_w))
    else:
        k_in_grid = -1

    for i in range(grid_h):
        for j in range(grid_w):
            start_h = i * grid_height
            start_w = j * grid_width
            feat_norm_grid = feature_norm[:, :, start_h:start_h+grid_height, start_w:start_w+grid_width]
            mask[:, start_h:start_h+grid_height, start_w:start_w+grid_width] = topk_coord_and_feat_single_grid(feat_norm_grid, sample_aug, k_in_grid)
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

def pt_sel(output, k_list, grid_list, sample_aug_list):
    feature1_norm = output[0]['feature_norm']
    
    for k in k_list:
        for g in grid_list:
            for sample_aug in sample_aug_list:
                mask1_topk = topk_coord_and_feat(feature1_norm, k=k, grid_h=g, grid_w=g, sample_aug=sample_aug)
                output[0]['mask_top_{}/grid_{}/sample_{}'.format(k, g, sample_aug)] = mask1_topk

    return

def pt_sel_log(sample_batch, output, output_folder, iter_overall, k_list, grid_list, sample_aug_list):
    feature1 = output[0]['feature_normalized']  
    ## before 10/23, the result is using ['feature'], which is centerred but not normalized, therefore is not consistent with what is used in training CVO loss
    ## but those result are all in sparsify_mode == 5, i.e. self.norm_dim = (1,2), L_norm == 1 which means the feature map is scaled altogether 
    imgname = sample_batch['imgname 1']

    feat_np1 = feat_to_np(feature1)

    for i in range(len(feat_np1)):
        feat_np1[i].tofile(os.path.join(output_folder, 'feature_map', 'feat_'+imgname[i] +'.bin') )
        print(iter_overall, 'imgname', i, ':', imgname[i])
        print("feat_np1[{}]: [{:.3f}, {:.3f}], {}".format(i, np.min(feat_np1[i]), np.max(feat_np1[i]), feat_np1[i].dtype))
        print("feature1[{}]: [{:.3f}, {:.3f}]".format(i, torch.min(feature1[i]), torch.max(feature1[i])))

    for k in k_list:
        for g in grid_list:
            for sample_aug in sample_aug_list:
                mask1_topk = output[0]['mask_top_{}/grid_{}/sample_{}'.format(k, g, sample_aug)]
                mask_np1 = mask_to_np(mask1_topk)
                for i in range(len(feat_np1)):
                    mask_np1[i].tofile(os.path.join(output_folder, 'mask_top_{}/grid_{}/sample_{}'.format(k, g, sample_aug), imgname[i] +'.bin') )
                    # print("mask_np1[{}]: [{}, {}], {}".format(i, np.min(mask_np1[i]), np.max(mask_np1[i]), mask_np1[i].dtype))
                    # print("mask1_topk[{}]: [{}, {}]".format(i, torch.min(mask1_topk[i]), torch.max(mask1_topk[i])))

    return

# def pt_sel_log(sample_batch, output, output_folder, iter_overall):
#     feature1 = output[0]['feature']  ## maybe should use feature_normalized, which is not calculated in eval_mode, so the code structure should be adjusted 
#     imgname = sample_batch['imgname 1']
#     mask1_topk = output[0]['mask_topk']

#     feat_np1 = feat_to_np(feature1)
#     mask_np1 = mask_to_np(mask1_topk)
#     # mask_np5k = mask_to_np(mask1_top5k)
#     # mask_np8k = mask_to_np(mask1_top8k)
#     # mask_np10k = mask_to_np(mask1_top10k)

#     for i in range(len(feat_np1)):
#         feat_np1[i].tofile(os.path.join(output_folder, 'feat_'+imgname[i] +'.bin') )
#         mask_np1[i].tofile(os.path.join(output_folder, 'mask_'+imgname[i] +'.bin') )
#         # mask_np5k[i].tofile(os.path.join(output_folder, 'mask5k_'+imgname[i] +'.bin') )
#         # mask_np8k[i].tofile(os.path.join(output_folder, 'mask8k_'+imgname[i] +'.bin') )
#         # mask_np10k[i].tofile(os.path.join(output_folder, 'mask10k_'+imgname[i] +'.bin') )

#         print(iter_overall, 'imgname', i, ':', imgname[i])