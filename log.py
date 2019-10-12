import torchvision
import torch

def scale_to_tensorboard(losses, writer, unet_options, loss_options, iter_overall, output=None):
    for item in losses:
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
    
    img1_raw = sample_batch['image 1 raw']
    img2_raw = sample_batch['image 2 raw']
    
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


    depth1 = sample_batch['depth 1']
    depth2 = sample_batch['depth 2']

    depth1 = torch.where(depth1 > 0, 1/depth1, torch.zeros_like(depth1))
    depth2 = torch.where(depth2 > 0, 1/depth2, torch.zeros_like(depth2))
    
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

    if unet_options.run_eval:
        mask1_topk = output[0]['mask_topk']

        grid1_mask = torchvision.utils.make_grid(mask1_topk)
        writer.add_image('mask1_topk',grid1_mask, iter_overall)
        # grid1_mask = torchvision.utils.make_grid(mask1_top5k)
        # writer.add_image('mask1_top5k',grid1_mask, iter_overall)
        # grid1_mask = torchvision.utils.make_grid(mask1_top8k)
        # writer.add_image('mask1_top8k',grid1_mask, iter_overall)
        # grid1_mask = torchvision.utils.make_grid(mask1_top10k)
        # writer.add_image('mask1_top10k',grid1_mask, iter_overall)

    if loss_options.pca_in_loss or loss_options.visualize_pca_chnl:
        feature1_chnl3 = output[0]['feature_chnl3']
        feature2_chnl3 = output[1]['feature_chnl3']

        # The tensorboard visualize value in (-1,0) the same as in (0, 1), e.g. -1.9 = -0.9 = 0.1 = 1.1, 1 is the brightest
        feat1_pos = vis_feat(feature1_chnl3)
        feat1_neg = vis_feat(feature1_chnl3, neg=True)
        feat2_pos = vis_feat(feature2_chnl3)
        feat2_neg = vis_feat(feature2_chnl3, neg=True)

        grid1pos = torchvision.utils.make_grid(feat1_pos)
        grid1neg = torchvision.utils.make_grid(feat1_neg)
        grid2pos = torchvision.utils.make_grid(feat2_pos)
        grid2neg = torchvision.utils.make_grid(feat2_neg)

        writer.add_image('img1/feat_pos',grid1pos, iter_overall)
        writer.add_image('img1/feat_neg',grid1neg, iter_overall)
        writer.add_image('img2/feat_pos',grid2pos, iter_overall)
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
    marg = 3
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