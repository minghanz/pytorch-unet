# for lr in 1e-3 1e-5 1e-7
# do
#     echo "new trial 1, lr: "$lr
#     python run.py --min_grad_mode --continue_train --lr=$lr
#     echo "new trial 2, lr: "$lr
#     python run.py --min_dist_mode --lr=$lr
#     echo "new trial 3, lr: "$lr
#     python run.py --min_dist_mode --min_grad_mode --lr=$lr
#     echo "new trial 4, lr: "$lr
#     python run.py --min_dist_mode --min_grad_mode --diff_mode --lr=$lr
# done

for lr in 1e-4 #1e-5
do
    for dist_coef_feature in 0.1 #0.2
    do
        # echo "new trial 1, lr: "$lr
        # python run.py --min_dist_mode --min_grad_mode --lr=$lr --dist_coef_feature=$dist_coef_feature
        # echo "new trial 2, lr: "$lr
        # python run.py --min_grad_mode --lr=$lr
        echo "new trial 3, lr: "$lr
        python run.py --min_dist_mode --lr=$lr --dist_coef_feature=$dist_coef_feature
        # echo "new trial 4, lr: "$lr
        # python run.py --min_dist_mode --lr=$lr --dist_coef_feature=$dist_coef_feature --self_trans
    done
done

# python run.py --self_sparse_mode --lr=1e-4