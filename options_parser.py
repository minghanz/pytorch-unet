import argparse

class ManualOptions:
    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description='Overwrite some params.')
        self.parser.add_argument('--min_dist_mode', action='store_true')
        self.parser.add_argument('--diff_mode', action='store_true')
        self.parser.add_argument('--min_grad_mode', action='store_true')
        self.parser.add_argument('--continue_train', action='store_true')
        self.parser.add_argument('--lr', type=float, default=1e-5)
        self.parser.add_argument('--run_eval', action='store_true')
        self.parser.add_argument('--pretrained_weight_path', type=str)
        self.parser.add_argument('--dist_coef_feature', type=float, default=0.1)
        # self.parser.add_argument('--reg_norm_weight', type=float, default=1)
        self.parser.add_argument('--self_sparse_mode', action='store_true')
        self.parser.add_argument('--self_trans', action='store_true')
        
        

    
    def parse(self):
        options = self.parser.parse_args()
        return options