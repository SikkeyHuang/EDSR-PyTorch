import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

test_dataset = ['x0','x1','../test2','../test3','../test4','../test']
#test_models = ['x0','x1','../models/edsr_baseline_x2-1bc95232.pt','../models/edsr_baseline_x3-abf2a44e.pt','../models/edsr_baseline_x4-6b446fab.pt','../models/MDSR.pt']
#test_models = ['x0','x1','../models/EDSR_x2.pt','../models/EDSR_x3.pt','../models/EDSR_x4.pt','../models/MDSR.pt']

test_model = '../models/mdsr_baseline-a00cab12.pt'
#test_model = '../models/EDSR_x3.pt'
scales = '2+3+4'

args.scale = list(map(lambda x: int(x), scales.split('+')))
args.pre_train = test_model
args.model='MDSR'
#args.n_feats = 64
#args.n_resblocks=16
#args.G0=256
args.dir_demo= test_dataset[5]


args.test_only = 'True'
args.save_results = 'True'

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)

        t.test()
        checkpoint.done()

if __name__ == '__main__':
    main()


