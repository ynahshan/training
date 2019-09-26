import torch
import torch.nn as nn
import os
import sys
import numpy as np
from tqdm import tqdm
import math
from neumf import NeuMF
import pickle

sys.path.append('/home/cvds_lab/yury/mxt-experiments/nn-quantization-pytorch')
from quantization.quantizer import ModelQuantizer
from quantization.posttraining.module_wrapper import ActivationModuleWrapperPost, ParameterModuleWrapperPost


def save_data(data_, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data_, f)

def load_data(fname):
    with open(fname, 'rb') as f:
        data_ = pickle.load(f)
    return data_



from argparse import ArgumentParser
import pickle

def parse_args():
    parser = ArgumentParser(description="Validate a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='path to test data files')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--load_ckp', type=str, default=None,
                        help='Path to load checkpoint from.')

    parser.add_argument('--quantize', '-q', action='store_true', help='Enable quantization', default=False)
    parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
    parser.add_argument('--bit_weights', '-bw', type=int, help='Number of bits for weights', default=None)
    parser.add_argument('--bit_act', '-ba', type=int, help='Number of bits for activations', default=None)
    parser.add_argument('--pre_relu', dest='pre_relu', action='store_true', help='use pre-ReLU quantization')
    parser.add_argument('--qtype', default='max_static', help='Type of quantization method')
    parser.add_argument('-lp', type=float, help='p parameter of Lp norm', default=3.)

    return parser.parse_args()


def data_loader(path):
    # load data:
    print('Data loading ...')
    data_ = load_data(path)
    test_users = []
    test_items = []
    dup_mask = []
    real_indices = []
    for i in tqdm(range(len(data_['test_users']))):
        test_users.append(torch.tensor(data_['test_users'][i]))
        test_items.append(torch.tensor(data_['test_items'][i]))
        dup_mask.append(torch.tensor(data_['dup_mask'][i]))
        real_indices.append(torch.tensor(data_['real_indices'][i]))

    K = data_['K']
    samples_per_user = data_['samples_per_user']
    num_user = data_['num_user']

    return test_users, test_items, dup_mask, real_indices, K, samples_per_user, num_user


def val(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user):
    print('Validation ...')
    log_2 = math.log(2)

    model.eval()
    hits = torch.tensor(0., device='cuda')
    ndcg = torch.tensor(0., device='cuda')

    with torch.no_grad():
        list_ = list(enumerate(zip(x,y)))
        for i, (u,n) in tqdm(list_):
            res = model(u.cuda().view(-1), n.cuda().view(-1), sigmoid=True).detach().view(-1,samples_per_user)
            # set duplicate results for the same item to -1 before topk
            res[dup_mask[i]] = -1
            out = torch.topk(res,K)[1]
            # topk in pytorch is stable(if not sort)
            # key(item):value(predicetion) pairs are ordered as original key(item) order
            # so we need the first position of real item(stored in real_indices) to check if it is in topk
            ifzero = (out == real_indices[i].cuda().view(-1,1))
            hits_ = ifzero.sum()
            ndcg_ = (log_2 / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()
            hits += hits_
            ndcg += ndcg_

    hits = hits.item()
    ndcg = ndcg.item()

    return hits/num_user, ndcg/num_user


def main():
    args = parse_args()

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Create model
    model = NeuMF(2197225, 855776,
                  mf_dim=64, mf_reg=0.,
                  mlp_layer_sizes=[256, 256, 128, 64],
                  mlp_layer_regs=[0. for i in [256, 256, 128, 64]])

    print(model)

    if use_cuda:
        # Move model and loss to GPU
        model = model.cuda()

    if args.load_ckp:
        ckp = torch.load(args.load_ckp)
        model.load_state_dict(ckp)

    if args.quantize:
        all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        all_relu = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
        all_relu6 = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU6)]
        layers = all_relu + all_relu6 + all_linear
        replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                               nn.ReLU6: ActivationModuleWrapperPost,
                               nn.Linear: ParameterModuleWrapperPost}
        mq = ModelQuantizer(model, args, layers, replacement_factory)
        # mq.log_quantizer_state(ml_logger, -1)

    test_users, test_items, dup_mask, real_indices, K, samples_per_user, num_user = data_loader(args.data)
    hr, ndcg = val(model, test_users, test_items, dup_mask, real_indices, K, samples_per_user, num_user)
    print('')
    print('')
    print('HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
          .format(K=K, hit_rate=hr, ndcg=ndcg))


if __name__ == '__main__':
    main()
