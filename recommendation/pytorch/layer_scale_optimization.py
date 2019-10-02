import torch
import torch.nn as nn
import os
import sys
import numpy as np
from tqdm import tqdm
import math
from neumf import NeuMF
import random
import torch.backends.cudnn as cudnn
from itertools import count
from pathlib import Path

sys.path.insert(0, '/home/cvds_lab/yury/mxt-experiments/nn-quantization-pytorch')
from quantization.quantizer import ModelQuantizer
from quantization.posttraining.module_wrapper import ActivationModuleWrapperPost, ParameterModuleWrapperPost
from quantization.methods.clipped_uniform import FixedClipValueQuantization
from utils.mllog import MLlogger
import scipy.optimize as opt


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
    parser.add_argument('--seed', '-s', type=int, default=0,
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

    parser.add_argument('--min_method', '-mm', help='Minimization method to use [Nelder-Mead, Powell, COBYLA]',
                        default='Powell')
    parser.add_argument('--maxiter', '-maxi', type=int, help='Maximum number of iterations to minimize algo',
                        default=None)
    parser.add_argument('--maxfev', '-maxf', type=int, help='Maximum number of function evaluations of minimize algo',
                        default=None)

    parser.add_argument('--init_method', default='static',
                        help='Scale initialization method [static, dynamic, random], default=static')
    parser.add_argument('-siv', type=float, help='Value for static initialization', default=1.)

    parser.add_argument('--dont_fix_np_seed', '-dfns', action='store_true',
                        help='Do not fix np seed even if seed specified')

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


class NcfData(object):
    def __init__(self, test_users, test_items, dup_mask, real_indices, K, samples_per_user, num_user):
        self.test_users = test_users
        self.test_items = test_items
        self.dup_mask = dup_mask
        self.real_indices = real_indices
        self.K = K
        self.samples_per_user = samples_per_user
        self.num_user = num_user

    def get_subset(self, N):
        return NcfData(self.test_users[:N], self.test_items[:N], self.dup_mask[:N], self.real_indices[:N], self.K,
                       self.samples_per_user, self.num_user)

    def remove_last(self, N):
        return NcfData(self.test_users[N:], self.test_items[N:], self.dup_mask[N:], self.real_indices[N:], self.K,
                       self.samples_per_user, self.num_user)


class CalibrationSet(object):
    def __init__(self, f_path):
        data_ = torch.load(f_path)
        self.users = data_['users']
        self.items = data_['items']
        self.labels = data_['labels']

    def cuda(self):
        self.users = self.users.cuda()
        self.items = self.items.cuda()
        self.labels = self.labels.cuda()
        return self

    def split(self, batch_size):
        self.users = self.users.split(batch_size)
        self.items = self.items.split(batch_size)
        self.labels = self.labels.split(batch_size)


def set_clipping(mq, clipping, device, verbose=False):
    qwrappers = mq.get_qwrappers()
    for i, qwrapper in enumerate(qwrappers):
        qwrapper.set_quantization(FixedClipValueQuantization,
                                  {'clip_value': clipping[i], 'device': device}, verbose=verbose)


def get_clipping(mq):
    clipping = []
    qwrappers = mq.get_qwrappers()
    for i, qwrapper in enumerate(qwrappers):
        q = qwrapper.get_quantization()
        clip_value = getattr(q, 'alpha')
        clipping.append(clip_value)

    return np.array(clipping)


def val(model, data):
    print('Validation ...')
    log_2 = math.log(2)

    model.eval()
    hits = torch.tensor(0., device='cuda')
    ndcg = torch.tensor(0., device='cuda')

    with torch.no_grad():
        list_ = list(enumerate(zip(data.test_users, data.test_items)))
        for i, (u,n) in tqdm(list_):
            res = model(u.cuda().view(-1), n.cuda().view(-1), sigmoid=True).detach().view(-1, data.samples_per_user)
            # set duplicate results for the same item to -1 before topk
            res[data.dup_mask[i]] = -1
            out = torch.topk(res, data.K)[1]
            # topk in pytorch is stable(if not sort)
            # key(item):value(predicetion) pairs are ordered as original key(item) order
            # so we need the first position of real item(stored in real_indices) to check if it is in topk
            ifzero = (out == data.real_indices[i].cuda().view(-1,1))
            hits_ = ifzero.sum()
            ndcg_ = (log_2 / (torch.nonzero(ifzero)[:, 1].view(-1).to(torch.float)+2).log_()).sum()
            hits += hits_
            ndcg += ndcg_

    hits = hits.item()
    ndcg = ndcg.item()

    return hits / data.num_user, ndcg / data.num_user


def evaluate_calibration(model, cal_data, criterion):
    total_loss = torch.tensor([0.]).cuda()
    for i in range(len(cal_data.users)):
        outputs = model(cal_data.users[i].view(-1), cal_data.items[i].view(-1), sigmoid=True)
        loss = criterion(outputs.view(-1), cal_data.labels[i])
        total_loss += loss

    loss = total_loss.item() / len(cal_data.users)
    return loss


def validate(model, data):
    hr, ndcg = val(model, data)
    print('')
    print('')
    print('HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
          .format(K=data.K, hit_rate=hr, ndcg=ndcg))
    return hr, ndcg


_eval_count = count(0)
_min_loss = 1e6
def run_inference_on_calibration(scales, model, mq, cal_data, criterion):
    global _eval_count, _min_loss
    eval_count = next(_eval_count)

    set_clipping(mq, scales, model.device, verbose=(eval_count % 300 == 0))
    loss = evaluate_calibration(model, cal_data, criterion)

    if loss < _min_loss:
        _min_loss = loss

    print_freq = 20
    if eval_count % 20 == 0:
        print("func eval iteration: {}, minimum loss of last {} iterations: {:.4f}".format(
            eval_count, print_freq, _min_loss))

    return loss


def main(args, ml_logger):
    # Fix the seed
    random.seed(args.seed)
    if not args.dont_fix_np_seed:
        np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        model.device = torch.device('cuda:{}'.format(0))

    if args.load_ckp:
        ckp = torch.load(args.load_ckp)
        model.load_state_dict(ckp)

    all_embeding = [n for n, m in model.named_modules() if isinstance(m, nn.Embedding)]
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    all_relu = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
    all_relu6 = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU6)]
    layers = all_relu + all_relu6 + all_linear + all_embeding
    replacement_factory = {nn.ReLU: ActivationModuleWrapperPost,
                           nn.ReLU6: ActivationModuleWrapperPost,
                           nn.Linear: ParameterModuleWrapperPost,
                           nn.Embedding: ActivationModuleWrapperPost}
    mq = ModelQuantizer(model, args, layers, replacement_factory)
    # mq.log_quantizer_state(ml_logger, -1)

    test_users, test_items, dup_mask, real_indices, K, samples_per_user, num_user = data_loader(args.data)
    data = NcfData(test_users, test_items, dup_mask, real_indices, K, samples_per_user, num_user)
    cal_data = CalibrationSet('ml-20mx16x32/cal_set').cuda()
    cal_data.split(batch_size=10000)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = criterion.cuda()

    print("init_method: {}, qtype {}".format(args.init_method, args.qtype))
    # evaluate to initialize dynamic clipping
    loss = evaluate_calibration(model, cal_data, criterion)
    print("Initial loss: {:.4f}".format(loss))

    # get clipping values
    init = get_clipping(mq)

    # evaluate
    hr, ndcg = validate(model, data)
    ml_logger.log_metric('HR init', hr, step='auto')

    # run optimizer
    min_options = {}
    if args.maxiter is not None:
        min_options['maxiter'] = args.maxiter
    if args.maxfev is not None:
        min_options['maxfev'] = args.maxfev

    _iter = count(0)

    def local_search_callback(x):
        it = next(_iter)
        loss = run_inference_on_calibration(x, model, mq, cal_data, criterion)
        print("\n[{}]: Local search callback".format(it))
        print("loss: {:.4f}\n".format(loss))

    res = opt.minimize(lambda scales: run_inference_on_calibration(scales, model, mq, cal_data, criterion), np.array(init),
                       method=args.min_method, options=min_options, callback=local_search_callback)

    print(res)
    scales = res.x
    set_clipping(mq, scales, model.device)
    # evaluate
    hr, ndcg = validate(model, data)
    ml_logger.log_metric('HR Powell', hr, step='auto')
    # save scales

home = str(Path.home())
if __name__ == '__main__':
    args = parse_args()
    with MLlogger(os.path.join(home, 'mxt-sim/mllog_runs'), args.experiment, args,
                  name_args=['NCF', '1B', "W{}A{}".format(args.bit_weights, args.bit_act)]) as ml_logger:
        main(args, ml_logger)
