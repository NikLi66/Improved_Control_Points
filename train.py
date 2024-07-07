import os, sys
import argparse
import torch
from torch.autograd import Variable
from utils.trainer import Trainer
import warnings
import time
from models.network import ImprovedControlPoints
from dataset.dataloader import DewarpDataSet
from models.loss import Losses

def train(args):
    # 初始化模型
    model = ImprovedControlPoints(n_classes=2, num_filter=32, BatchNorm='BN', in_channels=3)

    start_epoch = 0

    # 并行训练
    if args.distributed:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.parallel
        torch.distributed.init_process_group(backend = 'nccl')
        args.rank = torch.distributed.get_rank()
        args.device = torch.device('cuda:{}'.format(args.rank))
        torch.cuda.set_device(args.device)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    elif args.parallel is not None:
        device_ids = [int(x) for x in args.parallel.split(',')]
        args.rank = 0
        args.device = torch.device('cuda:'+str(args.rank))
        torch.cuda.set_device(args.device)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda()
    else:
        warnings.warn('no found gpu')
        exit()

    # 设置优化器
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, weight_decay=1e-12)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.l_rate, weight_decay=1e-10)
    else:
        assert 'please choice optimizer'
        exit('error')

    # 加载预训练模型
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model_state'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'])) #改.format(args.resume.name, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume.name))

    # 设置loss
    loss_class = Losses(k=4)
    loss_dict = {}
    for loss_func in dir(loss_class):
        if loss_func.startswith('loss') and loss_func in args:
            loss_dict[loss_func] = {"weight": getattr(args, loss_func), "function": getattr(loss_class, loss_func)}

    # 训练
    trainer = Trainer(args=args, model=model, output_path=args.output_path, start_epoch=start_epoch, optimizer=optimizer, loss_dict=loss_dict, \
                 dataset=DewarpDataSet, data_path=args.data_path, data_path_validate=args.data_path_validate, validate=True)          # , valloaderSet=valloaderSet, v_loaderSet=v_loaderSet
    ''' load data '''

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=300,
                        help='# of the epochs')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization')

    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--print-freq', '-p', default=60, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency

    parser.add_argument('--data_path', default="", type=str,
                        help='the path of train images.')  # train image path

    parser.add_argument('--data_path_validate', default="", type=str,
                        help='the path of validate images.')  # validate image path

    parser.add_argument('--output_path', default="./outputs/test/", type=str, help='the path is used to  save output --img or result.')

    parser.add_argument('--resume', default=None, type=str,
                        help='Path to previous saved model to restart from')

    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
                        help='Batch Size')


    parser.add_argument('--parallel', default='0', type=str,
                        help='choice the gpu id for parallel ')

    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')

    parser.add_argument('--local_rank',type=int,default=0,metavar='N')

    parser.add_argument('--loss_regress', default=1, type=int,
                        help='Weight of the regression loss')

    parser.add_argument('--loss_segment', default=1, type=int,
                        help='Weight of the segmentation loss')



    args = parser.parse_args()

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise Exception(args.resume+' -- not exist')

    if os.path.exists(args.output_path) is False:
        os.makedirs(args.output_path, exist_ok=True)

    train(args)
