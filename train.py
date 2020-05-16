"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')

parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--snapshot_pe', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--hanet', nargs='*', type=int, default=[0,0,0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--hanet_set', nargs='*', type=int, default=[0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--hanet_pos', nargs='*', type=int, default=[0,0,0],
                    help='Row driven attention networks module')
parser.add_argument('--pos_rfactor', type=int, default=0,
                    help='number of position information, if 0, do not use')
parser.add_argument('--aux_loss', action='store_true', default=False,
                    help='auxilliary loss on intermediate feature map')
parser.add_argument('--attention_loss', type=float, default=0.0)
parser.add_argument('--hanet_poly_exp', type=float, default=0.0)
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')
parser.add_argument('--hanet_lr', type=float, default=0.0,
                    help='different learning rate on attention module')
parser.add_argument('--hanet_wd', type=float, default=0.0001,
                    help='different weight decay on attention module')                    
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--pos_noise', type=float, default=0.0)
parser.add_argument('--no_pos_dataset', action='store_true', default=False,
                    help='get dataset with position information')
parser.add_argument('--use_hanet', action='store_true', default=False,
                    help='use hanet')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')


args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

# if args.apex:
# Check that we are running with cuda as distributed is only supported for cuda.
torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                        init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.local_rank)

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    if args.attention_loss>0 and args.hanet[4]==0:
        print("last hanet is not defined !!!!")
        exit()

    train_loader, val_loader, train_obj = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    if args.aux_loss:
        criterion_aux = loss.get_loss_aux(args)
        net = network.get_net(args, criterion, criterion_aux)
    else:
        net = network.get_net(args, criterion)      

    for i in range(5):
        if args.hanet[i] > 0:
            args.use_hanet = True

    if (args.use_hanet and args.hanet_lr > 0.0):
        optim, scheduler, optim_at, scheduler_at = optimizer.get_optimizer_attention(args, net)
    else:
        optim, scheduler = optimizer.get_optimizer(args, net)
  
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0

    if args.snapshot:
        if (args.use_hanet and args.hanet_lr > 0.0):
            epoch, mean_iu = optimizer.load_weights_hanet(net, optim, optim_at, scheduler, scheduler_at,
                                args.snapshot, args.restore_optimizer)
            if args.restore_optimizer is True:
                iter_per_epoch = len(train_loader)
                i = iter_per_epoch * epoch
            else:
                epoch = 0
            print("mean_iu", mean_iu)
        else:
            epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                                args.snapshot, args.restore_optimizer)
            if args.restore_optimizer is True:
                iter_per_epoch = len(train_loader)
                i = iter_per_epoch * epoch
            else:
                epoch = 0

    if args.snapshot_pe:
        if (args.use_hanet and args.hanet_lr > 0.0):
            optimizer.load_weights_pe(net, args.snapshot_pe)
            #optimizer.freeze_pe(net)
        
    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):

    if (args.use_hanet and args.hanet_pos[1] == 0):  # embedding
        if args.hanet_lr > 0.0:
            validate(val_loader, net, criterion_val, optim, scheduler, epoch, writer, i, optim_at, scheduler_at)
        else:
            validate(val_loader, net, criterion_val, optim, scheduler, epoch, writer, i)

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        if (args.use_hanet and args.hanet_lr > 0.0):
            # validate(val_loader, net, criterion_val, optim, epoch, writer, i, optim_at)
            i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter, optim_at, scheduler_at)
            train_loader.sampler.set_epoch(epoch + 1)
            validate(val_loader, net, criterion_val, optim, scheduler, epoch+1, writer, i, optim_at, scheduler_at)
        else:
            # validate(val_loader, net, criterion_val, optim, epoch, writer, i)
            i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter)
            train_loader.sampler.set_epoch(epoch + 1)
            validate(val_loader, net, criterion_val, optim, scheduler, epoch+1, writer, i)

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                # if args.apex:
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()
        epoch += 1


def train(train_loader, net, optim, curr_epoch, writer, scheduler, max_iter, optim_at=None, scheduler_at=None):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    requires_attention = False

    if args.attention_loss>0:
        get_attention_gt = Generate_Attention_GT(args.dataset_cls.num_classes)
        criterion_attention = loss.get_loss_bcelogit(args)
        requires_attention = True
    
    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        # inputs = (2,3,713,713)
        # gts    = (2,713,713)
        if curr_iter >= max_iter:
            break
        start_ts = time.time()

        if args.no_pos_dataset:
            inputs, gts, _img_name = data
        elif args.pos_rfactor > 0:
            inputs, gts, _img_name, aux_gts, (pos_h, pos_w) = data
        else:
            inputs, gts, _img_name, aux_gts = data

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs, gts = inputs.cuda(), gts.cuda()

        optim.zero_grad()
        if optim_at is not None:
            optim_at.zero_grad()

        if args.no_pos_dataset:
            main_loss = net(inputs, gts=gts)        
            del inputs, gts
        else:
            if args.pos_rfactor > 0:
                outputs = net(inputs, gts=gts, aux_gts=aux_gts, pos=(pos_h, pos_w), attention_loss=requires_attention)
            else:
                outputs = net(inputs, gts=gts, aux_gts=aux_gts, attention_loss=requires_attention)

            if args.aux_loss:
                main_loss, aux_loss = outputs[0], outputs[1]
                if args.attention_loss>0:
                    attention_map = outputs[2]
                    attention_labels = get_attention_gt(aux_gts, attention_map.shape)
                    # print(attention_map.shape, attention_labels.shape)
                    attention_loss = criterion_attention(input=attention_map.transpose(1,2), target=attention_labels.transpose(1,2))
            else:
                if args.attention_loss>0:
                    main_loss = outputs[0]
                    attention_map = outputs[1]
                    attention_labels = get_attention_gt(aux_gts, attention_map.shape)
                    # print(attention_map.shape, attention_labels.shape)
                    attention_loss = criterion_attention(input=attention_map.transpose(1,2), target=attention_labels.transpose(1,2))
                else:
                    main_loss = outputs

            del inputs, gts, aux_gts

        if args.no_pos_dataset:
            total_loss = main_loss
        elif args.attention_loss>0:
            if args.aux_loss:
                total_loss = main_loss + (0.4 * aux_loss) + (args.attention_loss * attention_loss)
            else:
                total_loss = main_loss + (args.attention_loss * attention_loss)
        else:
            if args.aux_loss:
                total_loss = main_loss + (0.4 * aux_loss)
            else:
                total_loss = main_loss

        log_total_loss = total_loss.clone().detach_()
        torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
        log_total_loss = log_total_loss / args.world_size
        train_total_loss.update(log_total_loss.item(), batch_pixel_size)

        total_loss.backward()
        optim.step()
        if optim_at is not None:
            optim_at.step()

        scheduler.step()
        if scheduler_at is not None:
            scheduler_at.step()

        time_meter.update(time.time() - start_ts)

        del total_loss, log_total_loss

        curr_iter += 1

        if args.local_rank == 0:
            if i % 50 == 49:
                if optim_at is not None:
                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [lr_at {:0.6f}], [time {:0.4f}]'.format(
                    curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                    optim.param_groups[-1]['lr'], optim_at.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)
                else:
                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                        optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)
    
                logging.info(msg)

                # Log tensorboard metrics for each iteration of the training phase
                writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                curr_iter)
                train_total_loss.reset()
                time_meter.reset()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, optim_at=None, scheduler_at=None):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        if args.no_pos_dataset:
            inputs, gt_image, img_names = data
        elif args.pos_rfactor > 0:
            inputs, gt_image, img_names, _, (pos_h, pos_w) = data
        else:
            inputs, gt_image, img_names, _ = data

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            if args.pos_rfactor > 0:
                if args.use_hanet and args.hanet_pos[0] > 0:  # use hanet and position
                    output, attention_map, pos_map = net(inputs, pos=(pos_h, pos_w), attention_map=True)
                else:
                    output = net(inputs, pos=(pos_h, pos_w))
            else:
                output = net(inputs)

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == args.dataset_cls.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             args.dataset_cls.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        if optim_at is not None:
            evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                        writer, curr_epoch, args.dataset_cls, curr_iter, optim_at, scheduler_at)
        else:
            evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                        writer, curr_epoch, args.dataset_cls, curr_iter)
        if args.use_hanet and args.hanet_pos[0] > 0:  # use pos and hanet
            visualize_attention(writer, attention_map, curr_iter)
            #if args.hanet_pos[1] == 0:  # embedding
            #    visualize_pos(writer, pos_map, curr_iter)

    return val_loss.avg

num_vis_pos = 0

def visualize_pos(writer, pos_maps, iteration):
    global num_vis_pos
    #if num_vis_pos % 5 == 0:
    #    save_pos_numpy(pos_maps, iteration)
    num_vis_pos += 1

    stage = 'valid'
    for i in range(len(pos_maps)):
        pos_map = pos_maps[i]
        if isinstance(pos_map, tuple):
            num_pos = 2
        else:
            num_pos = 1

        for j in range(num_pos):
            if num_pos == 2:
                pos_embedding = pos_map[j]
            else:
                pos_embedding = pos_map

            H, D = pos_embedding.shape
            pos_embedding = pos_embedding.unsqueeze(0)  # 1 X H X D
            if H > D:   # e.g. 32 X 8
                pos_embedding = F.interpolate(pos_embedding, H, mode='nearest') # 1 X 32 X 8
                D = H
            elif H < D:   # H < D, e.g. 32 X 64
                pos_embedding = F.interpolate(pos_embedding.transpose(1,2), D, mode='nearest').transpose(1,2) # 1 X 32 X 64
                H = D
            if args.hanet_pos[1]==1: # pos encoding
                pos_embedding = torch.cat((torch.ones(1, H, D).cuda(), pos_embedding/2, pos_embedding/2), 0)
            else:   # pos embedding
                pos_embedding = torch.cat((torch.ones(1, H, D).cuda(), torch.sigmoid(pos_embedding*20),
                                        torch.sigmoid(pos_embedding*20)), 0)
            pos_embedding = vutils.make_grid(pos_embedding, padding=5, normalize=False, range=(0,1))
            writer.add_image(stage + '/Pos/layer-' + str(i) + '-' + str(j), pos_embedding, iteration)

def save_pos_numpy(pos_maps, iteration):
    file_fullpath = '/home/userA/shchoi/Projects/visualization/pos_data/'
    file_name = str(args.date) + '_' + str(args.hanet_pos[0]) + '_' + str(args.exp) + '_layer'

    for i in range(len(pos_maps)):
        pos_map = pos_maps[i]
        if isinstance(pos_map, tuple):
            num_pos = 2
        else:
            num_pos = 1

        for j in range(num_pos):
            if num_pos == 2:
                pos_embedding = pos_map[j]
            else:
                pos_embedding = pos_map

            H, D = pos_embedding.shape
            pos_embedding = pos_embedding.data.cpu().numpy()   # H X D
            file_name_post = str(i) + '_' + str(j) + '_' + str(H) + 'X' + str(D) + '_' + str(iteration)
            np.save(file_fullpath + file_name + file_name_post, pos_embedding)

def visualize_attention(writer, attention_map, iteration, threshold=0):
    stage = 'valid'
    for i in range(len(attention_map)):
        C = attention_map[i].shape[1]
        #H = alpha[2].shape[2]
        attention_map_sb = F.interpolate(attention_map[i], C, mode='nearest')
        attention_map_sb = attention_map_sb[0].transpose(0,1).unsqueeze(0)  # 1 X H X C X 1, 
        attention_map_sb = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(attention_map_sb - 1.0),
                        torch.abs(attention_map_sb - 1.0)), 0)
        attention_map_sb = vutils.make_grid(attention_map_sb, padding=5, normalize=False, range=(threshold,1))
        writer.add_image(stage + '/Attention/Row-wise-' + str(i), attention_map_sb, iteration)

from threading import Thread
#import cupy as cp
    
class Generate_Attention_GT(object):   # 34818
    def __init__(self, n_classes=19):
        self.channel_weight_factor = 0   # TBD
        self.ostride = 0
        self.labels = 0
        self.attention_labels = 0
        self.n_classes = n_classes

    def rows_hasclass(self, B, C):
        rows = cp.where(self.labels[B]==C)[0]
        if len(rows) > 0:
            row = cp.asnumpy(cp.unique((rows//self.ostride), return_counts=False))
            print("channel", C, "row", row)
            self.attention_labels[B][C][row] = 1

    def __call__(self, labels, attention_size):
        B, C, H = attention_size
        # print(labels.shape, attention_size)
        self.labels = cp.asarray(labels)
        self.attention_labels = torch.zeros(B, self.n_classes, H).cuda()
        self.ostride = labels.shape[1] // H

        # threads = []
        for j in range(0, labels.shape[0]):
            for k in range(0, self.n_classes):
                rows = cp.where(self.labels[j]==k)[0]
                if len(rows) > 0:
                    row = cp.asnumpy(cp.unique((rows//self.ostride), return_counts=False))
                    # print("channel", k, "row", row)
                    self.attention_labels[j][k][row] = 1

        return self.attention_labels


if __name__ == '__main__':
    main()
