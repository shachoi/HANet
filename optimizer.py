"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from config import cfg


def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    if args.backbone_lr > 0.0:
        base_params = []
        resnet_params = []
        resnet_name = []
        resnet_name.append('layer0')
        resnet_name.append('layer1')
        #resnet_name.append('layer2')
        #resnet_name.append('layer3')
        #resnet_name.append('layer4')
        len_resnet = len(resnet_name)
    else:
        param_groups = net.parameters()

    if args.backbone_lr > 0.0:
        for name, param in net.named_parameters():
            is_resnet = False
            for i in range(len_resnet):
                if resnet_name[i] in name:
                    resnet_params.append(param)
                    # param.requires_grad=False
                    print("resnet_name", name)
                    is_resnet = True
                    break
            if not is_resnet:
                base_params.append(param)

    if args.sgd:
        if args.backbone_lr > 0.0:
            optimizer = optim.SGD([
                                    {'params': base_params},
                                    {'params': resnet_params, 'lr':args.backbone_lr}
                                ],
                                lr=args.lr,
                                weight_decay=5e-4, #args.weight_decay,
                                momentum=args.momentum,
                                nesterov=False)
        else:
            optimizer = optim.SGD(param_groups,
                                lr=args.lr,
                                weight_decay=5e-4, #args.weight_decay,
                                momentum=args.momentum,
                                nesterov=False)
    else:
        raise ValueError('Not a valid optimizer')

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_ITER == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_ITER
        scale_value = args.rescale
        lambda1 = lambda iteration: \
             math.pow(1 - iteration / args.max_iter,
                      args.poly_exp) if iteration < rescale_thresh else scale_value * math.pow(
                          1 - (iteration - rescale_thresh) / (args.max_iter - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly':
        lambda1 = lambda iteration: math.pow(1 - iteration / args.max_iter, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler


def get_optimizer_attention(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    attention_params = []
    base_params = []
    hanet_name = []

    if args.backbone_lr > 0.0:
        resnet_params = []
        resnet_name = []
        resnet_name.append('layer0')
        resnet_name.append('layer1')
        #resnet_name.append('layer2')
        #resnet_name.append('layer3')
        #resnet_name.append('layer4')
        len_resnet = len(resnet_name)

    for i in range(5):
        if args.hanet[i] > 0:    # HANet_Diff
            hanet_name.append('hanet' + str(i))

    len_hanet = len(hanet_name)

    for name, param in net.named_parameters():
        is_hanet = False
        is_resnet = False
        if args.backbone_lr > 0.0:
            for i in range(len_resnet):
                if resnet_name[i] in name:
                    resnet_params.append(param)
                    # param.requires_grad=False
                    print("resnet_name", name)
                    is_resnet = True
                    break
        if not is_resnet:
            for i in range(len_hanet):
                if hanet_name[i] in name:
                    attention_params.append(param)
                    #print("hanet_name", name)
                    is_hanet = True
                    break
        if not is_hanet and not is_resnet:
            base_params.append(param)
            #print("base", name)

    if args.sgd:
        if args.backbone_lr > 0.0:
            optimizer = optim.SGD([
                                    {'params': base_params},
                                    {'params': resnet_params, 'lr':args.backbone_lr}
                                ],
                                lr=args.lr,
                                weight_decay=5e-4, #args.weight_decay,
                                momentum=args.momentum,
                                nesterov=False)
        else:
            optimizer = optim.SGD(base_params,
                                lr=args.lr,
                                weight_decay=5e-4, #args.weight_decay,
                                momentum=args.momentum,
                                nesterov=False)
    else:
        raise ValueError('Not a valid optimizer')

    print(" ############# HANet Number", len_hanet)
    optimizer_at = optim.SGD(attention_params,
                            lr=args.hanet_lr,
                            weight_decay=args.hanet_wd,
                            momentum=args.momentum,
                            nesterov=False)



    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_ITER == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_ITER
        scale_value = args.rescale
        lambda1 = lambda iteration: \
             math.pow(1 - iteration / args.max_iter,
                      args.poly_exp) if iteration <= rescale_thresh else scale_value * math.pow(
                          1 - (iteration - rescale_thresh) / (args.max_iter - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        if args.hanet_poly_exp > 0.0:
            lambda2 = lambda iteration: \
                math.pow(1 - iteration / args.max_iter,
                        args.hanet_poly_exp) if iteration <= rescale_thresh else scale_value * math.pow(
                            1 - (iteration - rescale_thresh) / (args.max_iter - rescale_thresh),
                            args.repoly)
            scheduler_at = optim.lr_scheduler.LambdaLR(optimizer_at, lr_lambda=lambda2)
        else:
            lambda2 = lambda iteration: \
                math.pow(1 - iteration / args.max_iter,
                        args.poly_exp) if iteration <= rescale_thresh else scale_value * math.pow(
                            1 - (iteration - rescale_thresh) / (args.max_iter - rescale_thresh),
                            args.repoly)
            scheduler_at = optim.lr_scheduler.LambdaLR(optimizer_at, lr_lambda=lambda2)    

    elif args.lr_schedule == 'poly':
        lambda1 = lambda iteration: math.pow(1 - iteration / args.max_iter, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        # for attention module
        if args.hanet_poly_exp > 0.0:
            lambda2 = lambda iteration: math.pow(1 - iteration / args.max_iter, args.hanet_poly_exp)
            scheduler_at = optim.lr_scheduler.LambdaLR(optimizer_at, lr_lambda=lambda2)
        else:
            lambda2 = lambda iteration: math.pow(1 - iteration / args.max_iter, args.poly_exp)
            scheduler_at = optim.lr_scheduler.LambdaLR(optimizer_at, lr_lambda=lambda2)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler, optimizer_at, scheduler_at 


def get_optimizer_by_epoch(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    param_groups = net.parameters()

    if args.sgd:
        optimizer = optim.SGD(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    elif args.adam:
        amsgrad = False
        if args.amsgrad:
            amsgrad = True
        optimizer = optim.Adam(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=amsgrad
                               )
    else:
        raise ValueError('Not a valid optimizer')

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_EPOCH == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_EPOCH
        scale_value = args.rescale
        lambda1 = lambda epoch: \
             math.pow(1 - epoch / args.max_epoch,
                      args.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                          1 - (epoch - rescale_thresh) / (args.max_epoch - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / args.max_epoch, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler

def load_weights_hanet(net, optimizer, optimizer_at, scheduler, scheduler_at, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer, optimizer_at, scheduler, scheduler_at, epoch, mean_iu = restore_snapshot_hanet(net, optimizer,
            optimizer_at, scheduler, scheduler_at, snapshot_file, restore_optimizer_bool)
    return epoch, mean_iu

def load_weights_pe(net, snapshot_file):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net = restore_snapshot_pe(net, snapshot_file)


def load_weights(net, optimizer, scheduler, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer, scheduler, epoch, mean_iu = restore_snapshot(net, optimizer, scheduler, snapshot_file,
            restore_optimizer_bool)
    return epoch, mean_iu

def restore_snapshot_pe(net, snapshot):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint PE Load Compelete")

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore_only_pe(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore_only_pe(net, checkpoint)

    return net

def forgiving_state_restore_only_pe(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            if 'pos_emb1d' in k:
                print("matched loading parameter", k)
                new_loaded_dict[k] = loaded_dict[k]
        # else:
        #     print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def freeze_pe(net):
    for name, param in net.named_parameters():
        if 'pos_emb1d' in name:
            print("freeze parameter", name)
            param.requires_grad = False

def restore_snapshot_hanet(net, optimizer, optimizer_at, scheduler, scheduler_at, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if optimizer_at is not None and 'optimizer_at' in checkpoint and restore_optimizer_bool:
        optimizer_at.load_state_dict(checkpoint['optimizer_at'])

    if scheduler is not None and 'scheduler' in checkpoint and restore_optimizer_bool:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if scheduler_at is not None and 'scheduler_at' in checkpoint and restore_optimizer_bool:
        scheduler_at.load_state_dict(checkpoint['scheduler_at'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer, optimizer_at, scheduler, scheduler_at, checkpoint['epoch'], checkpoint['mean_iu']

def restore_snapshot(net, optimizer, scheduler, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint and restore_optimizer_bool:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer, scheduler, checkpoint['epoch'], checkpoint['mean_iu']


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def forgiving_state_copy(target_net, source_net):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = target_net.state_dict()
    loaded_dict = source_net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
            print("Matched", k)
        else:
            print("Skipped loading parameter ", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    target_net.load_state_dict(net_state_dict)
    return target_net
