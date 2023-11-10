import os
import json
import time
import pickle
from statistics import mean
from collections import OrderedDict

import wandb
import open3d as o3d
import torch
import torch.nn as nn

from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


# Vaagn add THIS
UNSEEN = False
CAR_SAVE = True
ADAPOINTR = True
MY_DATA = False
HALF = '' # если нет то просто ""

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    config.dataset.train.bs = 128
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    print('config.model', config.model)
    print('args.local_rank', args.local_rank)
    
#     if args.use_pretrained_mt:
#         print('use pretrained Modality Transfer')
#         ckpt_path = '/home/jovyan/vchopuryan/PoinTr/pretrained/modality_transfer_last.pth'
#         state_dict = torch.load(ckpt_path)
#         base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model_state_dict'].items()}
#         base_model.base_img_model.load_state_dict(base_ckpt)
#         del base_ckpt
        
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)
    
    dataset_name = config.dataset.train._base_.NAME
    if 'SegImgPCN' in dataset_name:
        print('Start load Segmentator weights')
        state_dict = torch.load(args.gdanet_w,
                                map_location=torch.device('cpu'))['model']

        new_state_dict = OrderedDict()
        for layer in state_dict:
            new_state_dict[layer.replace('module.', '')] = state_dict[layer]
        base_model.base_model.segmentator.load_state_dict(new_state_dict)
        for parameter in base_model.base_model.segmentator.parameters():
            parameter.requires_grad = False
        print('Finish load Segmentator weights')
    
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
#         print('[args.local_rank % torch.cuda.device_count()]', [args.local_rank % torch.cuda.device_count()])
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True
                                                        )
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
        print('resume_optimize', optimizer)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)
    
    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])
        
        if args.wandb:
            wandb.init(project="point-cloud-completion",
                       entity="3d-team",
                      )
            wandb.run.name = config.model.NAME

        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        if args.wandb:
            train_loss_cd_fine = []
            train_loss_cd_coarse = []
            train_loss_all = []
            
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            
            if 'ImgProjPCN' in dataset_name:
                print('RotImgPCN')
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                rot_tensor = data[3].cuda()
                cam_dist = data[4].cuda()
                
            elif 'SegImgPCN' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                cls_vec = data[3].cuda()
            
            elif 'ImgPCN' in dataset_name or 'ViPC' in dataset_name:
#                 print('ImgPCN ' * 10)
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    # partial, gt = misc.random_scale(partial, gt) # specially for KITTI finetune
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif  'PCN' in dataset_name or dataset_name == 'Completion3D' or 'ProjectShape' or 'ViPC' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    # partial, gt = misc.random_scale(partial, gt) # specially for KITTI finetune
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune
                    
            elif 'ShapeNet' in dataset_name:
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt,
                                                      npoints,
                                                      [int(npoints * 1/4) , int(npoints * 3/4)], 
                                                      fixed_points=None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
            if 'ImgProjPCN' in dataset_name:
                ret = base_model(partial, img, rot_tensor, cam_dist)
            elif 'SegImgPCN' in dataset_name:
                ret = base_model(partial, img, cls_vec)
            elif 'ImgPCN' in dataset_name or 'ViPC' in dataset_name:
                ret = base_model(partial, img)
            else:
                ret = base_model(partial)
                
            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch)
            _loss = sparse_loss + dense_loss 
            _loss.backward()
            
            
            if args.wandb:
                train_loss_cd_fine.append(dense_loss.item())
                train_loss_cd_coarse.append(sparse_loss.item())
                train_loss_all.append(_loss.item())

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in base_model.parameters() if p.requires_grad],
#                     base_model.parameters(),
                    getattr(config, 'grad_norm_clip', 10),
                    norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
        
        if args.wandb:
                wandb.log({
                    "train/coarse_loss_cd": mean(train_loss_cd_coarse) * 1000,
                    "train/fine_loss_cd": mean(train_loss_cd_fine) * 1000,
                    "train/all_loss_cd": mean(train_loss_all) * 1000
                    }, step=epoch)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics, test_L1, test_L2 = validate(base_model,
                                                 test_dataloader,
                                                 epoch,
                                                 ChamferDisL1,
                                                 ChamferDisL2,
                                                 val_writer,
                                                 args,
                                                 config,
                                                 logger=logger)
            if args.wandb:
                wandb.log({
                        "valid/CD_L1": mean(test_L1),
                        "valid/CD_L2": mean(test_L2)
                    }, step=epoch)
            
            
            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, 
                                        optimizer, 
                                        epoch, 
                                        metrics, 
                                        best_metrics, 
                                        'ckpt-best', 
                                        args, 
                                        logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model,
                                    optimizer, 
                                    epoch, 
                                    metrics, 
                                    best_metrics, 
                                    f'ckpt-epoch-{epoch:03d}', 
                                    args, 
                                    logger=logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 10
    loss_L1 = []
    loss_L2 = []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            
            if 'ImgProjPCN' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                rot_tensor = data[3].cuda()
                cam_dist = data[4].cuda()
                
            elif 'SegImgPCN' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                cls_vec = data[3].cuda()
                
            elif 'ImgPCNScaler' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                
            elif 'ImgPCN' in dataset_name  or 'ViPC' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                
            elif 'PCN' in dataset_name or dataset_name == 'Completion3D' or 'ProjectShape' or 'ViPC' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                
            elif 'ShapeNet' in dataset_name:
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt,
                                                      npoints, 
                                                      [int(npoints * 1/4) , int(npoints * 3/4)], 
                                                      fixed_points=None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
                
            if 'ImgProjPCN' in dataset_name:
                ret = base_model(partial, img, rot_tensor, cam_dist)
            elif 'SegImgPCN' in dataset_name:
                ret = base_model(partial, img, cls_vec)
            elif 'ImgPCNScaler' in dataset_name:
                ret = base_model(partial, img)
            elif 'ImgPCN' in dataset_name or 'ViPC' in dataset_name:
                ret = base_model(partial, img)
            else:
                ret = base_model(partial)

            coarse_points = ret[0]
            dense_points = ret[-1]
            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)
            loss_L1.append(dense_loss_l1.item() * 1000)
            loss_L2.append(dense_loss_l2.item() * 1000)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000,
                                sparse_loss_l2.item() * 1000, 
                                dense_loss_l1.item() * 1000, 
                                dense_loss_l2.item() * 1000])
            _metrics = Metrics.get(dense_points, gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]),
                  logger=logger)
        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg()), loss_L1, loss_L2


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
#     print('ARGS_' * 25, args)
#     print('config.dataset.val_' *10, config.dataset.val)
    base_model = builder.model_builder(config.model)
    print(base_model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):
    print('TEST' * 20)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    
    nnetwork_name = config.model.NAME

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]
#             print(idx, (taxonomy_id, model_id))
            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
  
            if 'SegImgPCN' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                cls_vec = data[3].cuda()
                
                ret = base_model(partial, img, cls_vec)
                coarse_points = ret[0]
                dense_points = ret[-1]
                
                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)
                
                test_losses.update([sparse_loss_l1.item() * 1000,
                                    sparse_loss_l2.item() * 1000,
                                    dense_loss_l1.item() * 1000,
                                    dense_loss_l2.item() * 1000])
                
                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif 'ImgPCN' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                img = data[2].cuda()
                
                ret = base_model(partial, img)
                coarse_points = ret[0]
                dense_points = ret[-1]
                
                # # for save coarse and fine
                # save_coarse_path = f'/home/jovyan/vchopuryan/PoinTr/clouds/{nnetwork_name}/{taxonomy_id}/coarse'
                # os.makedirs(save_coarse_path, exist_ok=True)
                # save_coarse_path = f'/home/jovyan/vchopuryan/PoinTr/clouds/{nnetwork_name}/{taxonomy_id}/coarse/{model_id}.ply'
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(coarse_points[0].detach().cpu().numpy())
                # o3d.io.write_point_cloud(save_coarse_path, pcd)
                
                # save_fine_path = f'/home/jovyan/vchopuryan/PoinTr/clouds/{nnetwork_name}/{taxonomy_id}/fine'
                # os.makedirs(save_fine_path, exist_ok=True)
                # save_fine_path = f'/home/jovyan/vchopuryan/PoinTr/clouds/{nnetwork_name}/{taxonomy_id}/fine/{model_id}.ply'
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(dense_points[0].detach().cpu().numpy())
                # o3d.io.write_point_cloud(save_fine_path, pcd)
                # # end for save coarse
                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)
                test_losses.update([sparse_loss_l1.item() * 1000,
                                    sparse_loss_l2.item() * 1000,
                                    dense_loss_l1.item() * 1000,
                                    dense_loss_l2.item() * 1000])
                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
                
                
                
            elif  'PCN' in dataset_name or 'ProjectShapeNet' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[-1]
            
                # # for save coarse and fine
                # save_coarse_path = f'/home/jovyan/vchopuryan/PoinTr/clouds/{nnetwork_name}/{taxonomy_id}/coarse'
                # os.makedirs(save_coarse_path, exist_ok=True)
                # save_coarse_path = f'/home/jovyan/vchopuryan/PoinTr/clouds/{nnetwork_name}/{taxonomy_id}/coarse/{model_id}.ply'
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(coarse_points[0].detach().cpu().numpy())
                # o3d.io.write_point_cloud(save_coarse_path, pcd)
                
                # save_fine_path = f'/home/jovyan/vchopuryan/PoinTr/clouds/{nnetwork_name}/{taxonomy_id}/fine'
                # os.makedirs(save_fine_path, exist_ok=True)
                # save_fine_path = f'/home/jovyan/vchopuryan/PoinTr/clouds/{nnetwork_name}/{taxonomy_id}/fine/{model_id}.ply'
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(dense_points[0].detach().cpu().numpy())
                # o3d.io.write_point_cloud(save_fine_path, pcd)
                # # end for save coarse

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)
                test_losses.update([sparse_loss_l1.item() * 1000,
                                    sparse_loss_l2.item() * 1000, 
                                    dense_loss_l1.item() * 1000, 
                                    dense_loss_l2.item() * 1000])
                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
                    
            elif 'ViPC' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[-1]

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, 
                                    sparse_loss_l2.item() * 1000, 
                                    dense_loss_l1.item() * 1000, 
                                    dense_loss_l2.item() * 1000])
                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
                        
            elif 'ShapeNet' in dataset_name:
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for choice_index, item in enumerate(choice):           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[-1]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000,
                                        sparse_loss_l2.item() * 1000, 
                                        dense_loss_l1.item() * 1000, 
                                        dense_loss_l2.item() * 1000])
                    _metrics = Metrics.get(dense_points ,gt)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
                            
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)
    
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.5f \t' % value
    print_log(msg, logger=logger)
    return 
