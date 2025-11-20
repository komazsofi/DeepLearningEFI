import os
import sys
import torch
import torch.optim as optim
import numpy as np
from collections import defaultdict
import importlib
import argparse
from tqdm import tqdm

# --- Project Imports ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models')) 
sys.path.append(os.path.join(ROOT_DIR, 'src')) 
sys.path.append(os.path.join(ROOT_DIR, 'utils')) 

# Configuration and Utilities
from config import DATASET_CONFIG, MODEL_CONFIG 
from dataset.generic_dataset import ForestryDataset
from models.loss_utils import calculate_total_loss 
from utils.metrics import MetricsCalculator
from utils.logger import setup_experiment_dir, setup_logger, plot_training_curves
import dataset.data_utils as data_utils

# --- Helper Functions from Original Script (kept here as they are tightly coupled with training loop) ---

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--model', type=str, default='pointnet', help='Model file name from MODEL_CONFIG keys')
    parser.add_argument('--dataset', type=str, default='petawawa_cls', 
                        choices=DATASET_CONFIG.keys(), help='Key from DATASET_CONFIG to use')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    """Helper to modify ReLU layers."""
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader, config, device, logger):
    """
    Evaluates the model and computes task-specific metrics using MetricsCalculator.
    """
    classifier = model.eval()
    metric_calc = MetricsCalculator(config['task'], config['num_classes'])

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader), desc="Validation"):
        points, target = points.to(device), target.to(device)

        # Transpose points to [B, C, N] format for models
        points = points.transpose(2, 1)
        points = points.float()

        # Forward pass
        pred, trans_feat = classifier(points) 
        
        # Calculate loss using the centralized utility
        loss = calculate_total_loss(
            pred, 
            target, 
            trans_feat, 
            task=config['task'],
            mat_diff_loss_scale=config.get('mat_diff_loss_scale', 0.0)
        )
        
        metric_calc.update(pred, target, loss.item())

    return metric_calc.compute_and_log(logger)


# --- Main Function ---

def main(args):
    
    # 1. Configuration Setup
    dataset_cfg = DATASET_CONFIG[args.dataset]
    model_cfg = MODEL_CONFIG[args.model]
    
    full_cfg = {
        'num_classes': dataset_cfg['num_classes'],
        'task': dataset_cfg['task'],
        'use_cpu': args.use_cpu, 
        'num_point': args.num_point,
        **model_cfg
    }
    
    # 2. Directory and Logging Setup
    checkpoints_dir, log_dir = setup_experiment_dir(args, full_cfg['task'])
    logger, log_string = setup_logger(log_dir, args.model, full_cfg['task'], args)
    
    # 3. Data Loading
    log_string('Load dataset ...')
    
    train_dataset = ForestryDataset(
        root=dataset_cfg['root'], 
        csv_path=dataset_cfg['csv'], 
        label_col=dataset_cfg['label_col'], 
        task_type=full_cfg['task'],
        split='train',
        num_points=args.num_point,
        classes_list=dataset_cfg['classes'],
        process_data=args.process_data,
        use_normals=full_cfg['use_normal'],
        use_fps=args.use_uniform_sample
    )
    test_dataset = ForestryDataset(
        root=dataset_cfg['root'], 
        csv_path=dataset_cfg['csv'], 
        label_col=dataset_cfg['label_col'], 
        task_type=full_cfg['task'],
        split='val',
        num_points=args.num_point,
        classes_list=dataset_cfg['classes'],
        process_data=args.process_data,
        use_normals=full_cfg['use_normal'],
        use_fps=args.use_uniform_sample
    )
    
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    device = torch.device("cuda" if not full_cfg['use_cpu'] and torch.cuda.is_available() else "cpu")

    # 4. Model Loading and Initialization
    model_module = importlib.import_module(f"models.{args.model}")
    classifier = model_module.get_model(full_cfg) 
    classifier.apply(inplace_relu)
    classifier = classifier.to(device)

    # 5. Load Checkpoint and Optimizer
    try:
        checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth', map_location=device)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    history = defaultdict(list) 
    global_epoch = 0
    # Track the primary metric for saving the best model
    best_metric = -float('inf') 

    # 6. Training Loop
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        # --- Training Phase ---
        classifier.train()
        train_metrics = MetricsCalculator(full_cfg['task'], full_cfg['num_classes'])
        scheduler.step()
        
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9, desc="Training"):
            optimizer.zero_grad()

            # Data Augmentation (on CPU, then move to GPU)
            points = points.data.numpy()
            points = data_utils.random_point_dropout(points)
            points[:, :, 0:3] = data_utils.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            points = points.float()

            pred, trans_feat = classifier(points)
            
            loss = calculate_total_loss(
                pred, 
                target, 
                trans_feat, 
                task=full_cfg['task'],
                mat_diff_loss_scale=full_cfg.get('mat_diff_loss_scale', 0.0)
            )
            
            loss.backward()
            optimizer.step()
            
            train_metrics.update(pred, target, loss.item())

        train_results = train_metrics.compute_and_log(logger)
        primary_train_metric = train_results.get('instance_acc', train_results.get('r2'))
        log_string(f"Train Loss: {train_results['loss']:.4f} | Primary Metric ({'Acc' if full_cfg['task']=='classification' else 'R2'}): {primary_train_metric:.4f}")


        # --- Validation Phase ---
        with torch.no_grad():
            val_results = test(classifier.eval(), testDataLoader, full_cfg, device, logger)

            # Determine the primary metric for saving the best model
            current_metric = val_results.get('instance_acc', val_results.get('r2')) 

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch + 1
                
                logger.info('Save best model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string(f'Saving at {savepath} with best metric: {best_metric:.4f}')
                state = {
                    'epoch': best_epoch,
                    'primary_metric': best_metric,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            
            log_string(f"Best Metric ({'Acc' if full_cfg['task']=='classification' else 'R2'}): {best_metric:.4f}")
            
            # ===== record history =====
            history['train_loss'].append(train_results['loss'])
            history['val_loss'].append(val_results['loss'])
            
            if full_cfg['task'] == 'classification':
                 history['train_acc'].append(train_results['instance_acc'])
                 history['val_acc'].append(val_results['instance_acc'])
            elif full_cfg['task'] == 'regression':
                 history['train_r2'].append(train_results['r2'])
                 history['val_r2'].append(val_results['r2'])
                 history['val_rmse'].append(val_results['rmse'])
                 
            global_epoch += 1

    logger.info('End of training...')
    # Plotting and saving history
    try:
        plot_training_curves(history, str(log_dir), task=full_cfg['task'])
        # also dump raw arrays for later
        np.savez(os.path.join(log_dir, 'training_history.npz'), **history)
        log_string(f"Saved training curves and history to {log_dir}")
    except Exception as e:
        log_string(f"Plotting failed: {e}")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)