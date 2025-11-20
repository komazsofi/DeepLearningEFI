"""
Author: Benny / Revised for Multi-Task & Modularity by AI
Date: Nov 2019 / Nov 2025
"""

import os
import sys
import torch
import importlib
import argparse
from tqdm import tqdm
from pathlib import Path

# --- Project Imports ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# Adjust sys.path for internal modules (assuming models/ and src/ are one level up)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'src')) 

from config import DATASET_CONFIG, MODEL_CONFIG
from dataset.generic_dataset import ForestryDataset
from utils.metrics import MetricsCalculator
from utils.logger import setup_logger # We'll reuse the logger setup for simplicity

# --- Helper Functions ---

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--dataset_config_key', type=str, default='petawawa_cls', 
                        choices=DATASET_CONFIG.keys(), help='Key from DATASET_CONFIG to use')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root directory (e.g., the timestamp folder)')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate scores with voting (only used for classification)')
    parser.add_argument('--model', type=str, default=None, help='Override model name if not inferrable from log_dir')
    return parser.parse_args()


def test(model, loader, full_cfg, device, logger):
    """
    Evaluates the model with optional voting for classification.
    """
    classifier = model.eval()
    metric_calc = MetricsCalculator(full_cfg['task'], full_cfg['num_classes'])
    
    vote_num = full_cfg['num_votes']
    task = full_cfg['task']
    num_outputs = full_cfg['num_classes']

    for points, target in tqdm(loader, total=len(loader), desc="Testing"):
        points, target = points.to(device), target.to(device)
        points = points.transpose(2, 1) # [B, N, C] -> [B, C, N]
        points = points.float() # Ensure float32

        if task == 'classification':
            # Classification with Voting
            vote_pool = torch.zeros(target.size()[0], num_outputs).to(device)
            for _ in range(vote_num):
                pred, _ = classifier(points)
                vote_pool += pred.data
            final_pred = vote_pool / vote_num
            
            # Use a dummy loss item for metric calculation, as we only care about performance
            loss_item = 0.0

        elif task == 'regression':
            # Regression (No Voting)
            final_pred, _ = classifier(points)
            loss_item = 0.0 # Loss is not required for evaluation metrics here

        metric_calc.update(final_pred, target, loss_item)

    # Compute and log final results
    results = metric_calc.compute_and_log(logger)
    
    # Return the primary metric for logging summary
    if task == 'classification':
        return results['instance_acc'], results['class_acc']
    else: # regression
        return results['r2'], results['rmse']


def main(args):
    # 1. Configuration and Directory Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # We must infer the model name and task from the log directory structure
    experiment_dir = Path('./log').joinpath(args.log_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Infer Task and Model Name from log path if not explicitly provided
    task = experiment_dir.parent.name
    
    # Load model name from checkpoint if not provided (more robust)
    try:
        log_file = next(iter(experiment_dir.joinpath('logs').glob('*.txt')))
        model_name = log_file.stem.split('_')[0]
    except Exception:
        if args.model is None:
             raise ValueError("Could not infer model name from log directory. Please specify --model argument.")
        model_name = args.model
    
    # Setup Logger (using a simplified version from train.py utils)
    logger, log_string = setup_logger(
        experiment_dir.joinpath('logs'), 
        model_name, 
        'eval', 
        args
    )

    # Load Configs
    dataset_cfg = DATASET_CONFIG[args.dataset_config_key]
    model_cfg = MODEL_CONFIG[model_name]
    
    # Finalize Configuration Dictionary (must match training parameters)
    full_cfg = {
        'num_classes': dataset_cfg['num_classes'],
        'use_normal': dataset_cfg['use_normal'],
        'task': task, # Ensure task is correctly inferred
        'num_point': args.num_point,
        'num_votes': args.num_votes,
        **model_cfg
    }
    
    device = torch.device("cuda" if not args.use_cpu and torch.cuda.is_available() else "cpu")
    
    # 2. Data Loading
    log_string('Load dataset ...')
    
    test_dataset = ForestryDataset(
        root=dataset_cfg['root'], 
        csv_path=dataset_cfg['csv'], 
        label_col=dataset_cfg['label_col'], 
        task_type=task,
        split='test',
        num_points=args.num_point,
        classes_list=dataset_cfg['classes'],
        process_data=False, # Assume pre-processed data exists from training
        use_normals=full_cfg['use_normal'],
        use_uniform_sample=args.use_uniform_sample
    )

    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 3. Model Loading
    log_string('MODEL LOADING...')
    model_module = importlib.import_module(f"models.{model_name}")

    # Load model using the full config
    classifier = model_module.get_model(full_cfg) 
    classifier = classifier.to(device)

    checkpoint_path = experiment_dir.joinpath('checkpoints/best_model.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string(f'Loaded model state from epoch {checkpoint["epoch"]}.')


    # 4. Testing
    log_string('\n--- Start Testing ---')
    with torch.no_grad():
        results = test(classifier.eval(), testDataLoader, full_cfg, device, logger)
        
        if task == 'classification':
            instance_acc, class_acc = results
            log_string(f'Test Instance Accuracy: {instance_acc:.4f}, Class Accuracy: {class_acc:.4f}')
        else:
            r2, rmse = results
            log_string(f'Test R2 Score: {r2:.4f}, RMSE: {rmse:.4f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)