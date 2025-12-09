import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import datetime

# --- Project Imports ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# Add necessary paths
sys.path.append(os.path.join(ROOT_DIR, 'src')) 

from config import DATASET_CONFIG, MODEL_CONFIG
from dataset.efi_datamodule import EfiDataModule
from models.efi_modelmodule import EfiModelModule
import dataset.data_utils as data_utils # For augmentations (if any)

# --- Argument Parsing (Updated) ---

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device (e.g., 0, 1, 0,1)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--model', type=str, default='pointnet', help='Model file name from MODEL_CONFIG keys')
    parser.add_argument('--dataset_config_key', type=str, default='petawawa_cls', 
                        choices=DATASET_CONFIG.keys(), help='Key from DATASET_CONFIG to use')
    parser.add_argument('--epoch', default=150, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='log', help='W&B run name/experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    
    # W&B specific arguments
    parser.add_argument('--wandb_project', type=str, default='EFI_DL', help='Weights & Biases Project Name')
    
    return parser.parse_args()


# --- Main Function ---

def main(args):
    
    # 1. Configuration Setup
    dataset_cfg = DATASET_CONFIG[args.dataset_config_key]
    model_cfg = MODEL_CONFIG[args.model]
    
    task = 'classification' if dataset_cfg['num_classes'] > 1 else 'regression'
    
    # Combined configuration dictionary passed to the Lightning Module
    full_cfg = {
        'model_name': args.model,
        'num_classes': dataset_cfg['num_classes'],
        'task': task,
        **dataset_cfg, # Include all dataset params (root, csv, etc.)
        **model_cfg    # Include all model params (mat_diff_loss_scale, etc.)
    }
    
    # 2. Setup W&B Logger
    wandb_logger = WandbLogger(
        project="forest_species_mapping",
        group=task,
        job_type=args.model,
        name=(args.log_dir or str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))),
        config={**full_cfg, **vars(args)}, # Log all configs and arguments
    )
    
    # Use W&B's run directory for checkpoints
    checkpoint_dir = Path('./log') / args.wandb_project / wandb_logger.experiment.name / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 3. Instantiate DataModule and Lightning Module
    data_module = EfiDataModule(full_cfg, args)
    model_module = EfiModelModule(full_cfg, args)

    # 4. Define Callbacks
    primary_metric = 'val/instance_acc' if task == 'classification' else 'val/r2_score'
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, 
        filename='best_model',
        monitor=primary_metric,
        mode='max', # Maximize accuracy/R2
        save_top_k=1,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',  # The name of the logged metric to monitor
        min_delta=0.00,      # Minimum change to qualify as an improvement
        patience=10,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,       # Whether to print messages when early stopping conditions are met
        mode='min'           # 'min' for metrics where lower is better (e.g., loss), 'max' for metrics where higher is better (e.g., accuracy)
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 5. Instantiate the Trainer
    # PL handles device setup automatically based on arguments
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        accelerator='gpu' if torch.cuda.is_available() and not args.use_cpu else 'cpu',
        devices=[int(g) for g in args.gpu.split(',')] if ',' in args.gpu else (1 if not args.use_cpu else 0),
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        enable_progress_bar=True
    )

    # 6. Start Training
    print('Starting PyTorch Lightning Training...')
    trainer.fit(model_module, data_module)
    
    # Test the best checkpoint after training
    print('\n--- Testing Best Model ---')
    trainer.test(datamodule=data_module, ckpt_path=os.path.join(checkpoint_dir, 'best_model.ckpt'))

if __name__ == '__main__':
    args = parse_args()
    main(args)