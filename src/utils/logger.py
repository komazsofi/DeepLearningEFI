# src/utils/logger.py
import logging
import datetime
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
import os

def setup_experiment_dir(args, task):
    """Creates directory structure and returns paths."""
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/').joinpath(task).joinpath(args.log_dir or timestr)
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    
    return checkpoints_dir, log_dir

def setup_logger(log_dir, model_name, task, args):
    """Initializes logger and returns the logger object and log_string helper."""
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(f'{log_dir}/{model_name}_{task}.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def log_string(text):
        logger.info(text)

    log_string('PARAMETER ...')
    log_string(args)
    
    return logger, log_string

def plot_training_curves(history, out_dir, task):
    """
    history: dict with keys:
      'train_loss', 'val_loss', 'train_acc', 'val_acc' (each is a list per epoch)
    Saves two PNGs to out_dir.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss figure
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_vs_epoch.png'), dpi=200)
    plt.close()

    # Accuracy figure
    plt.figure()
    if task == 'classification':
        plt.plot(epochs,  history['train_acc'], label='Train Acc')
        plt.plot(epochs, history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Epoch')
    else:
        plt.plot(epochs, history['train_r2'] , label='Train R2')
        plt.plot(epochs, history['val_r2'], label='Val R2')
        plt.xlabel('Epoch')
        plt.ylabel('R2')
        plt.title('R2 vs. Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'r2_vs_epoch.png'), dpi=200)
    plt.close()