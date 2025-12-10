import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .efi_dataset import EfiDataset

class EfiDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the EFI point cloud dataset.
    Handles data preparation, setup, and creation of DataLoaders.
    """
    def __init__(self, cfg, args):
        super().__init__()
        self.cfg = cfg # Full dataset config (e.g., root, csv_path)
        self.args = args # Command line arguments (e.g., batch_size, num_point)
        self.num_workers = 4 # Fixed worker count for robustness

    def prepare_data(self):
        """
        Logic for dataset preparation (only run on 1 GPU/process).
        If 'process_data' is true, this triggers the pre-processing step.
        """
        if getattr(self.args, "process_data", False):
            print("--- Running Data Pre-processing (train) ---")
            EfiDataset(
                root=self.cfg['root'], 
                csv_path=self.cfg['csv'], 
                label_col=self.cfg['label_col'], 
                task_type=self.cfg['task'],
                split='train',
                num_points=self.args.num_point,
                classes_list=self.cfg['classes'],
                process_data=True, # Triggers cache generation
                use_normals=self.cfg['use_normal'],
                use_fps=self.cfg.get('use_fps', False)
            )
            print("--- Running Data Pre-processing (val) ---")
            EfiDataset(
                root=self.cfg['root'], 
                csv_path=self.cfg['csv'], 
                label_col=self.cfg['label_col'], 
                task_type=self.cfg['task'],
                split='val',
                num_points=self.args.num_point,
                classes_list=self.cfg['classes'],
                process_data=True, # Triggers cache generation
                use_normals=self.cfg['use_normal'],
                use_fps=self.cfg.get('use_fps', True)
            )
            print("--- Running Data Pre-processing (test) ---")
            EfiDataset(
                root=self.cfg['root'], 
                csv_path=self.cfg['csv'], 
                label_col=self.cfg['label_col'], 
                task_type=self.cfg['task'],
                split='test',
                num_points=self.args.num_point,
                classes_list=self.cfg['classes'],
                process_data=True, # Triggers cache generation
                use_normals=self.cfg['use_normal'],
                use_fps=self.cfg.get('use_fps', True)
            )

    def setup(self, stage=None):
        """
        Load datasets (run on every GPU/process).
        """
        common_params = dict(
            root=self.cfg['root'], 
            csv_path=self.cfg['csv'], 
            label_col=self.cfg['label_col'], 
            task_type=self.cfg['task'],
            num_points=self.args.num_point,
            classes_list=self.cfg['classes'],
            process_data=False, # We rely on cache or on-the-fly loading
            use_normals=self.cfg['use_normal'],
            use_fps=self.cfg.get('use_fps', True)
        )

        if stage == 'fit' or stage is None:
            self.train_dataset = EfiDataset(split='train', **common_params)
            self.val_dataset = EfiDataset(split='val', **common_params)
        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = EfiDataset(split='test', **common_params)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
