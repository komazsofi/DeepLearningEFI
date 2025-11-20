import os
import csv
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
from .data_utils import estimate_normals, sample_points


class ForestryDataset(Dataset):
    def __init__(self, root, csv_path, label_col, split='train', 
                 num_points=8192, task_type='classification', classes_list=None,
                 process_data=False, use_normals=False, use_fps=True):
        
        self.root = root
        self.csv_path = csv_path
        self.label_col = label_col
        self.npoints = num_points
        self.task_type = task_type
        self.use_normals = use_normals
        self.use_fps = use_fps
        self.split = split

        # 1. Parse CSV
        self.meta_data = [] # Stores (plot_id, path, raw_label_value)
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('split', '').strip().lower() == self.split:
                    # ID fetch
                    pid = row.get('plot_id') or row.get('plot') or row.get('id')
                    
                    # Label fetch
                    raw_val = row.get(self.label_col)

                    if pid and raw_val:
                        pc_path = os.path.join(self.root, f"{pid}.npy")
                        if os.path.exists(pc_path):
                            self.meta_data.append({
                                'id': pid,
                                'path': pc_path,
                                'raw_label': raw_val
                            })

        # 2. Setup Label Mapping (Task dependent)
        self.cls_to_idx = {}
        self.class_names = []
        
        if self.task_type == 'classification':
            self.class_names = classes_list
            self.cls_to_idx = {name: i for i, name in enumerate(self.class_names)}
            self.num_classes = len(self.class_names)
            print(f"[{split}] Using class list from config: {self.class_names}")
        else:
            self.num_classes = 1 # Regression output dim
            print(f"[{self.split}] Regression task on '{label_col}'")

        # 3. Caching Logic
        cache_name = f"{self.split}_{self.npoints}_{self.label_col}"
        if self.use_normals: cache_name += "_normals"
        if self.use_fps: cache_name += "_fps"
        
        self.cache_path = os.path.join(self.root, f"cache_{cache_name}.pkl")

        if process_data:
            self._process_and_cache()
        elif os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self.list_of_points = None
            print("No cache found. Loading data on-the-fly (slow).")

    # --- Processing Function ---
    def _process_and_cache(self):
        print(f"Processing data to {self.cache_path}...")
        
        processed_points_list = []
        processed_labels_list = []

        for item in tqdm(self.meta_data):
            # Load Point Cloud
            points = np.load(item['path']).astype(np.float32)
            
            # Sample
            points = sample_points(points, self.npoints, use_fps=self.use_fps)

            # Normals
            if self.use_normals:
                xyz = points[:, :3]
                normals = estimate_normals(xyz)
                points = np.concatenate([points, normals], axis=1)

            processed_points_list.append(points)

            # D. Process Label based on Task
            if self.task_type == 'classification':
                # String -> Integer Index
                label_idx = self.cls_to_idx[item['raw_label']]
                processed_labels_list.append(label_idx)
            else:
                processed_labels_list.append(float(item['raw_label']))

        # Save to disk
        data = {
            'points': processed_points_list, 
            'labels': processed_labels_list,
            'meta': {'cls_map': self.cls_to_idx}
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)
            
        # Load into memory immediately
        self.list_of_points = processed_points_list
        self.list_of_labels = processed_labels_list
    
    def _load_cache(self):
        print(f"Loading cached data from {self.cache_path}")
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)
            self.list_of_points = data['points']
            self.list_of_labels = data['labels']
            self.cls_to_idx = data['meta'].get('cls_map', {})

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        # 1. Get Points
        if self.list_of_points is not None:
            points = self.list_of_points[index]
        else:
            # On-the-fly fallback
            item = self.meta_data[index]
            points = np.load(item['path']).astype(np.float32)
            points = sample_points(points, self.npoints, use_fps=self.use_fps)
            if self.use_normals:
                normals = estimate_normals(points[:,:3])
                points = np.concatenate([points, normals], axis=1)

        # 2. Get Labels
        if self.list_of_points is not None:
            label = self.list_of_labels[index]
        else:
            # On-the-fly label parsing
            raw = self.meta_data[index]['raw_label']
            if self.task_type == 'classification':
                label = self.cls_to_idx[raw]
            else:
                label = float(raw)

        # 3. Return Correct Types
        if self.task_type == 'classification':
            return points, torch.tensor(label, dtype=torch.long)
        else:
            # For regression, return float32 (usually shape [1] or scalar)
            return points, torch.tensor(label, dtype=torch.float32)