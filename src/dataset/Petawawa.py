import os
import csv
import numpy as np
import warnings
import pickle
import torch
from tqdm import tqdm
import open3d as o3d
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def farthest_point_sample(points, npoint):
    """
    FPS on first 3 dims (XYZ). Returns sampled points [npoint, D].
    """
    N, D = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int32)
    distances = np.ones((N,), dtype=np.float32) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    return points[centroids]

def estimate_normals(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
    normals = np.asarray(pcd.normals)
    return normals
    
def _sample_n(points, npoint, use_fps=False):
    """Sample to npoint with FPS or simple/with-replacement fallback."""
    N = points.shape[0]
    if use_fps and N >= npoint:
        return farthest_point_sample(points, npoint)
    if N >= npoint:
        return points[:npoint, :]
    # If too few points, pad by sampling with replacement
    idx = np.random.choice(N, size=npoint, replace=True)
    return points[idx, :]


class PetawawaPointCloudDataset(Dataset):
    """
    Dataloader for Petawawa dataset:
      - root: directory containing <plot_id>.npy point clouds
      - csv_path: CSV with columns: plot_id, dom_sp_type, split (train/val/test)
      - args must define: num_point, use_uniform_sample (bool)
    """

    def __init__(self, root, csv_path, args, split='train', process_data=False, cache_tag=None):
        self.root = root
        self.csv_path = csv_path
        self.npoints = int(args.num_point)
        self.uniform = bool(args.use_uniform_sample)
        self.split = split.lower()
        self.use_normals = bool(getattr(args, 'use_normals', False))
        assert self.split in ('train', 'val', 'test'), "split must be one of ['train','val','test']"
        self.process_data = process_data

        # 1) Read CSV and collect all labels to make a stable mapping across splits
        rows_all = []
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter=',')
            # tolerate possible tabs; DictReader handles it as content, not delimiter
            for row in reader:
                # Normalize field names just in case
                pid = row.get('plot_id') or row.get('plot') or row.get('id') or ''
                label = row.get('dom_sp_type') or row.get('label') or ''
                spl = (row.get('split') or '').strip().lower()
                if not pid or not label or not spl:
                    continue
                rows_all.append({'plot_id': pid.strip(), 'dom_sp_type': label.strip(), 'split': spl})

        # Build label space from ALL rows for consistent indices
        label_set = sorted(list({r['dom_sp_type'] for r in rows_all}))
        self.classes = {lab: i for i, lab in enumerate(label_set)}
        self.class_names = label_set
        self.num_category = len(self.class_names)

        # 2) Filter by split and build datapath list
        rows_split = [r for r in rows_all if r['split'] == self.split]
        self.datapath = []
        for r in rows_split:
            plot_id = r['plot_id']
            label = r['dom_sp_type']
            pc_fp = os.path.join(self.root, f"{plot_id}.npy")
            if not os.path.isfile(pc_fp):
                # Skip missing files but warn
                warnings.warn(f"Point cloud not found: {pc_fp} (skipped)")
                continue
            self.datapath.append((plot_id, pc_fp, self.classes[label]))

        print(f"The size of {self.split} data is {len(self.datapath)} | #classes={self.num_category}: {self.class_names}")

        # 3) Where to cache processed data
        cache_name = f"petawawa_{self.split}_{self.npoints}pts"
        if self.uniform:
            cache_name += "_fps"
        if self.use_normals:
            cache_name += "_normals"
        self.save_path = os.path.join(self.root, cache_name + ".pkl")

        if self.process_data:
            if not os.path.exists(self.save_path):
                print(f"Processing data {self.save_path} (first run only)...")
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    _, pc_fp, cls_idx = self.datapath[index]
                    point_set = self._load_point(pc_fp)  # [N, D]
                    point_set = _sample_n(point_set, self.npoints, use_fps=self.uniform)
                    
                    if self.use_normals:
                        xyz = point_set[:, :3]
                        normals = estimate_normals(xyz)
                        point_set = np.concatenate([point_set, normals], axis=1)

                    self.list_of_points[index] = point_set.astype(np.float32)
                    self.list_of_labels[index] = np.array([cls_idx], dtype=np.int32)

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels, self.class_names], f)
            else:
                print(f"Load processed data from {self.save_path}...")
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels, self.class_names = pickle.load(f)

    def _load_point(self, filepath):
        arr = np.load(filepath)
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Point cloud at {filepath} must be 2D [N, D], got shape {arr.shape}")
        if arr.shape[1] < 3:
            raise ValueError(f"Point cloud at {filepath} must have >=3 columns (XYZ). Got {arr.shape[1]}")
        return arr.astype(np.float32)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data and hasattr(self, 'list_of_points'):
            point_set = self.list_of_points[index]
            label = self.list_of_labels[index]
        else:
            _, pc_fp, cls_idx = self.datapath[index]
            label = np.array([cls_idx], dtype=np.int32)

            point_set = self._load_point(pc_fp)  # [N, D]
            point_set = _sample_n(point_set, self.npoints, use_fps=self.uniform)
            # compute normals on-the-fly if no cache
            if self.use_normals:
                xyz = point_set[:, :3]
                normals = estimate_normals(xyz)
                point_set = np.concatenate([point_set, normals], axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)

    def get_class_mapping(self):
        """Returns dict: {label_str: class_index}"""
        return dict(self.classes)


if __name__ == '__main__':
    from types import SimpleNamespace

    args = SimpleNamespace(
        num_point=8192,
        use_uniform_sample=True,  # FPS
    )

    dataset = PetawawaPointCloudDataset(
        root='./data/plot_point_clouds',          # contains PRF014.npy, PRF100.npy, ...
        csv_path='./data/labels.csv',       # with columns: plot_id, dom_sp_type, split
        args=args,
        split='train',                         # 'train' | 'val' | 'test'
        process_data=True                     # cache processed arrays for speed
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True, num_workers=4, drop_last=False)
    for pts, lab in loader:
        # pts: [B, npoints, 3 or D]; lab: [B]
        print(pts.shape, lab.shape)
        break
