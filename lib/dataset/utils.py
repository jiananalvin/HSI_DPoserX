import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from lib.utils.transforms import axis_angle_to_rot6d, rot6d_to_axis_angle


class Posenormalizer:
    def __init__(self, data_path, device='cuda:0', normalize=True, min_max=True, rot_rep=None):
        assert rot_rep in ['rot6d', 'axis']
        self.normalize = normalize
        self.min_max = min_max
        self.rot_rep = rot_rep
        if self.min_max:
            normalize_params = torch.load(os.path.join(data_path, '{}_normalize1.pt'.format(rot_rep)))
            self.min_poses, self.max_poses = normalize_params['min_poses'].to(device), normalize_params['max_poses'].to(
                device)
            # Load global_orient statistics if available (for backward compatibility, use None if not present)
            if 'min_global_orient' in normalize_params:
                self.min_global_orient = normalize_params['min_global_orient'].to(device)
                self.max_global_orient = normalize_params['max_global_orient'].to(device)
            else:
                self.min_global_orient = None
                self.max_global_orient = None
        else:
            normalize_params = torch.load(os.path.join(data_path, '{}_normalize2.pt'.format(rot_rep)))
            self.mean_poses, self.std_poses = normalize_params['mean_poses'].to(device), normalize_params[
                'std_poses'].to(device)
            # Load global_orient statistics if available (for backward compatibility, use None if not present)
            if 'mean_global_orient' in normalize_params:
                self.mean_global_orient = normalize_params['mean_global_orient'].to(device)
                self.std_global_orient = normalize_params['std_global_orient'].to(device)
            else:
                self.mean_global_orient = None
                self.std_global_orient = None

    def offline_normalize(self, poses, from_axis=False):
        assert len(poses.shape) == 2 or len(poses.shape) == 3  # [b, data_dim] or [t, b, data_dim]
        pose_shape = poses.shape
        if from_axis and self.rot_rep == 'rot6d':
            poses = axis_angle_to_rot6d(poses.reshape(-1, 3)).reshape(*pose_shape[:-1], -1)

        if not self.normalize:
            return poses

        if self.min_max:
            min_poses = self.min_poses.view(1, -1)
            max_poses = self.max_poses.view(1, -1)

            if len(poses.shape) == 3:  # [t, b, data_dim]
                min_poses = min_poses.unsqueeze(0)
                max_poses = max_poses.unsqueeze(0)

            normalized_poses = 2 * (poses - min_poses) / (max_poses - min_poses) - 1

        else:
            mean_poses = self.mean_poses.view(1, -1)
            std_poses = self.std_poses.view(1, -1)

            if len(poses.shape) == 3:  # [t, b, data_dim]
                mean_poses = mean_poses.unsqueeze(0)
                std_poses = std_poses.unsqueeze(0)

            mean_poses = mean_poses.to(poses.device)
            std_poses = std_poses.to(poses.device)
            normalized_poses = (poses - mean_poses) / std_poses

        return normalized_poses

    def offline_denormalize(self, poses, to_axis=False):
        assert len(poses.shape) == 2 or len(poses.shape) == 3  # [b, data_dim] or [t, b, data_dim]

        if not self.normalize:
            denormalized_poses = poses
        else:
            if self.min_max:
                min_poses = self.min_poses.view(1, -1)
                max_poses = self.max_poses.view(1, -1)

                if len(poses.shape) == 3:  # [t, b, data_dim]
                    min_poses = min_poses.unsqueeze(0)
                    max_poses = max_poses.unsqueeze(0)

                min_poses = min_poses.to(poses.device)
                max_poses = max_poses.to(poses.device)
                denormalized_poses = 0.5 * ((poses + 1) * (max_poses - min_poses) + 2 * min_poses)

            else:
                mean_poses = self.mean_poses.view(1, -1)
                std_poses = self.std_poses.view(1, -1)

                if len(poses.shape) == 3:  # [t, b, data_dim]
                    mean_poses = mean_poses.unsqueeze(0)
                    std_poses = std_poses.unsqueeze(0)

                mean_poses = mean_poses.to(poses.device)
                std_poses = std_poses.to(poses.device)
                denormalized_poses = poses * std_poses + mean_poses

        if to_axis and self.rot_rep == 'rot6d':
            pose_shape = denormalized_poses.shape
            denormalized_poses = rot6d_to_axis_angle(denormalized_poses.reshape(-1, 6)).reshape(*pose_shape[:-1], -1)

        return denormalized_poses

    def offline_normalize_global_orient(self, global_orient):
        """Normalize global orientation (3D) using pre-computed statistics."""
        if not self.normalize:
            return global_orient
        
        assert global_orient.shape[-1] == 3, f"Expected 3D global_orient, got shape {global_orient.shape}"
        
        if self.min_max:
            if self.min_global_orient is None:
                raise ValueError("Global orientation statistics not found. Please recompute normalization params with global_orient=True")
            min_go = self.min_global_orient.view(1, -1)
            max_go = self.max_global_orient.view(1, -1)
            
            if len(global_orient.shape) == 3:  # [t, b, 3]
                min_go = min_go.unsqueeze(0)
                max_go = max_go.unsqueeze(0)
            
            min_go = min_go.to(global_orient.device)
            max_go = max_go.to(global_orient.device)
            normalized = 2 * (global_orient - min_go) / (max_go - min_go + 1e-8) - 1
        else:
            if self.mean_global_orient is None:
                raise ValueError("Global orientation statistics not found. Please recompute normalization params with global_orient=True")
            mean_go = self.mean_global_orient.view(1, -1)
            std_go = self.std_global_orient.view(1, -1)
            
            if len(global_orient.shape) == 3:  # [t, b, 3]
                mean_go = mean_go.unsqueeze(0)
                std_go = std_go.unsqueeze(0)
            
            mean_go = mean_go.to(global_orient.device)
            std_go = std_go.to(global_orient.device)
            normalized = (global_orient - mean_go) / std_go
        
        return normalized

    def offline_denormalize_global_orient(self, global_orient):
        """Denormalize global orientation (3D) using pre-computed statistics."""
        if not self.normalize:
            return global_orient
        
        assert global_orient.shape[-1] == 3, f"Expected 3D global_orient, got shape {global_orient.shape}"
        
        if self.min_max:
            if self.min_global_orient is None:
                raise ValueError("Global orientation statistics not found. Please recompute normalization params with global_orient=True")
            min_go = self.min_global_orient.view(1, -1)
            max_go = self.max_global_orient.view(1, -1)
            
            if len(global_orient.shape) == 3:  # [t, b, 3]
                min_go = min_go.unsqueeze(0)
                max_go = max_go.unsqueeze(0)
            
            min_go = min_go.to(global_orient.device)
            max_go = max_go.to(global_orient.device)
            denormalized = 0.5 * ((global_orient + 1) * (max_go - min_go) + 2 * min_go)
        else:
            if self.mean_global_orient is None:
                raise ValueError("Global orientation statistics not found. Please recompute normalization params with global_orient=True")
            mean_go = self.mean_global_orient.view(1, -1)
            std_go = self.std_global_orient.view(1, -1)
            
            if len(global_orient.shape) == 3:  # [t, b, 3]
                mean_go = mean_go.unsqueeze(0)
                std_go = std_go.unsqueeze(0)
            
            mean_go = mean_go.to(global_orient.device)
            std_go = std_go.to(global_orient.device)
            denormalized = global_orient * std_go + mean_go
        
        return denormalized


class CombinedNormalizer:
    def __init__(self, data_path_dict, device='cuda:0', normalize=True, min_max=True, rot_rep=None, model='whole-body'):
        assert rot_rep in ['rot6d', 'axis']
        self.normalize = normalize
        self.min_max = min_max
        self.rot_rep = rot_rep
        self.POSE_DIM = 3 if rot_rep == 'axis' else 6
        self.device = device

        # TODO: add global orient for body pose in the future
        self.supported_keys = ['body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'expression', 'betas']
        assert all(
            [k in self.supported_keys for k in data_path_dict.keys()]), f'unexpected keys {data_path_dict.keys()}'
        self.normalizers = {k: Posenormalizer(v, device, normalize, min_max, rot_rep) for k, v in data_path_dict.items()}
        print('created combined normalizer supporting keys:', data_path_dict.keys())

        # Define model configurations
        self.model_configurations = {
            'face': [('jaw_pose', 1 * self.POSE_DIM), ('expression', 100)],
            'full-face': [('jaw_pose', 1 * self.POSE_DIM), ('expression', 100), ('betas', 100)],
            'two-hand': [('left_hand_pose', 15 * self.POSE_DIM), ('right_hand_pose', 15 * self.POSE_DIM)],
            'whole-body': [('body_pose', 21 * self.POSE_DIM), ('left_hand_pose', 15 * self.POSE_DIM),
                           ('right_hand_pose', 15 * self.POSE_DIM), ('jaw_pose', 1 * self.POSE_DIM), ('expression', 100)]
        }
        assert model in self.model_configurations.keys(), f'unsupported model {model}'

        self.model = model
        self.used_keys = [key for key, _ in self.model_configurations[model]]

    def flip_left_hand_pose(self, poses):
        assert poses.shape[-1] == 45, f'expected axis-angle rep 45(15x3), got {poses.shape[-1]}'
        assert len(poses.shape) == 2 or len(poses.shape) == 3  # [b, data_dim] or [t, b, data_dim]
        fliped_poses = poses.clone()
        fliped_poses[..., 1::3] *= -1
        fliped_poses[..., 2::3] *= -1

        return fliped_poses

    def _slice_and_process(self, poses, process_function, from_to_axis=False):
        last_dim = poses.dim() - 1
        start_idx = 0
        processed_parts = []
        for part, size in self.model_configurations[self.model]:
            part_data = poses.narrow(last_dim, start_idx, size)  # slicing along the last dimension
            processed_data = process_function(part_data, from_to_axis, part)
            processed_parts.append(processed_data)
            start_idx += size
        return torch.cat(processed_parts, dim=-1)

    def offline_normalize(self, poses, from_axis=False, data_key=None):
        if isinstance(poses, dict):
            return {k: self.offline_normalize(v, from_axis, k) for k, v in poses.items()}
        elif data_key is not None:
            if data_key == 'left_hand_pose':
                poses = self.flip_left_hand_pose(poses)
            return self.normalizers[data_key].offline_normalize(poses, from_axis)
        else:
            return self._slice_and_process(poses, self.offline_normalize, from_axis)

    def offline_denormalize(self, poses, to_axis=False, data_key=None):
        if isinstance(poses, dict):
            return {k: self.normalizers[k].offline_denormalize(v, to_axis) for k, v in poses.items()}
        elif data_key is not None:
            poses = self.normalizers[data_key].offline_denormalize(poses, to_axis)
            if data_key == 'left_hand_pose':
                poses = self.flip_left_hand_pose(poses)
            return poses
        else:
            return self._slice_and_process(poses, self.offline_denormalize, to_axis)


def calculate_normalize_params(dataset, data_key='', rot_rep='axis', min_max=False,
                               batch_size=12800, num_workers=16, output_dir='', split_num=1):
    """
    Calculate normalization parameters for poses and global orientation.
    Global orientation is always included (always computed).
    
    Args:
        dataset: Dataset to compute statistics from
        data_key: Key for pose data in dataset batch (e.g., 'body_pose')
        rot_rep: Rotation representation ('axis' or 'rot6d')
        min_max: If True, use min-max normalization; else use z-score (mean/std)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        output_dir: Directory to save normalization files
        split_num: Number of splits for memory-efficient computation
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.makedirs(output_dir, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    all_data = []
    all_global_orient = []

    for batch in tqdm(dataloader, desc='fetching params'):
        poses = batch[data_key]
        pose_shape = poses.shape
        if rot_rep == 'rot6d':
            poses = axis_angle_to_rot6d(poses.reshape(-1, 3)).reshape(*pose_shape[:-1], -1)
        all_data.append(poses)
        
        # Always compute global_orient statistics
        if 'global_orient' in batch:
            global_orient = batch['global_orient']
            if isinstance(global_orient, torch.Tensor):
                all_global_orient.append(global_orient)
            else:
                all_global_orient.append(torch.tensor(global_orient, dtype=torch.float32))
        else:
            print("Warning: 'global_orient' not found in batch. Using zeros for global_orient statistics.")
            all_global_orient.append(torch.zeros(3, dtype=torch.float32))

    all_data_concat = torch.cat(all_data, dim=0)
    num_samples = all_data_concat.shape[0]
    split_size = num_samples // split_num

    print('data concatenated, calculating normalize params...')
    if min_max:
        min_values = []
        max_values = []
        for i in range(split_num):
            split_data = all_data_concat[i * split_size:(i + 1) * split_size].cuda()
            min_values.append(torch.quantile(split_data, 0.001, dim=0))
            max_values.append(torch.quantile(split_data, 0.999, dim=0))
            torch.cuda.empty_cache()
        min_percentile = torch.min(torch.stack(min_values), dim=0).values.cpu()
        max_percentile = torch.max(torch.stack(max_values), dim=0).values.cpu()
        
        save_dict = {'min_poses': min_percentile, 'max_poses': max_percentile}
        
        # Always compute global_orient statistics
        global_orient_concat = torch.cat(all_global_orient, dim=0)
        min_go_values = []
        max_go_values = []
        go_split_size = global_orient_concat.shape[0] // split_num
        for i in range(split_num):
            split_go = global_orient_concat[i * go_split_size:(i + 1) * go_split_size].cuda()
            min_go_values.append(torch.quantile(split_go, 0.001, dim=0))
            max_go_values.append(torch.quantile(split_go, 0.999, dim=0))
            torch.cuda.empty_cache()
        min_go_percentile = torch.min(torch.stack(min_go_values), dim=0).values.cpu()
        max_go_percentile = torch.max(torch.stack(max_go_values), dim=0).values.cpu()
        save_dict['min_global_orient'] = min_go_percentile
        save_dict['max_global_orient'] = max_go_percentile
        print(f'   Global orient range: min={min_go_percentile.numpy()}, max={max_go_percentile.numpy()}')
        
        torch.save(save_dict, os.path.join(output_dir, f'{rot_rep}_normalize1.pt'))
    else:  # mean and std
        means = []
        stds = []
        for i in range(split_num):
            split_data = all_data_concat[i * split_size:(i + 1) * split_size].cuda()
            means.append(torch.mean(split_data, dim=0))
            stds.append(torch.std(split_data, dim=0))
            torch.cuda.empty_cache()
        mean_poses = torch.mean(torch.stack(means), dim=0).cpu()
        std_poses = torch.sqrt(torch.mean(torch.stack(stds) ** 2, dim=0)).cpu()  # Aggregate stds
        
        save_dict = {'mean_poses': mean_poses, 'std_poses': std_poses}
        
        # Always compute global_orient statistics
        global_orient_concat = torch.cat(all_global_orient, dim=0)
        go_means = []
        go_stds = []
        go_split_size = global_orient_concat.shape[0] // split_num
        for i in range(split_num):
            split_go = global_orient_concat[i * go_split_size:(i + 1) * go_split_size].cuda()
            go_means.append(torch.mean(split_go, dim=0))
            go_stds.append(torch.std(split_go, dim=0))
            torch.cuda.empty_cache()
        mean_go = torch.mean(torch.stack(go_means), dim=0).cpu()
        std_go = torch.sqrt(torch.mean(torch.stack(go_stds) ** 2, dim=0)).cpu()
        save_dict['mean_global_orient'] = mean_go
        save_dict['std_global_orient'] = std_go
        print(f'   Global orient stats: mean={mean_go.numpy()}, std={std_go.numpy()}')
        
        torch.save(save_dict, os.path.join(output_dir, f'{rot_rep}_normalize2.pt'))
    torch.cuda.empty_cache()

    print(f'normalize params saved to {output_dir}')
