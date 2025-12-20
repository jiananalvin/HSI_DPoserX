import os
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append("/home/jxudt/DPoser-X")  # Add DPoser-X root to path
from lib.body_model.body_model import BodyModel
from lib.body_model.visual import multiple_render 


# class AMASSDataset(torch.utils.data.Dataset):
#     def __init__(self, root_path, version='version0', subset='train', sample_interval=None,):

#         self.root_path = root_path
#         self.version = version
#         assert subset in ['train', 'valid', 'test']
#         self.subset = subset
#         self.sample_interval = sample_interval

#         self.global_orients, self.body_poses = self.read_data()

#         if self.sample_interval:
#             self._sample(sample_interval)

#         self.real_data_len = len(self.body_poses)

#     def __getitem__(self, idx):
#         """
#         Return:
#             [21, 3] or [21, 6] for poses including body and root orient
#             [10] for shapes (betas)  [Optimal]
#         """
#         global_orient = self.global_orients[idx % self.real_data_len]
#         body_pose = self.body_poses[idx % self.real_data_len]
#         data_dict = {'global_orient': global_orient, 'body_pose': body_pose}

#         return data_dict

#     def __len__(self, ):
#         return len(self.body_poses)

#     def _sample(self, sample_interval):
#         print(f'Class AMASSDataset({self.subset}): sample dataset every {sample_interval} frame')
#         self.global_orients = self.global_orients[::sample_interval]
#         self.body_poses = self.body_poses[::sample_interval]

#     def read_data(self):
#         data_path = os.path.join(self.root_path, self.version, self.subset)
#         global_orient = torch.load(os.path.join(data_path, 'root_orient.pt'))
#         body_pose = torch.load(os.path.join(data_path, 'pose_body.pt'))

#         return global_orient, body_pose

import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np

class AMASSDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        # Keep original args (for compatibility with DPoser-X's training code)
        root_path,          # Unused (but kept to avoid changing training scripts)
        version='version0', # Unused (kept for compatibility)
        subset='train', 
        sample_interval=None,
        # PoseScript-specific args (hardcode if you want, or pass in training script)
        posescript_data_dir="/path/to/posescript/data",
        amass_dir="/path/to/raw/amass",
        use_human_annotations=False
    ):
        # --------------------------
        # Core PoseScript Setup (no AMASS logic)
        # --------------------------
        assert subset in ['train', 'val', 'test']
        self.subset = subset
        self.sample_interval = sample_interval
        self.posescript_data_dir = posescript_data_dir
        self.amass_dir = amass_dir
        self.use_human_annotations = use_human_annotations

        # Load PoseScript metadata (IDs, captions, AMASS mappings)
        self._load_posescript_metadata()

        # Apply sampling interval (DPoser-X's original logic)
        if self.sample_interval:
            self._sample(sample_interval)

        # Dataset length (PoseScript only)
        self.real_data_len = len(self.valid_split_ids)
        print(f"PoseScriptDataset initialized (subset: {subset}, size: {self.real_data_len})")

    # --------------------------
    # PoseScript Metadata Loading (no AMASS code)
    # --------------------------
    def _load_posescript_metadata(self):
        # 1. Load split IDs (train/val/test) - match your 100k file names
        split_file = f"{self.subset}_ids_100k.json"  # e.g., train_ids_100k.json, test_ids_100k.json
        split_path = os.path.join(self.posescript_data_dir, split_file)
        with open(split_path, "r") as f:
            self.split_ids = json.load(f)
        print(f"Loaded {len(self.split_ids)} raw {self.subset} PoseScript IDs")

        # 2. Load captions (use the same caption files for all splits, as is standard)
        if self.use_human_annotations:
            caption_file = "posescript_human_6293.json"    # human caption file
        else:
            caption_file = "posescript_auto_100k.json"  # auto caption file
        caption_path = os.path.join(self.posescript_data_dir, caption_file)        
        self.valid_split_ids, self.captions = self._load_posescript_captions([caption_path], self.split_ids)

        # 3. Load PoseScript → AMASS mapping (same for all splits)
        mapping_file = "ids_2_dataset_sequence_and_frame_index_100k.json"  # mapping file
        mapping_path = os.path.join(self.posescript_data_dir, mapping_file)
        with open(mapping_path, "r") as f:
            self.pose_mappings = json.load(f)

        # 4. Precompute caption list (one per valid ID)
        self.caption_list = [self.captions[pid][0] for pid in self.valid_split_ids]

    def _load_posescript_captions(self, caption_files, split_ids):
        captions = {pose_id: [] for pose_id in split_ids}
        for caption_file in caption_files:
            with open(caption_file, "r") as f:
                capts = json.load(f)
            for pose_id_str, caption_list in capts.items():
                pose_id = int(pose_id_str)
                if pose_id in captions:
                    captions[pose_id].extend(caption_list if isinstance(caption_list, list) else [caption_list])
        # Filter valid IDs (with captions)
        valid_split_ids = [pid for pid in split_ids if len(captions[pid]) > 0]
        valid_captions = {pid: captions[pid] for pid in valid_split_ids}
        print(f"Filtered to {len(valid_split_ids)} valid PoseScript poses")
        return valid_split_ids, valid_captions

    # --------------------------
    # DPoser-X Compatible Pose Loading (on-the-fly)
    # --------------------------
    def _get_posescript_pose(self, pose_id):
        """Load AMASS pose in DPoser-X format (3D global_orient + 63D body_pose)"""
        pose_id_str = str(pose_id)
        amass_info = self.pose_mappings[pose_id_str]  # [dataset, sequence, frame_idx]
        dataset_name, sequence_name, frame_idx = amass_info
        frame_idx = int(frame_idx)

        # Load raw AMASS sequence (on-the-fly)
        amass_seq_path = os.path.join(self.amass_dir, dataset_name, "sequences", sequence_name)
        if not os.path.exists(amass_seq_path):
            raise FileNotFoundError(f"AMASS sequence missing: {amass_seq_path}")
        
        # Extract DPoser-X's standard pose parameters
        seq_data = np.load(amass_seq_path)
        smplh_pose = seq_data["poses"][frame_idx]  # (156,) SMPL-H pose
        global_orient = torch.tensor(smplh_pose[:3], dtype=torch.float32)  # 3D root
        body_pose = torch.tensor(smplh_pose[3:66], dtype=torch.float32)    # 63D body
        return global_orient, body_pose

    # --------------------------
    # DPoser-X's Original Methods (simplified for PoseScript)
    # --------------------------
    def _sample(self, sample_interval):
        """Subsample (exact DPoser-X logic, adapted for PoseScript)"""
        print(f"PoseScriptDataset({self.subset}): sampling every {sample_interval} frame")
        self.valid_split_ids = self.valid_split_ids[::sample_interval]
        self.caption_list = self.caption_list[::sample_interval]

    def __len__(self):
        return self.real_data_len

    def __getitem__(self, idx):
        """Exact output format DPoser-X expects (+ optional caption)"""
        idx = idx % self.real_data_len  # Original wrap-around logic

        # Load PoseScript pose (DPoser-X format)
        pose_id = self.valid_split_ids[idx]
        global_orient, body_pose = self._get_posescript_pose(pose_id)

        # Output dict (matches DPoser-X's original AMASSDataset)
        data_dict = {
            'global_orient': global_orient,    # [3] tensor (DPoser-X standard)
            'body_pose': body_pose,            # [63] tensor (DPoser-X standard)
            'caption': self.caption_list[idx], # For text conditioning
            'pose_id': pose_id
        }

        return data_dict
    

if __name__ == "__main__":
    # --------------------------
    # Config (UPDATE THESE PATHS!)
    # --------------------------
    TEST_CONFIG = {
        "posescript_data_dir": "/home/jxudt/HSI_GenPose/data/posescript_release",  # PoseScript JSONs (train_ids.json, etc.)
        "amass_dir": "/home/jxudt/HSI_GenPose/data/amass",               # Raw AMASS sequences (e.g., AMASS/CMU/sequences)
        "bodymodel_path": "/home/jxudt/DPoser-X/body_models/smplx/SMPLX_NEUTRAL.npz",  # Your SMPLX model
        "output_path": "/home/jxudt/DPoser-X/output/posescript_test",  # Save renders here
        "subset": "val",                                         # Train/val/test split
        "sample_interval": 1,
        "num_samples": 5,                                        # Number of samples to test
        # "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device":"cpu",
        "faster_render": True,                                   # Faster (lower quality) render (DPoser-X flag)
        "view": "front"                                          # Render view (front/left/right/back)
    }

    # --------------------------
    # 1. Initialize PoseScript Dataset
    # --------------------------
    dataset = AMASSDataset(
        root_path="/unused",  # Unused (compatibility with DPoser-X)
        subset=TEST_CONFIG["subset"],
        sample_interval=TEST_CONFIG["sample_interval"],
        posescript_data_dir=TEST_CONFIG["posescript_data_dir"],
        amass_dir=TEST_CONFIG["amass_dir"],
        use_human_annotations=True
    )

    # --------------------------
    # Print sample keys
    # --------------------------
    print("\\n[info] Debug: Check sample keys (first sample):")
    first_sample = dataset[0]
    print(f"Available keys in sample: {list(first_sample.keys())}")  # Available keys in sample: ['global_orient', 'body_pose', 'caption']
    print("-" * 50)

    # --------------------------
    # 2. Load DPoser-X's BodyModel (exact as demo)
    # --------------------------
    body_model = BodyModel(
        bm_path=TEST_CONFIG["bodymodel_path"],
        num_betas=10,
        batch_size=TEST_CONFIG["num_samples"],
        model_type='smplx'
    ).to(TEST_CONFIG["device"])

    # --------------------------
    # 3. DPoser-X's Rendering Params (match demo)
    # --------------------------
    bg_img = np.ones([512, 384, 3]) * 255  # White background (demo default)
    focal = [1500, 1500]                   # Focal length (demo default)
    princpt = [200, 192]                   # Principal point (demo default)

    # --------------------------
    # 4. Load Test Samples (Pose + Caption)
    # --------------------------
    print(f"\n[info] Loading {TEST_CONFIG['num_samples']} test samples...")
    body_pose_batch = []  # Only body_pose (63 values) for rendering
    full_pose_batch = []  # Optional: global_orient + body_pose (66 values) for other uses
    captions = []
    pose_ids = []

    for i in range(TEST_CONFIG["num_samples"]):
        sample = dataset[i]
        # Extract body_pose (63 values) for rendering (SMPLX expects this!)
        body_pose = sample["body_pose"]  # Shape: [63]
        # Optional: Combine global_orient + body_pose (for DPoser-X model input later)
        full_pose = torch.cat([sample["global_orient"], sample["body_pose"]], dim=0)
        
        body_pose_batch.append(body_pose)
        full_pose_batch.append(full_pose)
        captions.append(sample["caption"])
        pose_ids.append(sample["pose_id"])

    # Convert to batch tensor (shape: [5, 63] for body_pose)
    body_pose_batch = torch.stack(body_pose_batch).to(TEST_CONFIG["device"])

    # --------------------------
    # 5. Render Poses (DPoser-X's official multiple_render)
    # --------------------------
    target_path = os.path.join(TEST_CONFIG["output_path"], "posescript_samples")
    os.makedirs(target_path, exist_ok=True)

    # Use DPoser-X's multiple_render (pass ONLY body_pose!)
    multiple_render(
        samples=body_pose_batch,  # <-- Changed from pose_batch (full_pose) to body_pose_batch
        denormalize_fn=None,  
        model=body_model,
        target_path=target_path,
        img_name="posescript_sample_{}.png",
        part='body',          
        convert=False,        
        faster=TEST_CONFIG["faster_render"],
        device=TEST_CONFIG["device"],
        bg_img=bg_img,
        focal=focal,
        princpt=princpt,
        view=TEST_CONFIG["view"]
    )

    # --------------------------
    # 6. Verify Pose-Caption Alignment
    # --------------------------
    # Save caption-PoseID mapping (JSON for reference)
    caption_mapping = {
        f"posescript_sample_{i+1}.png": {
            "pose_id": pose_ids[i],
            "caption": captions[i]
        } for i in range(TEST_CONFIG["num_samples"])
    }

    mapping_path = os.path.join(target_path, "caption_pose_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(caption_mapping, f, indent=2, ensure_ascii=False)

    # Print alignment to console (quick check)
    print("\n Pose-Caption Alignment Check:")
    print("-" * 80)
    for i in range(TEST_CONFIG["num_samples"]):
        img_name = f"posescript_sample_{i+1}.png"
        print(f"Image: {img_name}")
        print(f"Pose ID: {pose_ids[i]}")
        print(f"Caption: {captions[i][:100]}..." if len(captions[i])>100 else f"Caption: {captions[i]}")
        print("-" * 80)

    # --------------------------
    # 7. Final Output
    # --------------------------
    print(f"\n Test Complete!")
    print(f"   - Rendered images: {target_path}")
    print(f"   - Caption-Pose mapping: {mapping_path}")
    print(f"   - Verify: Check the rendered images and captions for alignment!")