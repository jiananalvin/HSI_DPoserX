import pickle
import numpy as np
import torch
import torch.nn as nn
from smplx import SMPL, SMPLH, SMPLX
from smplx.utils import Struct
import os

from lib.body_model import constants
from lib.body_model.joint_mapping import smpl_to_openpose
from lib.utils.transforms import rot6d_to_axis_angle

regressor_paths = ['/data3/ljz24/projects/3d/Hand4Whole/common/utils/human_model_files/smplx/SMPLX_to_J14.pkl',
                   '/data3/ljz24/projects/3d/body_models/smpl/J_regressor_h36m.npy']

def fullpose_to_params(fullpose):
    body_pose = fullpose[:, 3:3 + 63]
    jaw_pose = fullpose[:, 66:66 + 3]
    left_hand_pose = fullpose[:, 75:75 + 45]
    right_hand_pose = fullpose[:, 120:120 + 45]
    pose_params = torch.cat([body_pose, left_hand_pose, right_hand_pose, jaw_pose], dim=-1)
    return pose_params


class BodyModel(nn.Module):
    '''
    Wrapper around SMPLX body model class.
    from https://github.com/davrempe/humor/blob/main/humor/body_model/body_model.py
    '''

    def __init__(self,
                 bm_path,
                 num_betas=10,
                 batch_size=1,
                 num_expressions=10,  # JIANAN
                 model_type='smplx',
                 regressor_path=None,
                 device='cuda'):
        super(BodyModel, self).__init__()
        '''
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        :param device: Target device (CUDA/CPU) for all tensors
        '''

        # Normalize device to avoid 'cuda:0' vs 'cuda' mismatch
        self.device = torch.device(device)
        kwargs = {
            'model_type': model_type,
            'num_betas': num_betas,
            'batch_size': batch_size,
            'num_expression_coeffs': num_expressions,
            'use_pca': False,
            'flat_hand_mean': False,
            'use_face_contour': False,  # Disable ALL landmark/face processing
            'device': self.device
        }
        self.num_expressions = num_expressions
        assert (model_type in ['smpl', 'smplh', 'smplx'])
        
        if model_type == 'smpl':
            self.bm = SMPL(bm_path, **kwargs).to(self.device)
            # ✅ Move tensors while preserving nn.Parameter type
            self._safe_move_all_tensors_to_device(self.bm)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == 'smplh':
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            if model_type == 'smplh':
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate(
                    [data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM - B))],
                    axis=-1)
            kwargs['data_struct'] = data_struct
            self.bm = SMPLH(bm_path, **kwargs).to(self.device)
            # ✅ Move tensors while preserving nn.Parameter type
            self._safe_move_all_tensors_to_device(self.bm)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == 'smplx':
            # CRITICAL: No landmark tensors when use_face_contour=False (skip deletion)
            self.bm = SMPLX(bm_path, **kwargs).to(self.device)
            
            # ✅ SAFE ULTIMATE FIX: Move tensors while preserving nn.Parameter type
            self._safe_move_all_tensors_to_device(self.bm, recursive=True)
            
            # ✅ Verify ALL critical tensors are on GPU (debug)
            critical_tensors = [
                'pose_mean', 'v_template', 'shapedirs', 'expr_dirs', 
                'J_regressor', 'posedirs', 'lbs_weights'
            ]
            for t_name in critical_tensors:
                if hasattr(self.bm, t_name):
                    t = getattr(self.bm, t_name)
                    print(f"[DEBUG] {t_name} device: {t.device}, type: {type(t)}")  # Verify device/type
            
            self.left_hand_mean = self.bm.left_hand_mean.to(self.device)
            self.right_hand_mean = self.bm.right_hand_mean.to(self.device)
            self.hand_mean = torch.cat([self.left_hand_mean, self.right_hand_mean], dim=0).to(self.device)
            self.num_joints = SMPLX.NUM_JOINTS
            # create mean poses and shape for fitting initialization
            smpl_mean_params = np.load(constants.SMPL_MEAN_PATH)
            rot6d_poses = torch.tensor(smpl_mean_params['pose'], dtype=torch.float32, device=self.device)
            axis_poses = rot6d_to_axis_angle(rot6d_poses.reshape(-1, 6)).reshape(-1)
            mean_poses = self.bm.pose_mean.clone().to(self.device)
            mean_poses[:22 * 3] = axis_poses[:22 * 3]
            self.register_buffer('mean_poses', mean_poses)
            self.register_buffer('mean_shape', torch.tensor(smpl_mean_params['shape'], dtype=torch.float32, device=self.device))
            # misc data for evaluation
            self.face_vertex_idx = np.load(
                os.path.join(constants.BODY_MODEL_DIR, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
            with open(os.path.join(constants.BODY_MODEL_DIR, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
                self.hand_vertex_idx = pickle.load(f, encoding='latin1')
            self.vertex_num = 10475
            
        self.model_type = model_type
        self.faces = self.bm.faces_tensor.cpu().numpy()
        self.initial_batch_size = batch_size  # Store initial batch size (for fallback)

        # Move regressors to target device if needed
        if regressor_path is None and os.path.exists(regressor_paths[0]):
            regressor_path = regressor_paths
        if regressor_path is not None:
            for model_path in regressor_path:
                if 'SMPLX_to_J14.pkl' in model_path:
                    with open(model_path, 'rb') as f:
                        self.j14_regressor = pickle.load(f, encoding='latin1')
                elif 'J_regressor_h36m.npy' in model_path:
                    self.j17_regressor = np.load(model_path)

        self.J_regressor = self.bm.J_regressor.cpu().numpy()
        if model_type == 'smplx':
            self.orig_hand_regressor = self.make_hand_regressor()
        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}
        self.openpose_mapping = smpl_to_openpose(model_type=model_type, use_face_contour=False)  # Match disabled face contour

    def _safe_move_all_tensors_to_device(self, obj, recursive=False):
        '''
        Safely move ALL tensors in an object to target device, preserving nn.Parameter/nn.Buffer types.
        '''
        # First handle named parameters (preserve nn.Parameter)
        for name, param in obj.named_parameters(recurse=False):
            if param.device != self.device:
                setattr(obj, name, nn.Parameter(param.to(self.device), requires_grad=param.requires_grad))
        
        # Then handle named buffers (preserve nn.Buffer)
        for name, buf in obj.named_buffers(recurse=False):
            if buf.device != self.device:
                obj._buffers[name] = buf.to(self.device)
        
        # Then handle other tensor attributes (non-parameter/buffer)
        for name in dir(obj):
            if name.startswith('_') or name in obj._parameters or name in obj._buffers:
                continue  # Skip private/already handled attrs
            attr = getattr(obj, name)
            if isinstance(attr, torch.Tensor) and attr.device != self.device:
                setattr(obj, name, attr.to(self.device))
        
        # Recursively process child modules
        if recursive:
            for child in obj.children():
                self._safe_move_all_tensors_to_device(child, recursive=True)

    def make_hand_regressor(self, ):
        regressor = self.J_regressor.copy()
        lhand_regressor = np.concatenate((regressor[[20, 37, 38, 39], :], np.eye(self.vertex_num)[5361, None],
                                          regressor[[25, 26, 27], :], np.eye(self.vertex_num)[4933, None],
                                          regressor[[28, 29, 30], :], np.eye(self.vertex_num)[5058, None],
                                          regressor[[34, 35, 36], :], np.eye(self.vertex_num)[5169, None],
                                          regressor[[31, 32, 33], :], np.eye(self.vertex_num)[5286, None]))
        rhand_regressor = np.concatenate((regressor[[21, 52, 53, 54], :], np.eye(self.vertex_num)[8079, None],
                                          regressor[[40, 41, 42], :], np.eye(self.vertex_num)[7669, None],
                                          regressor[[43, 44, 45], :], np.eye(self.vertex_num)[7794, None],
                                          regressor[[49, 50, 51], :], np.eye(self.vertex_num)[7905, None],
                                          regressor[[46, 47, 48], :], np.eye(self.vertex_num)[8022, None]))
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

    def forward(self, global_orient=None, body_pose=None, left_hand_pose=None, right_hand_pose=None,
                jaw_pose=None, eye_poses=None, expression=None, betas=None, trans=None, dmpls=None,
                wholebody_params=None, return_dict=False, **kwargs):
        '''
        Note dmpls are not supported.
        '''
        assert (dmpls is None)
        assert 'pose_body' not in kwargs, 'use body_pose instead of pose_body'
        
        # ✅ SAFE FINAL SAFEGUARD: Re-move tensors (preserve parameter types)
        self._safe_move_all_tensors_to_device(self.bm, recursive=True)
        
        # Step 1: Get DYNAMIC batch size from input
        if body_pose is not None:
            batch_size = body_pose.shape[0]
        elif wholebody_params is not None:
            batch_size = wholebody_params.shape[0]
        else:
            batch_size = self.initial_batch_size

        # Step 2: Update SMPLX batch size dynamically (safe for all versions)
        if self.model_type == 'smplx':
            self.bm.batch_size = batch_size

        # Step 3: Parse wholebody_params (if provided)
        if wholebody_params is not None:
            assert self.model_type == 'smplx' and wholebody_params.shape[1] == 156 + self.num_expressions
            body_pose = wholebody_params[:, :63].to(self.device)
            left_hand_pose = wholebody_params[:, 63:63 + 45].to(self.device)
            right_hand_pose = wholebody_params[:, 63 + 45:63 + 45 + 45].to(self.device)
            jaw_pose = wholebody_params[:, 63 + 90:63 + 90 + 3].to(self.device)
            expression = wholebody_params[:, 63 + 90 + 3:].to(self.device)

        # Step 4: Explicitly set ALL params to match batch size AND DEVICE
        # Core params (force CUDA)
        global_orient = torch.zeros(batch_size, 3, device=self.device) if global_orient is None else global_orient.to(self.device)
        betas = torch.zeros(batch_size, 10, device=self.device) if betas is None else betas.to(self.device)
        body_pose = torch.zeros(batch_size, 63, device=self.device) if body_pose is None else body_pose.to(self.device)
        trans = torch.zeros(batch_size, 3, device=self.device) if trans is None else trans.to(self.device)

        # SMPLX-specific params (force batch size + CUDA)
        if self.model_type == 'smplx':
            left_hand_pose = torch.zeros(batch_size, 45, device=self.device) if left_hand_pose is None else left_hand_pose.to(self.device)
            right_hand_pose = torch.zeros(batch_size, 45, device=self.device) if right_hand_pose is None else right_hand_pose.to(self.device)
            jaw_pose = torch.zeros(batch_size, 3, device=self.device) if jaw_pose is None else jaw_pose.to(self.device)
            expression = torch.zeros(batch_size, self.num_expressions, device=self.device) if expression is None else expression.to(self.device)
            eye_poses = torch.zeros(batch_size, 6, device=self.device) if eye_poses is None else eye_poses.to(self.device)
        else:
            left_hand_pose = None
            right_hand_pose = None
            jaw_pose = None
            expression = None

        # Step 5: Sanity Checks (device + batch size)
        assert global_orient.device.type == self.device.type, \
            f"global_orient device type mismatch: {global_orient.device.type} != {self.device.type}"
        assert betas.device.type == self.device.type, \
            f"betas device type mismatch: {betas.device.type} != {self.device.type}"
        assert body_pose.device.type == self.device.type, \
            f"body_pose device type mismatch: {body_pose.device.type} != {self.device.type}"
        
        assert global_orient.shape[0] == batch_size, f"global_orient batch size mismatch: {global_orient.shape[0]} != {batch_size}"
        assert betas.shape[0] == batch_size, f"betas batch size mismatch: {betas.shape[0]} != {batch_size}"
        assert body_pose.shape[0] == batch_size, f"body_pose batch size mismatch: {body_pose.shape[0]} != {batch_size}"
        
        if self.model_type == 'smplx':
            assert left_hand_pose.device.type == self.device.type, \
                f"left_hand_pose device type mismatch: {left_hand_pose.device.type} != {self.device.type}"
            assert right_hand_pose.device.type == self.device.type, \
                f"right_hand_pose device type mismatch: {right_hand_pose.device.type} != {self.device.type}"
            assert jaw_pose.device.type == self.device.type, \
                f"jaw_pose device type mismatch: {jaw_pose.device.type} != {self.device.type}"
            assert left_hand_pose.shape[0] == batch_size, f"left_hand_pose batch size mismatch"
            assert right_hand_pose.shape[0] == batch_size, f"right_hand_pose batch size mismatch"
            assert jaw_pose.shape[0] == batch_size, f"jaw_pose batch size mismatch"

        # Step 6: Forward pass (100% safe, no landmarks)
        out_obj = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=trans,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=eye_poses[:, :3] if eye_poses is not None else None,
            reye_pose=eye_poses[:, 3:] if eye_poses is not None else None,
            return_full_pose=True,
            return_landmarks=False,  # Explicitly disable landmarks (redundant but safe)
            **kwargs
        )

        # Step 7: Build output (all tensors stay on CUDA)
        out = {
            'v': out_obj.vertices,
            'f': self.bm.faces_tensor,
            'betas': out_obj.betas,
            'Jtr': out_obj.joints,
            'OpJtr': out_obj.joints[:, self.openpose_mapping],
            'body_joints': out_obj.joints[:, :22],
            'body_pose': out_obj.body_pose,
            'full_pose': out_obj.full_pose,
            'global_orient': out_obj.global_orient,
            'transl': trans,
        }
        if self.model_type in ['smplh', 'smplx']:
            out['hand_poses'] = torch.cat([out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1)
        if self.model_type == 'smplx':
            out['jaw_pose'] = out_obj.jaw_pose
            out['expression'] = out_obj.expression
            out['eye_poses'] = eye_poses

        if not return_dict:
            out = Struct(**out)

        return out
