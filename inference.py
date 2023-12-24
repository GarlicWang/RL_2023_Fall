import os
import torch
os.environ['DATA_DIR'] = "./data"
os.environ['EXP_DIR'] = "./experiments"

from spirl.components.data_loader import GlobalSplitVideoDataset
from spirl.models.skill_prior_mdl import ImageSkillPriorMdl
from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.utils.pytorch_utils import map2torch
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import map_dict
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.modules.recurrent_modules import RecurrentPredictor
from spirl.modules.variational_inference import Gaussian
from scripts.generate_videos_from_actions import action_to_video
import pandas as pd
import numpy as np
import imp
import h5py
import json

def get_traj(dens, init, targ, rollout_id):
    traj_dict = dict()
    file_path = f"spirl/data/medMaze_dens{dens}/init{init}_targ{targ}/rollout_{rollout_id}.h5"
    with h5py.File(file_path, "r") as f:
        traj = f['traj0']
        for key in traj.keys():
            traj_dict[key] = torch.Tensor(traj[key][()])
        traj_dict = AttrDict(traj_dict)
    return traj_dict

def load_model(dens, dim):
    model_path = f"spirl/experiments/skill_prior_learning/maze/hierarchical/24task_1000rollout_dens{dens}_dim{dim}/weights/weights_ep199.pth"
    model_config = {'state_dim': 4, 'action_dim': 2, 'n_rollout_steps': 10, 'kl_div_weight': 0.01, 'prior_input_res': 32, 'n_input_frames': 2, 'batch_size': 1, 'dataset_class': GlobalSplitVideoDataset, 'n_actions': 2, 'split': {'train': 0.9, 'val': 0.1, 'test': 0.0}, 'res': 32, 'crop_rand_subseq': True, 'max_seq_len': 300, 'subseq_len': 12, 'device': 'cpu'}
    skill_prior_ = ImageSkillPriorMdl(model_config)
    if dim == 128: # need to rebuild the network
        skill_prior_._hp.nz_vae, skill_prior_._hp.nz_enc = 64, 64
        skill_prior_.build_network()
    skill_prior_.load_state_dict(torch.load(model_path)['state_dict'])
    return skill_prior_

def encode(skill_prior_, action_seq_):
    z = skill_prior_.q(action_seq_.reshape(1,-1,2))[:,-1]
    z_sample = MultivariateGaussian(z).sample()
    return z, z_sample

def decode(skill_prior_, emb, step):
    cond_inputs = torch.zeros(1,262,3,32,32)
    lstm_init_input = skill_prior_.decoder_input_initalizer(cond_inputs) # torch.Size([1, 2])
    lstm_init_hidden = skill_prior_.decoder_hidden_initalizer(cond_inputs) # torch.Size([1, 256])
    reconstructed_action_seq_ = skill_prior_.decoder(lstm_initial_inputs=AttrDict(x_t=lstm_init_input),
                                lstm_static_inputs=AttrDict(z=emb),
                                steps=step,
                                lstm_hidden_init=lstm_init_hidden).pred
    return reconstructed_action_seq_

def get_action_subseq(traj_dict_, split_num=1):
    subseq_length = traj_dict_.actions.shape[0]/((split_num+1)/2)
    action_subseq_list = [traj_dict_.actions[int(subseq_length*i/2):int(subseq_length*(i/2+1))] for i in range(split_num)]
    return action_subseq_list



if __name__ == "__main__":
    train_tasks = [(12, 16)]
    test_tasks = [(11,56), (16,56), (51,56)]
    
    info_dict = {
    'init': 16,
    'targ': 56,
    'emb_dim': 20,
    'split_num': 5,
    'rollout_id': 0,
    'dens': 1000,
    'decode_length': 100,
    'mapped': True
    }
    test_mode = True
    random_mode = True
    print(f"random: {random_mode}")
    
    inference_train_df = pd.read_parquet("inference_train.parquet")
    inference_df = pd.read_parquet("inference.parquet")
    skill_prior = load_model(dens=1000, dim=20)
    min_distance_dict = {}
    success_rate_dict = {}
    
    if test_mode:
        tasks = test_tasks
    else:
        tasks = train_tasks
        
    for init, targ in tasks:
        print(f"init: {init}, targ: {targ}")
        info_dict['init'], info_dict['targ'] = init, targ
        success_count = 0
        min_distance_list = []
        for rollout_id in range(1000):
            info_dict['rollout_id'] = rollout_id
            if test_mode:
                emb_list = [torch.Tensor(inference_df.loc[(inference_df["init"]==init) & (inference_df["targ"]==targ) & (inference_df["rollout_id"]==rollout_id) &  (inference_df["seq"]==seq), "embedding"].iloc[0]).reshape(1, -1) for seq in ["0", "2", "4"]]
            else:
                emb_list = [torch.Tensor(inference_train_df.loc[(inference_train_df["init"]==init) & (inference_train_df["targ"]==targ) & (inference_train_df["rollout_id"]==rollout_id) &  (inference_train_df["seq"]==seq), "embedding"].iloc[0]).reshape(1, -1) for seq in ["0", "2", "4"]]
            emb_sample_list = [MultivariateGaussian(emb).sample() for emb in emb_list]
            reconstructed_action_seq = []
            
            if not random_mode:
                for emb in emb_sample_list:
                    reconstructed_action_subseq = decode(skill_prior, emb=emb, step=100)
                    reconstructed_action_seq.append(reconstructed_action_subseq.squeeze())
            else:
                for _ in range(3):
                    random_emb = Gaussian(torch.zeros((1, 20), device="cpu")).sample()
                    reconstructed_action_subseq = decode(skill_prior, emb=random_emb, step=100)
                    reconstructed_action_seq.append(reconstructed_action_subseq.squeeze())
                
            reconstructed_action_seq = torch.cat(reconstructed_action_seq).detach().numpy()
            success, min_distance = action_to_video(info_dict, reconstructed_action_seq)
            min_distance_list.append(min_distance)
            success_count += success
            success_rate = success_count/(rollout_id+1)
            print(f"success rate: {success_count}/{rollout_id+1}")
        min_distance_dict[f"init{init}_targ{targ}"] = min_distance_list
        success_rate_dict[f"init{init}_targ{targ}"] = success_rate
        with open ("min_distance_train_random.json", "w") as f:
            json.dump(min_distance_dict, f)
        with open ("success_rate_train_random.json", "w") as f:
            json.dump(success_rate_dict, f)

        