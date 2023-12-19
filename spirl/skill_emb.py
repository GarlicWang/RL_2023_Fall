import os
import torch
import h5py
import pandas as pd
import multiprocessing as mp
from spirl.utils.general_utils import AttrDict
from spirl.models.skill_prior_mdl import ImageSkillPriorMdl
from spirl.components.data_loader import GlobalSplitVideoDataset
from spirl.modules.variational_inference import MultivariateGaussian


class spirlModel:
    def __init__(self, dens=1000, dim=20):
		assert dens in [250, 1000], "invalid density"
		assert dim in [20, 128], "invalid dimension"
        self.model_path = f"./experiments/skill_prior_learning/maze/hierarchical/24task_1000rollout_dens{str(dens)}_dim{str(dim)}/weights/weights_ep199.pth"
        assert os.path.isfile(self.model_path), "model not found"
        self.model_config = {
            "state_dim": 4,
            "action_dim": 2,
            "n_rollout_steps": 10,
            "kl_div_weight": 0.01,
            "prior_input_res": 32,
            "n_input_frames": 2,
            "batch_size": 1,
            "dataset_class": GlobalSplitVideoDataset,
            "n_actions": 2,
            "split": {"train": 0.9, "val": 0.1, "test": 0.0},
            "res": 32,
            "crop_rand_subseq": True,
            "max_seq_len": 300,
            "subseq_len": 12,
            "device": "cpu",
        }
        self.skill_prior = ImageSkillPriorMdl(self.model_config)
        if dim == 128: # need to rebuild the network
            skill_prior._hp.nz_vae, skill_prior._hp.nz_enc = 64, 64
            skill_prior.build_network()
        self.skill_prior.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.encoder = self.skill_prior.q


def load_traj(init, targ, rollout_id):
    file_path = (
        f"data/data_medMaze_scripts/init{init}_targ{targ}/rollout_{rollout_id}.h5"
    )
    assert os.path.isfile(file_path), "traj file not found"
    traj_dict = dict()
    with h5py.File(file_path, "r") as f:
        traj = f["traj0"]
        for key in traj.keys():
            traj_dict[key] = torch.Tensor(traj[key][()])
        traj_dict = AttrDict(traj_dict)
    return traj_dict


def process_traj(init, targ):
    print(f"init: {init}, targ: {targ}")
    df = pd.DataFrame(
        columns=["init", "targ", "rollout_id", "embedding", "sampled_embedding"]
    )
    for rollout_id in range(1000):
        traj_dict = load_traj(init, targ, rollout_id)
        z = spirl_model.encoder(traj_dict.actions.reshape(1, -1, 2))[:, -1]
        z_sample = MultivariateGaussian(z).sample()

        ### split rollout to multiple subsequences
        split_num = 5 # choose an odd number
        subseq_length = traj_dict.actions.shape[0]/((split_num+1)/2) # with half subsequence overlapping
        action_subseq_list = [traj_dict.actions[int(subseq_length*i/2):int(subseq_length*(i/2+1))] for i in range(split_num)] # note that each subsequence may have different length due to the int()
        z_list = []
        for action_subseq in action_subseq_list:
        	z_list.append(skill_prior.q(action_subseq.reshape(1,-1,2))[:,-1])
        ###

        df.loc[len(df.index)] = [
            init,
            targ,
            rollout_id,
            z.squeeze().detach().numpy().tolist(),
            z_sample.squeeze().detach().numpy().tolist(),
        ]

    df.to_csv(f"domainA/init{init}_targ{targ}.csv", index=False)
    return True


train_tasks = [
    (12, 16),
    (12, 61),
    (12, 65),
    (15, 11),
    (15, 61),
    (15, 65),
    (21, 16),
    (21, 61),
    (21, 65),
    (26, 11),
    (26, 61),
    (26, 65),
    (46, 11),
    (46, 16),
    (46, 61),
    (51, 11),
    (51, 16),
    (51, 65),
    (63, 11),
    (63, 16),
    (63, 65),
    (65, 11),
    (65, 16),
    (65, 61),
]

if __name__ == "__main__":
    df = pd.DataFrame(
        columns=["init", "targ", "rollout_id", "embedding", "sampled_embedding"]
    )

    spirl_model = spirlModel()
    N = mp.cpu_count()

    with mp.Pool(processes=N - 2) as p:
        results = p.starmap(
            process_traj,
            train_tasks,
        )
    print(results)
