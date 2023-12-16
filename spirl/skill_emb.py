import os
import torch
import h5py
from spirl.utils.general_utils import AttrDict
from spirl.models.skill_prior_mdl import ImageSkillPriorMdl
from spirl.components.data_loader import GlobalSplitVideoDataset
from spirl.modules.variational_inference import MultivariateGaussian

class spirlModel:
	def __init__(self):
		self.model_path = "./experiments/skill_prior_learning/maze/hierarchical/weights/weights_ep199.pth"
		assert os.path.isfile(self.model_path), "model not found"
		self.model_config = {'state_dim': 4, 'action_dim': 2, 'n_rollout_steps': 10, 'kl_div_weight': 0.01, 'prior_input_res': 32, 'n_input_frames': 2, 'batch_size': 1, 'dataset_class': GlobalSplitVideoDataset, 'n_actions': 2, 'split': {'train': 0.9, 'val': 0.1, 'test': 0.0}, 'res': 32, 'crop_rand_subseq': True, 'max_seq_len': 300, 'subseq_len': 12, 'device': 'cpu'}
		self.skill_prior = ImageSkillPriorMdl(self.model_config)
		self.skill_prior.load_state_dict(torch.load(self.model_path)['state_dict'])
		self.encoder = self.skill_prior.q

def load_traj(init, targ, rollout_id):
	file_path = f"data/data_medMaze_scripts/init{init}_targ{targ}/rollout_{rollout_id}.h5"
	assert os.path.isfile(file_path), "traj file not found"
	traj_dict = dict()
	with h5py.File(file_path, "r") as f:
		traj = f['traj0']
		for key in traj.keys():
			traj_dict[key] = torch.Tensor(traj[key][()])
		traj_dict = AttrDict(traj_dict)
	return traj_dict

if __name__ == "__main__":
	spirl_model = spirlModel()
	traj_dict = load_traj(12, 61, 3)
	z = spirl_model.encoder(traj_dict.actions.reshape(1,-1,2))[:,-1] # reshape to (batch_size, action_length, action_dim), only use the embed from the last cell of LSTM
	z_sample = MultivariateGaussian(z).sample() # original spirl adopts sampling, which introduce randomness
	print(f"embedding: {z}")
	print(f"sampled embedding: {z_sample}")
