import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model, maze_layouts
import numpy as np
import torch
import pickle
import gzip
import h5py
import os
import argparse
import tqdm
import cv2
from spirl.utils.general_utils import AttrDict

from spirl.utils.vis_utils import add_caption_to_img, add_captions_to_seq

# python3 d4rl/scripts/generate_videos_from_actions.py --data_dir=data/{path to action seqs} --caption
def reset_data():
    return {'observations': [],
            'actions': [],
            'images':[],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, img, tgt, done, env_data, caption=False):
    try:
        data['observations'].append(s)
    except:
        list(data['observation']).append(s)
    # if caption:
        
    data['actions'].append(a)
    data['images'].append(img)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def save_video(info_dict, frames, fps=20, video_format='mp4'):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec based on the desired video format
    
    init_pos = str(info_dict['init'])
    targ_pos = str(info_dict['targ'])
    emb_dim = str(info_dict['emb_dim'])
    split_num = str(info_dict['split_num'])
    rollout_id = str(info_dict['rollout_id'])
    dens = str(info_dict['dens'])
    if info_dict['mapped']:
    	dens += "_mapped"
    decode_length = str(info_dict['decode_length'])

    video_dir_path = f"./videos/medMaze/dim{emb_dim}/init{init_pos}_targ{targ_pos}_dens{dens}/rollout{rollout_id}_split{split_num}/"
    os.makedirs(video_dir_path, exist_ok=True)
    video_file_name = f"video_action_length{decode_length}.mp4"
    video_path = os.path.join(video_dir_path, video_file_name)
    out = cv2.VideoWriter(video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    # Write each frame to the video
    for frame in frames:
        frame = frame.astype(np.uint8)
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print(f"Video saved at {video_path}")

def action_to_video(info_dict, action_seq):

    init_pos = str(info_dict['init'])
    targ_pos = str(info_dict['targ'])

    env_name = "maze2d-medium-v1"
    possible_pos = [(1, 1), (1, 2), (1, 5), (1, 6), (2, 1), (2, 2), (2, 4), (2, 5), (2, 6), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (5, 1), (5, 3), (5, 4), (5, 6), (6, 1), (6, 2), (6, 3), (6, 5), (6, 6)]

    init_pos = (int(init_pos[0]), int(init_pos[1]))
    targ_pos = [float(targ_pos[0]), float(targ_pos[1])]

    init_pos_idx = possible_pos.index((int(init_pos[0]),int(init_pos[1])))
    env = gym.make(env_name)
    maze = env.str_maze_spec
    max_episode_steps = 3000
    env = maze_model.MazeEnv(maze, targ_pos=targ_pos)
    
    s = env.reset(init_pos_idx)
    env.set_target(targ_pos)
    
    done = False
    data = reset_data()
    ts, cnt = 0, 0
    vids = []
    solve_thresh = 0.3

    images = action_seq
    min_distance = float('inf')
    max_episode_steps = len(action_seq)
    for act in action_seq:
        position = s[0:2]
        velocity = s[2:4]
        act = np.clip(act, -1.0, 1.0)
        append_data(data, s, act, env.render(mode='rgb_array'), env._target, done, env.sim.data)

        info = {'actions': act,
            'rewards': 0.0,
            'infos/goal': data['infos/goal'][-1],
            'infos/qpos': data['infos/qpos'][-1],
            'infos/qvel': data['infos/qvel'][-1] }
        vid = add_caption_to_img(data['images'][-1], info)
        vids.append(vid)
        ns, _, _, _ = env.step(act)


        # if len(data['observations']) % 100 == 0:
        #     print(len(data['observations']))
            # print(f"done: {done}")
        
        ts += 1
        distance_to_targ = np.linalg.norm(position - env._target)
        min_distance = min(distance_to_targ, min_distance)
        
        if ts == max_episode_steps:
            print(f"min_distance: {min_distance}")
            if min_distance < solve_thresh:
                # done = True
                success = 1
                print("Success!")
            else:
                success = 0
                print("Fail!")
            # if len(vids) > 0:
            #     save_video(info_dict, vids)
            # else:
            #     save_video(info_dict, data['images'])
            # print(f"total action: {len(data['actions'])}")
            
            # data = reset_data()
            # env = maze_model.MazeEnv(maze, targ_pos=targ_pos)
            # s = env.reset(init_pos_idx)
            
            # ts = 0
            # break
        else:
            s = ns
    return success, min_distance

if __name__ == "__main__":
    info_dict = {
        'init': 12,
        'targ': 65,
        'emb_dim': 20,
        'split_num': 1,
        'rollout_id': 0,
        'dens': 1000,
        'decode_length': 50,
        'mapped': False
    }

    traj_dict = dict()
    file_path = f"/home/yuhsiangwang/spirl/data/medMaze_dens{info_dict['dens']}/init{info_dict['init']}_targ{info_dict['targ']}/rollout_{info_dict['rollout_id']}.h5"
    with h5py.File(file_path, "r") as f:
        traj = f['traj0']
        for key in traj.keys():
            traj_dict[key] = torch.Tensor(traj[key][()])
            traj_dict = AttrDict(traj_dict)
    
    success, min_distance = action_to_video(info_dict, traj_dict.actions)
