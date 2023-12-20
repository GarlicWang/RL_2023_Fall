import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model, maze_layouts
import numpy as np
import pickle
import gzip
import h5py
import os
import argparse
# import ipdb
import tqdm

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
        # ipdb.set_trace()
        data['observations'].append(s)
    except:
        # ipdb.set_trace()
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

def save_video(file_name, frames, fps=20, video_format='mp4'):
    import cv2 
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec based on the desired video format
    video_path = os.path.join('./videos/medMaze/', file_name )
    out = cv2.VideoWriter(video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    # Write each frame to the video
    for frame in frames:
        frame = frame.astype(np.uint8)
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print(f"Video saved at {video_path}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--init_pos',type=str, default='11', help='initial postion')
    parser.add_argument('--targ_pos',type=str, default='65',help='target postion')
    parser.add_argument('--data_dir', type=str, default='.', help='Base directory for dataset')
    parser.add_argument('--caption', action='store_true', help='add caption to video actions')
    # init_

    args = parser.parse_args()
    env_name = "maze2d-medium-v1"
    possible_pos = [(1, 1), (1, 2), (1, 5), (1, 6), (2, 1), (2, 2), (2, 4), (2, 5), (2, 6), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6), (5, 1), (5, 3), (5, 4), (5, 6), (6, 1), (6, 2), (6, 3), (6, 5), (6, 6)]

    # targ_pos = [6., 5.]
    # init_pos = [1., 1.]
    init_pos = (int(args.init_pos[0]), int(args.init_pos[1]))
    targ_pos = [float(args.targ_pos[0]), float(args.targ_pos[1])]

    init_pos_idx = possible_pos.index((int(init_pos[0]),int(init_pos[1])))
    env = gym.make(env_name)
    maze = env.str_maze_spec
    max_episode_steps = 3000
    env = maze_model.MazeEnv(maze, targ_pos=targ_pos)
    
    s = env.reset(init_pos_idx)
    env.set_target(targ_pos)
    
    done = False
    data = reset_data()
    ts, cnt, tts = 0, 0, 0
    vids = []
    solve_thresh = 0.1
    f_num = 0

    for file in os.listdir(args.data_dir):
        # print(f)
        file = h5py.File(os.path.join(args.data_dir, file), mode="r+")
        print(file.keys())
        traj0 = file['traj0']
        traj_per_file = file['traj_per_file']
        # ipdb.set_trace()
        actions = traj0['actions']
        images = traj0['actions']
        # ipdb.set_trace()
        max_episode_steps = len(traj0['actions'])
        for act in traj0['actions']:
            position = s[0:2]
            velocity = s[2:4]
            act = np.clip(act, -1.0, 1.0)
            append_data(data, s, act, env.render(mode='rgb_array'), env._target, done, env.sim.data)

            if args.caption:
                info = {'actions': act,
                    'rewards': 0.0,
                    'infos/goal': data['infos/goal'][-1],
                    'infos/qpos': data['infos/qpos'][-1],
                    'infos/qvel': data['infos/qvel'][-1] }
                vid = add_caption_to_img(data['images'][-1], info)
                vids.append(vid)
            ns, _, _, _ = env.step(act)


            if len(data['observations']) % 100 == 0:
                print(len(data['observations']))
                print(f"done: {done=}")
            
            ts += 1
            if ts == max_episode_steps or (np.linalg.norm(position - env._target)<solve_thresh):
                if (np.linalg.norm(position - env._target)<solve_thresh):
                    done = True
                    print("====success====")
                else:
                    print("====failure====")
                    print('ts')
                if len(vids) > 0:
                    save_video("medMaze_transfer_{}.mp4".format(f_num), vids)
                else:
                    save_video("medMaze_transfer_{}.mp4".format(f_num), data['images'])
                data = reset_data()
                env = maze_model.MazeEnv(maze, targ_pos=targ_pos)
                s = env.reset(init_pos_idx)
                
                print(f"total action: {len(data['actions'])}")
                ts = 0
                tts += 1
                break

            else:
                s = ns
                    
        f_num += 1
        file.close()




            


if __name__ == "__main__":
    main()