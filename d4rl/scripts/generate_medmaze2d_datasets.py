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
import ipdb
import tqdm



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

def append_data(data, s, a, img, tgt, done, env_data):
    try:
        # ipdb.set_trace()
        data['observations'].append(s)
    except:
        ipdb.set_trace()
        list(data['observation']).append(s)
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
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print(f"Video saved at {video_path}")

def reset_env(env, agent_centric=False):
    s = env.reset()
    env.set_target()
    if agent_centric:
        [env.render(mode='rgb_array') for _ in range(100)]    # so that camera can catch up with agent
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    # Save videos for trajectory
    parser.add_argument('--save_images', action='store_true', help='Whether rendered images are saved.')
    # maze2d-medium-v1
    # maze2d-large-v1
    parser.add_argument('--agent_centric', action='store_true', help='Whether agent-centric images are rendered.')

    parser.add_argument('--min_traj_len', type=int, default=int(5), help='Min number of samples per trajectory')
    parser.add_argument('--env_name', type=str, default='maze2d-medium-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--num_trajs', type=int, default=int(100), help='Num trajs to collect')

    parser.add_argument('--data_dir', type=str, default='.', help='Base directory for dataset')
    parser.add_argument('--batch_idx', type=int, default=int(-1), help='(Optional) Index of generated data batch')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)

    s = env.reset(10)
    targ_pos = (1,5)
    env.set_target(targ_pos)

    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts, cnt = 0, 0
    print(f"All init pos: {env.empty_and_goal_locations}")
    num_traj = 0
    subdir_name = f'init{int(s[0])}{int(s[1])}_targ{int(targ_pos[0])}{int(targ_pos[1])}'
    for _ in range(args.num_samples):
    # for tt in tqdm.tqdm(range(args.num_samples)):
        if ts == 0:
            # subdir_name = f"init{int(s[0])}{int(s[1])}"
            print(f"position: {s[0:2]}")
        position = s[0:2]
        velocity = s[2:4]
        act, done = controller.get_action(position, velocity, env._target)
        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True
        append_data(data, s, act, env.render(mode='rgb_array'), env._target, done, env.sim.data)

        ns, _, _, _ = env.step(act)

        if len(data['observations']) % 1000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            if len(data['actions']) > args.min_traj_len:
                if num_traj > 0:
                    save_data(args, data, cnt, subdir_name)
                    cnt += 1
                num_traj += 1
            # ipdb.set_trace()
            data = reset_data()
            controller = waypoint_controller.WaypointController(maze)
            env = maze_model.MazeEnv(maze)

            s = env.reset(10)
            # env.set_target()
            
            # env.set_target()
            done = False
            # s = reset_env(env, agent_centric=args.agent_centric)

            ts = 0
            if num_traj > args.num_trajs:
                break
        else:
            s = ns

        if args.render:
            env.render()

    # ipdb.set_trace()
    # maze_name = "medMaze"
    # if args.batch_idx >= 0:
    #     dir_name = 'maze2d-%s-noisy' % maze_name if args.noisy else 'maze2d-%s-sparse' % maze_name
    #     os.makedirs(os.path.join(args.data_dir, dir_name), exist_ok=True)
    #     fname = os.path.join(args.data_dir, dir_name, "rollouts_batch_{}.h5".format(args.batch_idx))
    # else:
    #     os.makedirs(args.data_dir, exist_ok=True)
    #     fname = 'maze2d-%s-noisy.hdf5' % maze_name if args.noisy else 'maze2d-%s-sparse.hdf5' % maze_name
    #     fname = os.path.join(args.data_dir, fname)

    # dataset = h5py.File(fname, 'w')
    # npify(data)
    # for k in data:
    #     dataset.create_dataset(k, data=data[k], compression='gzip')


def save_data(args, data, idx, subdir_name):
    # ipdb.set_trace()
    save_video("medMaze_scripts_{}.mp4".format(idx), data['images'])
    dir_name = ''
    if args.batch_idx >= 0:
        dir_name = os.path.join(dir_name, "batch_{}".format(args.batch_idx))
    os.makedirs(os.path.join(args.data_dir, subdir_name, dir_name), exist_ok=True)
    file_name = os.path.join(args.data_dir, subdir_name, dir_name, "rollout_{}.h5".format(idx))

    # save rollout to file
    f = h5py.File(file_name, "w")
    f.create_dataset("traj_per_file", data=1)

    # store trajectory info in traj0 group
    npify(data)
    traj_data = f.create_group("traj0")
    traj_data.create_dataset("observations", data=data['observations'])
    if args.save_images:
        traj_data.create_dataset("images", data=data['images'], dtype=np.uint8)
    else:
        traj_data.create_dataset("images", data=np.zeros((data['observations'].shape[0], 2, 2, 3), dtype=np.uint8))
    traj_data.create_dataset("actions", data=data['actions'])

    terminals = data['terminals']
    if np.sum(terminals) == 0:
        terminals[-1] = True

    # build pad-mask that indicates how long sequence is
    is_terminal_idxs = np.nonzero(terminals)[0]
    pad_mask = np.zeros((len(terminals),))
    pad_mask[:is_terminal_idxs[0]] = 1.
    traj_data.create_dataset("pad_mask", data=pad_mask)

    f.close()



if __name__ == "__main__":
    main()
