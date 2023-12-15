export EXP_DIR=./experiments
export DATA_DIR=./data
export export CUDA_VISIBLE_DEVICES=0
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/YOUR_PATH/.mujoco/mujoco210/bin

### train RL model by pretrained skill prior
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/spirl_cl --seed=0 --prefix=generated_maze_seed0
