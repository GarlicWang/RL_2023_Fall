# RL_2023_Fall
Final Project for RL 2023 Fall

Topic: Skill-Based Domain-Transfer Meta-RL 

## Environment
```bash
# Activate conda
conda create -n sbtmdrl python=3.9
conda activate sbtmdrl

# Install Necessary Package
pip install -r requirements.txt

cd spirl
pip install -r requirements.txt
pip install -e .
cd ..

cd d4rl
pip install -e .
cd ..
```

Set the environment variables that specify the root experiment and data directories.
```bash
mkdir ./spirl/experiments
mkdir ./spirl/data
export EXP_DIR=./spirl/experiments
export DATA_DIR=./spirl/data
```

## Reproduce
1. Generate Script policy rollout and place it in data/maze/
```bash
# generate script policy
python3 generate_medMaze2d_datasets.py --render --noisy --save_images --min_traj_len=10 --num_trajs=1000 --data_dir=path/you/want/to/save/data 

# You can download pretrained rollout
gdown 1TaeB0m9ZW0oiEiP4p9vP3ma7iSUCblXO
gdown 1RtJWxGtHukJUpTTAvRpsiYK3sMcenprl
unzip task24_roll1000_dens250.zip
unzip task24_roll1000_dens1000.zip

# Move data to spirl/rl/data/data_
mkdir -p ./spirl/data/data_medMaze_scripts/task24_roll1000_dens250/
mkdir -p ./spirl/data/data_medMaze_scripts/task24_roll1000_dens1000/
mv .data/maze/scripts/medMaze/task24_roll1000_dens250/* ./spirl/data/data_medMaze_scripts/task24_roll1000_dens250/
mv .data/maze/scripts/medMaze/task24_roll1000_dens1000/* ./spirl/data/data_medMaze_scripts/task24_roll1000_dens1000/

# You can also generate skill policy by command
```
2. Train spirl skill encoder/decoder
```bash
# You need to change the data path in the spirl/configs/skill_prior_learning/maze/hierarchical/conf.py file for the two different domains
python spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical --prefix 24task_1000rollout_dens1000_dim20
python spirl/train.py --path=spirl/configs/skill_prior_learning/maze/hierarchical --prefix 24task_1000rollout_dens250_dim20
```

3. Generate skill for each domain
```bash
# Start extract skill
# You need to change the data path in the skill_emb.py file for the two different domains
cd spirl
mkdir script_data
python skill_emb.py
```

4. Train mapping function
```bash
# Move extracted data to target directory
# Please rerun step 3 and change `dens` in `class spirlModel` to 1000
cp ./spirl/script_data/* ./mapping/data/density250 # or 1000

cd mapping
python train.py
```

5. Generate transfered target skill
```bash
# You can download pretrained rollout
gdown 1Z-W8_MkTr5fu4Zty_e-_8Xsw3P8gXVnq
gdown 1fyeBh7adyHeUkgRwXWPa1eY_IeSagUES
unzip task3_roll1000_dens250.zip
unzip task3_roll1000_dens1000.zip

# Move data to spirl/rl/data/data_
mv .data/maze/scripts/medMaze/task3_roll1000_dens1000/* ./spirl/data/data_medMaze_scripts/

# Start extract skill
cd spirl
mkdir testdata
python skill_emb_target.py

# Move extracted data to target directory
# Please rerun step 3 and change `dens` in `class spirlModel`
cp ./spirl/testdata/* ./mapping/data/density250

cd mapping
python inference.py
```

6. Inference transfered target task and generate video
```bash
# Move data for inference
mv mapping/inference.parquet
python inference.py
```