# RL_2023_Fall
Final Project for RL 2023 Fall

## Mapping

Map feature vectors from domain 1 to 2.

### Environment

```bash
pip install -r requirements.txt
cd mapping
```

### Run

```bash
python3 train.py
```

### Developer Guide

1. `dataset.py`: Dataset for train mapping networks
2. `network.py`: Mapping networks
3. `tools.py`: Additional tools
4. `train.py`: Main file for training
5. `trainer.py`: Helper class for training
6. Files in `config/`: Config file for each experiment. Note thate the file should be `.yaml` instead of `.yml`