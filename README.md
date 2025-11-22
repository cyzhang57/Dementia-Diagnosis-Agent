# Dementia Diagnosis Agent


Code for the paper: **Towards Cost-Effective Cognitive Impairment Diagnosis Systems by Emulating Doctors' Reasoning with Deep Reinforcement Learning**

A new framework for cost-effective cognitive impairment diagnosis system  using Rainbow DQN. 

## Installation

```bash
pip install -r requirements.txt
```

## Data

Sample data is provided in the `data/` directory. The CHARLS survey are accessible online at http://charls.pku.edu.cn

## Usage

### Training

```bash
python train_rainbow.py
```

### Evaluation

```bash
python -m eval.eval2
```

### Policy Distillation

First collect policy traces:

```bash
python utils/collect_policy_trace.py
```

Then distill the policy:

```bash
python utils/distill_policy.py
```

## Citation

If you use this code in your research, please cite:

```

```