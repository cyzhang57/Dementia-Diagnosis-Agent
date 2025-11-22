#!/usr/bin/env python3
import os, sys, argparse, logging, time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from config       import load_data, select_col_0, select_col_1, select_col_2, \
                         select_col_3, select_col_4, select_col_5, select_col_6, \
                         select_col_7, select_col_8, select_col_9, select_col_10
from environment  import QuestionnaireEnv
from rainbow      import RainbowAgent
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="policy_trace2.csv",
                    help="输出 csv 文件名")
    ap.add_argument("--agent", default="rainbow_agent.pth",
                    help="已训练模型权重路径")
    return ap.parse_args()

args = parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

X_tr, y_tr, X_te, y_te = load_data()
dynamic_groups = {0:select_col_1, 1:select_col_2, 2:select_col_3,
                  3:select_col_4, 4:select_col_5, 5:select_col_6,
                  6:select_col_7, 7:select_col_8, 8:select_col_9,
                  9:select_col_10}

env = QuestionnaireEnv(X_train=X_tr, y_train=y_tr,
                       X_test =X_te,  y_test =y_te,
                       feature_groups=dynamic_groups,
                       base_features =select_col_0)

agent = RainbowAgent(state_dim=env.observation_space[0],
                     action_dim=len(env.action_space))
agent.load(args.agent)
agent.policy_net.eval()

G          = env.G
n_classes  = len(np.unique(y_tr))

records = []

def _extract_vectors(state: np.ndarray):
    """
    从 env.state (1D array) 切出：
    gvec (0/1)、proba(5维)、time_norm、score_vec(10维或空)
    """
    gvec = state[0:G]
    proba = state[G : G + n_classes]
    time_norm = state[G + n_classes]

    score_start = G + n_classes + 1
    score_end   = score_start + G
    if len(state) >= score_end:
        score_vec = state[score_start:score_end]
    else:
        score_vec = np.full(G, -1.0)
    return gvec, proba, time_norm, score_vec

for idx in range(len(X_te)):
    s  = env.reset(sample_idx=idx)
    done = False
    while not done:
        valid   = env.available_actions()
        action  = agent.select_action(s, valid_actions=valid, training=False)

        gvec, proba, t_norm, score_vec = _extract_vectors(s)

        pred_cls  = int(np.argmax(proba))
        pred_conf = float(np.max(proba))
        rec = OrderedDict(
            sample_idx = idx,
            pred_class = pred_cls,
            pred_conf  = pred_conf,
            time_norm  = t_norm,
            action     = action
        )
        rec.update({f"group_{i}": int(gvec[i]) for i in range(G)})
        rec.update({f"score_Stage{i+1}": float(score_vec[i]) for i in range(G)})
        rec.update({f"prob_{k}": float(proba[k]) for k in range(n_classes)})
        records.append(rec)

        s, r, done, info = env.step(action)

log.info(f"采样完成，共 {len(records):,} 条 state–action 记录")
pd.DataFrame(records).to_csv(args.out, index=False)
log.info(f"已写出 {args.out}")