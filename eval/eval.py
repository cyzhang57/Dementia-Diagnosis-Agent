import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import OrderedDict
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import load_data, select_col_0, select_col_1, select_col_2, \
                   select_col_3, select_col_4, select_col_5, select_col_6, \
                   select_col_7, select_col_8, select_col_9, select_col_10
from environment import QuestionnaireEnv
from rainbow import RainbowAgent


def _build_xgb_params(n_classes):
    base = {
        'max_depth': 7,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'gamma': 0.1,
        'min_child_weight': 1,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    if n_classes > 2:
        base.update(dict(objective='multi:softprob', num_class=n_classes))
    else:
        base.update(dict(objective='binary:logistic'))
    return base


def evaluate_models(out_dir: str = "viz_output2"):
    os.makedirs(out_dir, exist_ok=True)

    X_tr, y_tr, X_te, y_te = load_data()
    n_test = len(X_te)

    bst = xgb.Booster()
    bst.load_model("xgboost_full_model.model")

    dtest_full = xgb.DMatrix(X_te)
    proba_full = bst.predict(dtest_full)
    if proba_full.ndim > 1:
        pred_full = np.argmax(proba_full, axis=1)
    else:
        pred_full = (proba_full > 0.5).astype(int)
    acc_full = accuracy_score(y_te, pred_full)
    print("=== 全量特征 XGBoost 准确率: {:.4f}  (n_test={})".format(acc_full, n_test))
    manual_stages = OrderedDict([
        ("stage1",  select_col_1),
        ("stage2",  select_col_2),
        ("stage3",  select_col_3),
        ("stage4",  select_col_4),
        ("stage5",  select_col_5),
        ("stage6",  select_col_6),
        ("stage7",  select_col_7),
        ("stage8",  select_col_8),
        ("stage9",  select_col_9),
        ("stage10", select_col_10)
    ])
    print("\n=== 分阶段累加测试 ===")
    cum_feats, stage_feats_num, stage_acc = [], [], []
    for name, grp in manual_stages.items():
        cum_feats += grp
        cum_feats = list(OrderedDict.fromkeys(cum_feats))
        X_sub = pd.DataFrame(np.nan, columns=X_te.columns, index=X_te.index)
        X_sub[cum_feats] = X_te[cum_feats]

        dtest_sub = xgb.DMatrix(X_sub)
        proba_sub = bst.predict(dtest_sub)
        if proba_sub.ndim > 1:
            pred_sub = np.argmax(proba_sub, axis=1)
        else:
            pred_sub = (proba_sub > 0.5).astype(int)
        acc = accuracy_score(y_te, pred_sub)
        stage_feats_num.append(len(cum_feats))
        stage_acc.append(acc)
        print(f"{name:8s} | 累计特征 {len(cum_feats):2d} | accuracy={acc:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(stage_feats_num, stage_acc, marker='o')
    plt.xlabel("累积特征数")
    plt.ylabel("Accuracy")
    plt.title("手工阶段累加 – 准确率曲线")
    plt.grid(alpha=.3)
    plt.savefig(os.path.join(out_dir, "manual_stage_acc.png"), dpi=140)
    plt.close()

    dynamic_groups = {
        0: select_col_1,
        1: select_col_2,
        2: select_col_3,
        3: select_col_4,
        4: select_col_5,
        5: select_col_6,
        6: select_col_7,
        7: select_col_8,
        8: select_col_9,
        9: select_col_10
    }
    env = QuestionnaireEnv(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        feature_groups=dynamic_groups,
        base_features=select_col_0
    )
    agent = RainbowAgent(state_dim=env.observation_space[0],
                         action_dim=len(env.action_space))
    agent.load("rainbow_agent0509.pth")

    sample_records = []

    correct_cnt, feats_cnts, times, acc_curve, acc_curve2 = 0, [], [], [], []
    correct = 0
    cost_times = []
    n_classes = len(np.unique(y_tr))
    xgb_params = _build_xgb_params(n_classes)

    print("\n=== RL Agent + per-sample XGBoost ===")
    pbar = tqdm(range(n_test), ncols=90, desc="RL predicting")

    for idx in pbar:
        sample_id = X_te.index[idx]

        selected_feats = list(select_col_0)
        state = env.reset(sample_idx=idx)
        done = False
        start_time = time.time()

        while not done:
            valid = env.available_actions()
            action = agent.select_action(state, valid_actions=valid, training=False)
            if action in dynamic_groups:
                selected_feats += dynamic_groups[action]
            state, r, done, info = env.step(action)

        rl_correct = int(info.get("pred", -1) == info.get("true", -2))
        correct += rl_correct

        time_cost = info.get("time_cost", np.nan)
        cost_times.append(time_cost)

        selected_feats = list(OrderedDict.fromkeys(selected_feats))
        Xtr_sub = X_tr[selected_feats]
        booster = xgb.train(
            params=xgb_params,
            dtrain=xgb.DMatrix(Xtr_sub, label=y_tr),
            num_boost_round=36
        )
        proba = booster.predict(xgb.DMatrix(X_te.loc[[idx], selected_feats]))
        if proba.ndim > 1:
            pred_xgb = int(np.argmax(proba, axis=1)[0])
        else:
            pred_xgb = int(proba[0] > .5)

        xgb_correct = int(pred_xgb == y_te.iloc[idx])
        correct_cnt += xgb_correct

        feats_cnts.append(len(selected_feats))
        times.append(time.time() - start_time)
        acc_curve.append(correct_cnt / (idx + 1))
        acc_curve2.append(correct / (idx + 1))

        pbar.set_postfix(acc=f"{acc_curve[-1]:.4f}", acc_rl=f"{acc_curve2[-1]:.4f}")

        sample_records.append({
            "sample_id":        sample_id,
            "num_feats":        len(selected_feats),
            "feat_time_cost":   time_cost,
            "rl_correct":       rl_correct,
            "rl_xgb_correct":   xgb_correct
        })

    pbar.close()

    acc_rl = correct / n_test
    print(f"RL Agent on {n_test} 样本上累计 accuracy = {acc_rl:.4f}")
    print(f"\n在 {n_test} 个测试样本上：")
    print(f"  ‑ RL+XGB准确率          = {correct_cnt/n_test:.4f}")
    print(f"  ‑ 平均使用特征数        = {np.mean(feats_cnts):.2f}")
    print(f"  ‑ 特征的平均时间成本    = {np.mean(cost_times):.1f}")
    print(f"  ‑ 平均推理耗时(ms)      = {np.mean(times)*1000:.1f}")

    plt.figure(figsize=(5, 4))
    plt.hist(feats_cnts, bins=range(min(feats_cnts), max(feats_cnts)+2),
             alpha=0.7, edgecolor='k')
    plt.xlabel("样本使用的特征数")
    plt.ylabel("样本数")
    plt.title("RL 模式 – 特征数分布")
    plt.grid(alpha=.25)
    plt.savefig(os.path.join(out_dir, "rl_feat_hist.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, n_test+1), acc_curve)
    plt.xlabel("已评估样本数")
    plt.ylabel("累计准确率")
    plt.title("RL 模式 – 进度曲线")
    plt.ylim(0, 1.01)
    plt.grid(alpha=.3)
    plt.savefig(os.path.join(out_dir, "rl_acc_curve.png"), dpi=140)
    plt.close()

    df_rec = pd.DataFrame(sample_records)
    df_rec.to_csv(os.path.join(out_dir, "sample_results.txt"),
                  sep="\t", index=False, encoding="utf-8")
    df_rec.to_excel(os.path.join(out_dir, "sample_results.xlsx"),
                    index=False)

    acc_by_feats = df_rec.groupby("num_feats")["rl_xgb_correct"] \
                         .agg(sample_count="count", avg_acc="mean") \
                         .reset_index()
    print("\n按特征数统计样本量和平均 RL+XGB 准确率：")
    print(acc_by_feats)

    plt.figure(figsize=(6, 4))
    plt.plot(acc_by_feats["num_feats"], acc_by_feats["avg_acc"], marker='o')
    plt.xlabel("使用特征数")
    plt.ylabel("平均 RL+XGB 准确率")
    plt.title("特征数 vs 平均准确率")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "featnum_vs_acc.png"), dpi=140)
    plt.close()


if __name__ == "__main__":
    evaluate_models()