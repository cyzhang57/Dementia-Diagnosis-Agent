import os
import time
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import OrderedDict, Counter
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


def evaluate_models(out_dir: str = "viz_output0612_02"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. load data
    X_tr, y_tr, X_te, y_te = load_data()
    n_test = len(X_te)

    # 2. baseline: full-feature XGB
    bst = xgb.Booster()
    bst.load_model("xgboost_full_model.model")
    dtest_full = xgb.DMatrix(X_te)
    proba_full = bst.predict(dtest_full)
    if proba_full.ndim > 1:
        pred_full = np.argmax(proba_full, axis=1)
    else:
        pred_full = (proba_full > 0.5).astype(int)
    acc_full = accuracy_score(y_te, pred_full)
    print(f"=== 全量特征 XGBoost 准确率: {acc_full:.4f}  (n_test={n_test})")

    # 3. 手工阶段累加测试
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
        proba_sub = bst.predict(xgb.DMatrix(X_sub))
        if proba_sub.ndim > 1:
            pred_sub = np.argmax(proba_sub, axis=1)
        else:
            pred_sub = (proba_sub > 0.5).astype(int)
        acc = accuracy_score(y_te, pred_sub)
        stage_feats_num.append(len(cum_feats))
        stage_acc.append(acc)
        print(f"{name:8s} | 累计特征 {len(cum_feats):2d} | accuracy={acc:.4f}")

    dynamic_groups = {
        0: select_col_1, 1: select_col_2, 2: select_col_3,
        3: select_col_4, 4: select_col_5, 5: select_col_6,
        6: select_col_7, 7: select_col_8, 8: select_col_9,
        9: select_col_10
    }
    env = QuestionnaireEnv(X_train=X_tr, y_train=y_tr,
                           X_test=X_te,   y_test=y_te,
                           feature_groups=dynamic_groups,
                           base_features=select_col_0)
    agent = RainbowAgent(state_dim=env.observation_space[0],
                         action_dim=len(env.action_space))
    agent.load("rainbow_agentf.pth")
    sample_records = []

    correct_cnt, feats_cnts, times, acc_curve = 0, [], [], []
    correct_rl = 0
    cost_times = []
    n_classes = len(np.unique(y_tr))
    xgb_params = _build_xgb_params(n_classes)

    pbar = tqdm(range(n_test), ncols=90, desc="RL predicting")
    for idx in pbar:
        sample_id = X_te.index[idx]
        selected_feats = list(select_col_0)  # 一定会用的 base 特征
        state = env.reset(sample_idx=idx)
        done = False
        t0 = time.time()
        while not done:
            valid = env.available_actions()
            action = agent.select_action(state, valid_actions=valid, training=False)
            if action in dynamic_groups:
                selected_feats += dynamic_groups[action]
            state, r, done, info = env.step(action)
        rl_corr = int(info.get("pred",-1)==info.get("true",-2))
        correct_rl += rl_corr
        t_cost = info.get("time_cost", np.nan)
        cost_times.append(t_cost)
        selected_feats = list(OrderedDict.fromkeys(selected_feats))
        booster = xgb.train(
            params=xgb_params,
            dtrain=xgb.DMatrix(X_tr[selected_feats], label=y_tr),
            num_boost_round=36
        )
        proba = booster.predict(xgb.DMatrix(X_te.loc[[idx], selected_feats]))
        if proba.ndim>1:
            pred2 = int(np.argmax(proba,axis=1)[0])
        else:
            pred2 = int(proba[0]>0.5)
        xgb_corr = int(pred2==y_te.iloc[idx])
        correct_cnt += xgb_corr

        feats_cnts.append(len(selected_feats))
        times.append(time.time()-t0)
        acc_curve.append(correct_cnt/(idx+1))
        pbar.set_postfix(acc=f"{acc_curve[-1]:.4f}", acc_rl=f"{correct_rl/(idx+1):.4f}")

        sample_records.append({
            "sample_id":      sample_id,
            "num_feats":      len(selected_feats),
            "feat_time_cost": t_cost,
            "rl_correct":     rl_corr,
            "rl_xgb_correct": xgb_corr,
            "selected_feats": selected_feats[:]  # 深拷贝一份
        })
    pbar.close()

    # 最终汇总打印
    print(f"\nRL Agent 共测 {n_test} 样本：")
    print(f"  - RL 自身准确率    = {correct_rl/n_test:.4f}")
    print(f"  - RL+XGB 准确率   = {correct_cnt/n_test:.4f}")
    print(f"  - 平均特征数      = {np.mean(feats_cnts):.2f}")
    print(f"  - 平均收集时间    = {np.mean(cost_times):.2f}")
    print(f"  - 平均推理时长(ms)= {np.mean(times)*1000:.1f}")

    df_rec = pd.DataFrame(sample_records)
    df_rec["selected_feats"] = df_rec["selected_feats"] \
                                  .apply(lambda lst: ",".join(lst))
    df_rec.to_csv(os.path.join(out_dir, "sample_results.txt"),
                  sep="\t", index=False)
    df_rec.to_excel(os.path.join(out_dir, "sample_results.xlsx"),
                    index=False)

    all_feats = [feat for rec in sample_records for feat in rec["selected_feats"]]
    freq = Counter(all_feats)
    df_freq = pd.DataFrame.from_records(
        sorted(freq.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "count"]
    )
    df_freq.to_csv(os.path.join(out_dir, "feature_usage_freq.txt"),
                   sep="\t", index=False)
    top20 = df_freq.head(20)
    plt.figure(figsize=(8,4))
    plt.barh(top20["feature"], top20["count"])
    plt.gca().invert_yaxis()
    plt.title("Top20 特征被选用频次")
    plt.xlabel("样本数")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top20_feature_freq.png"), dpi=140)
    plt.close()

    # 2) 每个特征选用时对应的平均准确率
    feat_acc = []
    for feat in df_freq["feature"]:
        mask = df_rec["selected_feats"].str.contains(fr"\b{feat}\b")
        acc_mean = df_rec.loc[mask, "rl_xgb_correct"].mean()
        feat_acc.append((feat, acc_mean, mask.sum()))
    df_facc = pd.DataFrame(feat_acc, columns=["feature","avg_acc","count"]) \
                   .sort_values("avg_acc", ascending=False)
    df_facc.to_csv(os.path.join(out_dir, "feature_vs_acc.txt"),
                   sep="\t", index=False)
    # 可视化：准确率 top10
    top_acc = df_facc.head(10)
    plt.figure(figsize=(6,4))
    plt.barh(top_acc["feature"], top_acc["avg_acc"])
    plt.gca().invert_yaxis()
    plt.title("选用后 RL+XGB 平均准确率 Top10 特征")
    plt.xlabel("平均准确率")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top10_feat_vs_acc.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(6,4))
    df_rec.boxplot(column="num_feats", by="rl_xgb_correct")
    plt.title("特征数分布：按 RL+XGB 是否正确")
    plt.suptitle("")
    plt.xlabel("RL+XGB 正确(1)/错误(0)")
    plt.ylabel("使用特征数")
    plt.savefig(os.path.join(out_dir, "box_num_feats_by_corr.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.scatter(df_rec["num_feats"], df_rec["feat_time_cost"], 
                c=df_rec["rl_xgb_correct"], cmap="bwr", alpha=0.6)
    plt.colorbar(label="RL+XGB Correct")
    plt.xlabel("使用特征数")
    plt.ylabel("特征收集时间")
    plt.title("时间成本 vs 特征数 (红=正确 蓝=错误)")
    z = np.polyfit(df_rec["num_feats"], df_rec["feat_time_cost"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df_rec["num_feats"].min(), df_rec["num_feats"].max(), 100)
    plt.plot(xs, p(xs), "k--", label="fitting")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "time_vs_feats.png"), dpi=140)
    plt.close()

    grp_nf = df_rec.groupby("num_feats").agg(
        sample_count=("sample_id","count"),
        avg_acc=("rl_xgb_correct","mean"),
        avg_time=("feat_time_cost","mean")
    ).reset_index()
    grp_nf.to_csv(os.path.join(out_dir, "by_num_feats.txt"),
                  sep="\t", index=False)
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax2 = ax1.twinx()
    ax1.plot(grp_nf["num_feats"], grp_nf["avg_acc"], 'g-o', label="AvgAcc")
    ax2.plot(grp_nf["num_feats"], grp_nf["avg_time"], 'b--s', label="AvgTime")
    ax1.set_xlabel("使用特征数")
    ax1.set_ylabel("平均准确率", color='g')
    ax2.set_ylabel("平均时间成本", color='b')
    plt.title("特征数 vs 平均准确率 & 平均时间成本")
    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.savefig(os.path.join(out_dir, "num_feats_acc_time.png"), dpi=140)
    plt.close()

    stage_stats = []
    for sid, feats in dynamic_groups.items():
        mask = df_rec["selected_feats"].apply(
            lambda s: any(f in s.split(",") for f in feats)
        )
        stage_stats.append({
            "stage": sid+1,
            "sample_count": mask.sum(),
            "avg_acc": df_rec.loc[mask,"rl_xgb_correct"].mean()
        })
    df_stage = pd.DataFrame(stage_stats)
    df_stage.to_csv(os.path.join(out_dir, "stage_usage_acc.txt"),
                    sep="\t", index=False)
    plt.figure(figsize=(6,4))
    plt.bar(df_stage["stage"].astype(str), df_stage["sample_count"])
    plt.plot(df_stage["stage"].astype(str), df_stage["avg_acc"]*df_stage["sample_count"],
             "r-o", label="正确数(右轴)")
    plt.xlabel("阶段(动态选择的stage编号)")
    plt.ylabel("样本数")
    plt.twinx().set_ylabel("阶段上正确样本数")
    plt.title("阶段(组) 使用频次 & 正确数量")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage_usage_acc.png"), dpi=140)
    plt.close()

    print("\n所有分析结果已保存到", out_dir)


if __name__ == "__main__":
    evaluate_models()