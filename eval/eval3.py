import os, sys, time, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict, Counter

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from config import load_data, select_col_0, select_col_1, select_col_2, \
                   select_col_3, select_col_4, select_col_5, select_col_6, \
                   select_col_7, select_col_8, select_col_9, select_col_10
from environment import QuestionnaireEnv
from rainbow import RainbowAgent

import matplotlib.pyplot as plt
import seaborn as sns
PALETTE = ["#0072B2","#D55E00","#009E73","#CC79A7","#F0E442"]
sns.set_palette(PALETTE)
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

def _bootstrap_roc(y_true, y_score, n_boot=1000, seed=0):
    """bootstrap ROC → 返回 (mean_tpr, lower, upper, fpr_grid, auc_ci)"""
    rng = np.random.RandomState(seed)
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    fpr_grid = np.linspace(0,1,101)
    tprs = []
    aucs = []
    for _ in range(n_boot):
        pos_idx = rng.choice(np.where(pos)[0], n_pos, replace=True)
        neg_idx = rng.choice(np.where(neg)[0], n_neg, replace=True)
        idx = np.concatenate([pos_idx, neg_idx])
        fpr, tpr, _ = roc_curve(y_true[idx], y_score[idx])
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(0)
    lower, upper = np.percentile(tprs, [2.5,97.5], axis=0)
    auc_ci = np.percentile(aucs, [2.5,97.5])
    return mean_tpr, lower, upper, fpr_grid, auc_ci

def plot_multiclass_roc(y_true, proba_mat, class_names,
                        out_dir="roc_figs", n_boot=1000, dpi=140):
    """
    生成七张图：总类 + 平均 + 单类(含CI、cut-off)
    """
    os.makedirs(out_dir, exist_ok=True)
    C = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(C))

    plt.figure(figsize=(5,5))
    for c, name, color in zip(range(C), class_names, PALETTE):
        fpr, tpr, _ = roc_curve(y_bin[:,c], proba_mat[:,c])
        auc_val     = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=color, label=f"{name}  AUC={auc_val:.3f}")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlabel("1-Specificity"); plt.ylabel("Sensitivity")
    plt.title("ROC – all classes")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"fig-cls.png"), dpi=dpi); plt.close()

    fpr_micro, tpr_micro,_ = roc_curve(y_bin.ravel(), proba_mat.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_bin[:,i], proba_mat[:,i])[0] for i in range(C)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(C):
        mean_tpr += np.interp(all_fpr,
                              roc_curve(y_bin[:,i], proba_mat[:,i])[0],
                              roc_curve(y_bin[:,i], proba_mat[:,i])[1])
    mean_tpr /= C
    auc_macro = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(5,5))
    plt.plot(fpr_micro, tpr_micro, ':', lw=3, color="#444444",
             label=f"micro  AUC={auc_micro:.3f}")
    plt.plot(all_fpr, mean_tpr, '-', lw=3, color="#E69F00",
             label=f"macro  AUC={auc_macro:.3f}")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlabel("1-Specificity"); plt.ylabel("Sensitivity")
    plt.title("ROC – macro & micro")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"fig-avg.png"), dpi=dpi); plt.close()

    for c, name, color in zip(range(C), class_names, PALETTE):
        fpr, tpr, thr = roc_curve(y_bin[:,c], proba_mat[:,c])
        auc_val = auc(fpr, tpr)
        m_tpr, low, up, grid, auc_ci = _bootstrap_roc(y_bin[:,c], proba_mat[:,c],
                                                      n_boot=n_boot)
        j_scores = tpr - fpr
        j_best   = j_scores.argmax()
        fpr_best, tpr_best, thr_best = fpr[j_best], tpr[j_best], thr[j_best]

        plt.figure(figsize=(5,5))
        plt.fill_between(grid, low, up, color=color, alpha=0.15,
                         label="95% CI")
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"AUC={auc_val:.3f} (95%CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f})")
        plt.scatter(fpr_best, tpr_best, s=30, c='black', zorder=10,
                    label=f"cut-off {thr_best:.2f}")
        plt.plot([0,1],[0,1],'k--',lw=1)
        plt.xlabel("1-Specificity"); plt.ylabel("Sensitivity")
        plt.title(f"ROC – {name}")
        plt.legend(frameon=False, loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f"fig-cls-{c}.png"), dpi=dpi)
        plt.close()

    print("全部 ROC 图已输出到文件夹:", out_dir)


def _build_xgb_params(n_classes):
    base = {
        "max_depth": 7, "learning_rate": 0.01,
        "subsample": 0.7, "colsample_bytree": 1.0,
        "gamma": 0.1, "min_child_weight": 1,
        "tree_method": "hist", "device": "cuda"
    }
    if n_classes > 2:
        base.update(objective="multi:softprob", num_class=n_classes)
    else:
        base.update(objective="binary:logistic")
    return base


def evaluate_models(out_dir: str = "viz_output"):
    os.makedirs(out_dir, exist_ok=True)

    X_tr, y_tr, X_te, y_te = load_data()
    n_test = len(X_te)
    n_classes = len(np.unique(y_tr))

    bst_full = xgb.Booster(); bst_full.load_model("xgboost_full_model.model")
    proba_full = bst_full.predict(xgb.DMatrix(X_te))
    pred_full  = np.argmax(proba_full,1) if proba_full.ndim>1 \
                 else (proba_full>0.5).astype(int)
    print(f"[Full-XGB] acc = {accuracy_score(y_te, pred_full):.4f}")

    dynamic_groups = {
        0: select_col_1, 1: select_col_2, 2: select_col_3,
        3: select_col_4, 4: select_col_5, 5: select_col_6,
        6: select_col_7, 7: select_col_8, 8: select_col_9,
        9: select_col_10}
    env = QuestionnaireEnv(X_tr, y_tr, X_te, y_te,
                           feature_groups=dynamic_groups,
                           base_features=select_col_0)
    agent = RainbowAgent(state_dim=env.observation_space[0],
                         action_dim=len(env.action_space))
    agent.load("rainbow_agent_0528_best.pth")

    xgb_params = _build_xgb_params(n_classes)
    sample_records = []
    y_pred_list = []
    feats_cnts, cost_times, times = [], [], []
    correct_cnt = correct_rl = 0

    pbar = tqdm(range(n_test), ncols=90, desc="RL predicting")
    for idx in pbar:
        sample_id = X_te.index[idx]
        selected_feats = list(select_col_0)
        state  = env.reset(sample_idx=idx)
        done, t0 = False, time.time()

        while not done:
            action = agent.select_action(state,
                                         valid_actions=env.available_actions(),
                                         training=False)
            if action in dynamic_groups:
                selected_feats += dynamic_groups[action]
            state, _, done, info = env.step(action)

        rl_corr = int(info["pred"] == info["true"])
        correct_rl += rl_corr
        t_cost = info["time_cost"]; cost_times.append(t_cost)

        selected_feats = list(OrderedDict.fromkeys(selected_feats))
        booster = xgb.train(
            params=xgb_params,
            dtrain=xgb.DMatrix(X_tr[selected_feats], label=y_tr),
            num_boost_round=36
        )
        proba = booster.predict(xgb.DMatrix(X_te.loc[[idx], selected_feats]))
        prob_vec = proba[0] if proba.ndim>1 else np.array([1-proba[0], proba[0]])
        pred2 = int(np.argmax(prob_vec))
        y_pred_list.append(pred2)
        xgb_corr = int(pred2 == y_te.iloc[idx])
        correct_cnt += xgb_corr

        feats_cnts.append(len(selected_feats))
        times.append(time.time()-t0)
        pbar.set_postfix(acc=f"{correct_cnt/(idx+1):.4f}")

        sample_records.append({
            "sample_id":      sample_id,
            "num_feats":      len(selected_feats),
            "feat_time_cost": t_cost,
            "rl_correct":     rl_corr,
            "rl_xgb_correct": xgb_corr,
            "selected_feats": selected_feats[:],
            "prob_vec":       prob_vec.tolist()
        })
    pbar.close()

    print(f"\n[Test-set] RL acc = {correct_rl/n_test:.4f} | "
          f"RL+XGB acc = {correct_cnt/n_test:.4f} | "
          f"#feat = {np.mean(feats_cnts):.2f} | "
          f"采集时长 = {np.mean(cost_times):.2f}")

    df_rec = pd.DataFrame(sample_records)
    df_rec["selected_feats"] = df_rec["selected_feats"] \
        .apply(lambda x: ",".join(x))
    df_rec.to_csv(os.path.join(out_dir,"sample_results.txt"),
                  sep="\t", index=False)
    df_rec.to_excel(os.path.join(out_dir,"sample_results.xlsx"), index=False)

    proba_mat = np.vstack([rec["prob_vec"] for rec in sample_records])
    plot_multiclass_roc(
        y_true      = y_te.values,
        proba_mat   = proba_mat,
        class_names = ["Normal cognition","MCI",
                       "Mild dementia","Moderate dementia","Severe dementia"],
        out_dir     = os.path.join(out_dir,"roc_figs"),
        n_boot      = 500)

    def macro_micro_metrics(y_true, y_pred, proba, n_cls):
        """
        返回 OrderedDict:
          acc_macro / acc_micro
          auc_macro / auc_micro
          sen_macro / sen_micro
          spe_macro / spe_micro
          ratio_macro / ratio_micro
        """
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_cls)).astype(float)
    
        TP = np.diag(cm)
        FN = cm.sum(axis=1) - TP
        FP = cm.sum(axis=0) - TP
        TN = cm.sum() - (TP + FP + FN)
    
        sens_cls = TP / (TP + FN + 1e-12)
        spec_cls = TN / (TN + FP + 1e-12)
        bac_cls  = 0.5 * (sens_cls + spec_cls)
    
        sen_macro = sens_cls.mean()
        spe_macro = spec_cls.mean()
        acc_macro = bac_cls.mean()
        ratio_macro = spe_macro / (sen_macro + 1e-12)
    
        if n_cls == 2:
            auc_val   = roc_auc_score(y_true, proba[:,1])
            auc_macro = auc_micro = auc_val
        else:
            auc_macro = roc_auc_score(y_true, proba,
                                      average="macro", multi_class="ovr")
            auc_micro = roc_auc_score(y_true, proba,
                                      average="micro", multi_class="ovr")
    
        TPg, FPg, FNg, TNg = TP.sum(), FP.sum(), FN.sum(), TN.sum()
        acc_micro = (TPg + TNg) / (TPg + FPg + FNg + TNg)
        sen_micro = TPg / (TPg + FNg + 1e-12)
        spe_micro = TNg / (TNg + FPg + 1e-12)
        ratio_micro = spe_micro / (sen_micro + 1e-12)
    
        return OrderedDict([
            ("acc_macro",  acc_macro),  ("acc_micro",  acc_micro),
            ("auc_macro",  auc_macro),  ("auc_micro",  auc_micro),
            ("sen_macro",  sen_macro),  ("sen_micro",  sen_micro),
            ("spe_macro",  spe_macro),  ("spe_micro",  spe_micro),
            ("ratio_macro",ratio_macro),("ratio_micro",ratio_micro)
        ])
    
    metrics = macro_micro_metrics(
        y_true = y_te.values,
        y_pred = np.array(y_pred_list),
        proba  = np.vstack([rec["prob_vec"] for rec in sample_records]),
        n_cls  = n_classes)
    
    print("\n================  Macro / Micro averaged metrics  ================")
    print("            Accuracy      AUC      Sens      Spec   Spec/Sens")
    print("Macro  :  {acc_macro:.4f}   {auc_macro:.4f}  {sen_macro:.4f}  "
          "{spe_macro:.4f}   {ratio_macro:.4f}".format(**metrics))
    print("Micro  :  {acc_micro:.4f}   {auc_micro:.4f}  {sen_micro:.4f}  "
          "{spe_micro:.4f}   {ratio_micro:.4f}".format(**metrics))
    print("==================================================================")
    
    # 保存到文件
    pd.Series(metrics).to_csv(
        os.path.join(out_dir, "macro_micro_metrics.txt"), sep="\t")
    print("\n所有分析结果已保存到", out_dir)
if __name__ == "__main__":
    evaluate_models(out_dir="viz_output0612-roc02")