#!/usr/bin/env python3
import os, sys, argparse, logging, textwrap
from pathlib import Path
from matplotlib.text import Text
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import tree as sktree

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Distill RL policy → decision tree + single-sample path")
    p.add_argument("--csv",      default="policy_trace2.csv")
    p.add_argument("--out_txt",  default="policy_tree3.txt")
    p.add_argument("--out_fig",  default="policy_tree3.pdf")
    p.add_argument("--max_depth",type=int, default=5)
    p.add_argument("--min_leaf", type=int, default=150)
    p.add_argument("--seed",     type=int, default=3407)
    p.add_argument("--purity",   type=float, default=0.9,
                   help="子节点若同父且占比≥阈值即剪")
    return p.parse_args()

args = parse_args()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

group_name_map = {
    0:"CSID", 1:"BLESSED", 2:"TICS", 3:"IQCODE", 4:"SC&BC",
    5:"SDMT&TMT", 6:"AN", 7:"WR&DR", 8:"LM&LMR", 9:"CP"
}
action_name_map = {i: f"Recommend test: {v}"
                   for i, v in group_name_map.items()}
action_name_map[10] = "STOP and give diagnosis"

label_name_map  = {0:"Normal cognition", 1:"MCI",
                   2:"Mild dementia",   3:"Moderate dementia",
                   4:"Severe dementia"}

if not os.path.isfile(args.csv):
    log.error(f"{args.csv} not found"); sys.exit(1)
df = pd.read_csv(args.csv)
log.info(f"data shape = {df.shape}")

need_group = [f"group_{i}" for i in range(10)]
need_score = [f"score_Stage{i+1}" for i in range(10)]
need_prob  = [f"prob_{k}" for k in range(5)]
need_other = ["pred_class", "pred_conf", "action"]
for c in need_group + need_score + need_prob + need_other:
    if c not in df.columns:
        df[c] = -1 if c not in need_prob else 0.0

df.rename(columns={f"group_{i}":f"has_{group_name_map[i]}" for i in range(10)},
          inplace=True)

feat_cols = ([c for c in df.columns if c.startswith("has_")] +
             need_score + need_prob + ["pred_class","pred_conf"])
X = pd.DataFrame(SimpleImputer(strategy="constant", fill_value=-1)
                 .fit_transform(df[feat_cols]), columns=feat_cols)
y = df["action"]

# ───────────────── 训练 ─────────────────
clf = DecisionTreeClassifier(max_depth=args.max_depth,
                             min_samples_leaf=args.min_leaf,
                             criterion="entropy",
                             random_state=args.seed)
clf.fit(X, y)
log.info("depth(before)=%d  leaves=%d", clf.get_depth(), clf.get_n_leaves())

def safe_prune(tree_: sktree._tree.Tree,
               stop_cls: int = 10,
               purity: float = 0.95):
    """
    安全剪枝：仅在"子节点是叶、且与父同类、且纯度≥purity"时剪掉子
    """
    L, R, VAL = tree_.children_left, tree_.children_right, tree_.value
    n_nodes   = VAL.shape[0]

    def majority(nid):
        """返回(类别, 纯度)"""
        v = VAL[nid, 0]
        total = v.sum()
        if total == 0:
            return -1, 0.0            # 理论不会出现
        cls = int(v.argmax())
        pur = v.max() / total
        return cls, pur

    order = []
    stack = [0]
    while stack:
        nid = stack.pop()
        order.append(nid)
        if L[nid] != sktree._tree.TREE_LEAF:
            stack.append(L[nid])
        if R[nid] != sktree._tree.TREE_LEAF:
            stack.append(R[nid])
    order = order[::-1]

    for nid in order:
        cls_p, _ = majority(nid)
        if L[nid] == sktree._tree.TREE_LEAF and R[nid] == sktree._tree.TREE_LEAF:
            continue

        cls_l, pur_l = majority(L[nid])
        cls_r, pur_r = majority(R[nid])

        if L[nid] != sktree._tree.TREE_LEAF and cls_l == cls_p and pur_l >= purity:
            L[nid] = sktree._tree.TREE_LEAF
        if R[nid] != sktree._tree.TREE_LEAF and cls_r == cls_p and pur_r >= purity:
            R[nid] = sktree._tree.TREE_LEAF

        if cls_p == stop_cls and \
           L[nid] == sktree._tree.TREE_LEAF and R[nid] == sktree._tree.TREE_LEAF:
            if cls_l == cls_r == stop_cls:
                L[nid] = R[nid] = sktree._tree.TREE_LEAF

safe_prune(clf.tree_, stop_cls=10, purity=0.95)

log.info("depth(after)=%d  leaves=%d", clf.get_depth(), clf.get_n_leaves())

log.info("Fidelity = %.4f", accuracy_score(y, clf.predict(X)))

raw = export_text(clf, feature_names=feat_cols,
                  show_weights=True, decimals=3)
def rep(l):
    if "class:" in l:
        k=int(l.split("class:")[1].split()[0])
        l=l.replace("class:","action:").replace(str(k),
             f"{k} ({action_name_map[k]})")
    return l
txt="\n".join(rep(i) for i in raw.splitlines())
Path(args.out_txt).write_text(txt,encoding="utf-8")
log.info(f"rules saved to {args.out_txt}")

def _draw_tree(fig_path):
    plt.figure(figsize=(26,12))
    plot_tree(clf,
        feature_names=feat_cols,
        class_names=[f"{k}:{action_name_map[k]}" for k in sorted(y.unique())],
        filled=True, rounded=True, proportion=True,
        impurity=False, fontsize=8)
    ax=plt.gca()
    for o in ax.get_children():
        if isinstance(o,Text) and "class =" in o.get_text():
            o.set_text(o.get_text().replace("class =","action ="))
    plt.tight_layout(); plt.savefig(fig_path,dpi=300); plt.close()
_draw_tree(args.out_fig)
log.info(f"full tree saved to {args.out_fig}")

def _node_path(sample_row: pd.Series):
    """返回该样本经过的节点 id list（含叶）"""
    inds = clf.decision_path(sample_row.values.reshape(1,-1)).indices
    return list(inds)

def print_decision_path(sample_idx:int):
    """打印 sample_idx 路径 & 诊断"""
    if sample_idx>=len(df): 
        print("index out of range"); return
    row = X.iloc[[sample_idx]]
    nid_path = _node_path(row.iloc[0])
    leaf  = nid_path[-1]
    print("\n=== decision path for sample",sample_idx,"===")
    diag = label_name_map[int(df.loc[sample_idx,'pred_class'])]
    print(diag)
    for nid in nid_path:
        fid = clf.tree_.feature[nid]
        thr = clf.tree_.threshold[nid]
        if fid!=sktree._tree.TREE_UNDEFINED:
            fname=feat_cols[fid]
            fval = row.iloc[0,fid]
            comp = "<=" if fval<=thr else ">"
            print(f"node {nid}: {fname} ({fval:.3f}) {comp} {thr:.3f}")
        else:
            act = clf.predict(row)[0]
            print(f"leaf {nid}: action = {act} ({action_name_map[act]})")
            print(diag)
            if act==10:
                diag = label_name_map[int(df.loc[sample_idx,'pred_class'])]
                print("diagnosis =",diag)
                print(diag)
    print("="*40)

def plot_decision_path(sample_idx:int,out_fig:str="sample_path.pdf"):
    """重新画树并把 sample_idx 经过的节点框高亮为红色"""
    print(len(df))
    if sample_idx>=len(df):
        print("index out of range"); return
    nid_path = _node_path(X.iloc[sample_idx])
    fig = plt.figure(figsize=(26,12))
    annots = plot_tree(clf,
        feature_names=feat_cols,
        class_names=[f"{k}:{action_name_map[k]}" for k in sorted(y.unique())],
        filled=True, rounded=True, proportion=True,
        impurity=False, fontsize=8)
    # annots 顺序 = node_id 顺序
    for nid in nid_path:
        ann = annots[nid]
        ann.get_bbox_patch().set_edgecolor("red")
        ann.get_bbox_patch().set_linewidth(3)
    # 替换 class→action
    ax=plt.gca()
    for o in ax.get_children():
        if isinstance(o,Text):
            s=o.get_text()
            if "class =" in s: o.set_text(s.replace("class =","action ="))
            if "action = 10" in s:
                # 附加诊断众数
                leaf_id = annots.index(o)  # node_id
                idxs = np.where(clf.apply(X)==leaf_id)[0]
                if len(idxs):
                    mode_label = df.loc[idxs,'pred_class'].mode().iloc[0]
                    o.set_text(s.replace("action =", "action =")+
                               f"\n diagnosis = {label_name_map[int(mode_label)]}")
    plt.title(f"Decision path for sample {sample_idx}",fontsize=16)
    plt.tight_layout(); plt.savefig(out_fig,dpi=300); plt.close()
    log.info(f"path figure saved to {out_fig}")

print_decision_path(30)
plot_decision_path(30, "sample5_path.pdf")