import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from config import time_dict_1, score_time


def entropy(p):
    return -np.sum(p * np.log(p + 1e-10))


def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))


class QuestionnaireEnv:
    def __init__(self, X_train, y_train, X_test, y_test,
                 feature_groups, base_features,
                 R_C=3.0, R_W=-3.0,
                 lambda_time=0.0034, 
                 lambda_time_end=0.001,
                 lambda_info=0,
                 C_repeat=2.0,
                 B=2.0, n0=8, kappa=2.0, T_max=1500.0):
        self.X_train, self.y_train = X_train, y_train
        self.X_test,  self.y_test  = X_test,  y_test
        self.base_feats = list(base_features)
        self.base_dim   = len(self.base_feats) 

        self.feature_groups = feature_groups
        self.G = len(feature_groups)
        self.action_space = list(range(self.G + 1))
        self.observation_space = (self.G + len(np.unique(y_train)) + 1 +  self.base_dim,)

        self._bf_mean = X_train[self.base_feats].mean().values
        self._bf_std  = X_train[self.base_feats].std(ddof=0).replace(0,1).values

        self.R_C, self.R_W = R_C, R_W
        self.lambda_time, self.lambda_time_end = lambda_time, lambda_time_end
        self.lambda_info = lambda_info
        self.C_repeat = C_repeat
        self.B, self.n0, self.kappa = B, n0, kappa
        self.T_max = T_max

        self.model = self._pretrain_full_model()
        self.reset()

    def _pretrain_full_model(self):
        params = {
            'max_depth': 7, 'learning_rate': 0.1,
            'subsample': 0.7, 'colsample_bytree': 1.0,
            'gamma': 0.1, 'min_child_weight': 1,
            'objective': 'multi:softprob',
            'max_leaves': 19,'nthread': 4,
            'eval_metric': 'mlogloss','seed': 3407,
            'num_class': len(np.unique(self.y_train)),
            'tree_method': 'hist', 'device': 'cuda'
        }
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        model = xgb.train(params, dtrain, num_boost_round=36)
        model.save_model('xgboost_full_model.model')
        return model 

    def reset(self, sample_idx=None):
        if sample_idx is None:
            self.curr_idx = np.random.randint(len(self.X_test))
        else:
            self.curr_idx = sample_idx
        self.selected = set()
        self.current_features = list(self.base_feats)
        self.time_cost = 0.0
        self.prev_proba = self._compute_proba(self.current_features)
        return self._get_state()

    def _compute_proba(self, feats):
        idx = self.curr_idx
        df = pd.DataFrame(np.nan, columns=self.X_train.columns, index=[0])
        for f in feats:
            df.at[0, f] = self.X_test.iloc[idx][f]
        proba = self.model.predict(xgb.DMatrix(df))
        return proba[0]
    
    def available_actions(self):
        """
        返回当前状态下允许的动作列表：
         还没被选过的组编号
         STOP 动作(编号 self.G)
        """
        valid = [a for a in range(self.G) if a not in self.selected]
        valid.append(self.G)
        return valid
    
    def _get_state(self):
        gvec = np.zeros(self.G, dtype=float)
        for i in self.selected:
            gvec[i] = 1.0
        pvec = self.prev_proba
        tnorm = np.array([self.time_cost / self.T_max], dtype=float)

        raw_base = self.X_test.iloc[self.curr_idx][self.base_feats].values.astype(float)
        base_norm = (raw_base - self._bf_mean) / (self._bf_std + 1e-8)
        return np.hstack([gvec, pvec, tnorm, base_norm])

    def step(self, action):
        assert 0 <= action <= self.G
        if action == self.G:
            idx = self.curr_idx
            df = pd.DataFrame(np.nan, columns=self.X_train.columns, index=[0])
            for f in self.current_features:
                df.at[0, f] = self.X_test.iloc[idx][f]
            proba = self.model.predict(xgb.DMatrix(df))[0]
            pred = int(proba.argmax())
            true = int(self.y_test.iloc[idx])
            acc_flag = 1 if pred == true else 0

            n_feats = len(self.current_features)
            base = acc_flag * self.R_C + (1 - acc_flag) * self.R_W
            bonus_n = self.B * expit((n_feats - self.n0) / self.kappa)
            penalty_t = self.lambda_time_end * self.time_cost
            reward = base + bonus_n - penalty_t
            next_state = self._get_state()
            info = {
                'sample_idx': idx,
                'pred': pred,
                'true': true,
                'n_feats': n_feats,
                'time_cost': self.time_cost
            }
            return next_state, reward, True, info

        if action in self.selected:
            return self._get_state(), -self.C_repeat, False, {}

        feats = self.feature_groups[action]
        delta_t = 0.0
        for f in feats:
            key, div = score_time[f]
            delta_t += time_dict_1[key] / div
        self.time_cost += delta_t

        self.selected.add(action)
        self.current_features.extend(feats)

        new_proba = self._compute_proba(self.current_features)
        info_gain3 = new_proba.max() - self.prev_proba.max()
 
        old_ent = entropy(self.prev_proba)
        new_ent = entropy(new_proba)
        info_gain = old_ent - new_ent
        info_gain2 = kl_divergence(new_proba, self.prev_proba)

        reward = self.lambda_info * (info_gain+ info_gain2+info_gain3) - self.lambda_time * delta_t
        self.prev_proba = new_proba
        gain_all = info_gain*0.5+ info_gain2 + info_gain3*2
        return self._get_state(), reward, False, {}
