from config import load_data, select_col_0, select_col_1, select_col_2, select_col_3, select_col_4
from config import *
from environment import QuestionnaireEnv
from rainbow import RainbowAgent 
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w=20):
    """简单滑动平均，窗口 w"""
    if len(x) < w:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[w:] - cumsum[:-w]) / w


def main_train():
    X_tr, y_tr, X_te, y_te = load_data()

    base_feats = select_col_0
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

    env = QuestionnaireEnv(X_tr, y_tr, X_te, y_te,
                           feature_groups=dynamic_groups,
                           base_features=base_feats)
    agent = RainbowAgent(
        state_dim=env.observation_space[0],
        action_dim=len(env.action_space))

    episodes = 200
    all_losses = []
    ep_rewards = []
    for ep in range(1, episodes+1):
        s = env.reset(); done=False; total_r=0
        while not done:
            valid = env.available_actions() 
            a = agent.select_action(s, valid_actions=valid, training=True)
            ns, r, done, info = env.step(a)
            agent.append(s, a, r, ns, float(done))
            loss = agent.train_step()
            if loss is not None: all_losses.append(loss)
            s = ns; total_r += r

        ep_rewards.append(total_r)
        if ep % 10 == 0:
            print(f"[EP {ep:04d}] R={total_r:.3f}  buffer={agent.memory.size}")

    agent.save("rainbow_agent.pth")

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(all_losses, color='tab:blue', alpha=0.6, label='DQN loss')
    plt.xlabel('Training steps')
    plt.ylabel('MSE Loss')
    plt.title('Rainbow Loss Curve')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(ep_rewards, color='tab:orange', alpha=0.6, label='Episode reward')
    ma = moving_average(ep_rewards, w=20)
    plt.plot(range(len(ma)), ma, color='tab:red', linewidth=2, label='MA(20)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Reward Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


if __name__ == "__main__":
    main_train()
