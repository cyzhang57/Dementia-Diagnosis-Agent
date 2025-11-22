import numpy as np
import pandas as pd
import os

time_dict_1 = {
    'time_MMSE': np.float32(507.46402),
    'time_TICS': np.float32(56.99367),
    'time_WR': np.float32(227.81012),
    'time_AN': np.float32(186.25148),
    'time_SC': np.float32(119.750984),
    'time_BC': np.float32(84.80715),
    'time_CSID': np.float32(135.12366),
    'time_DR': np.float32(48.519096),
    'time_LM': np.float32(169.52255),
    'time_WRE': np.float32(169.97237),
    'time_CP': np.float32(183.15634),
    'time_SDMT': np.float32(213.85132),
    'time_LMR': np.float32(57.17439),
    'time_NS': np.float32(231.32835),
    'time_TMT': np.float32(340.24023),
    'time_CESD': np.float32(247.0013),
    'time_IQCODE': np.float32(468.061),
    'time_BLESSED': np.float32(42.91882),
    'time_CSI_D': np.float32(69.368095)
}

score_time = {
    'score_BLESSED1': ['time_BLESSED', 1],
    'score_CSI_D2': ['time_CSI_D', 1],
    'score_TICS1': ['time_TICS', 1],
    'score_CSID': ['time_CSID', 1],
    'score_IQCODE1': ['time_IQCODE', 1],
    'score_SC': ['time_SC', 1],
    'score_BC1': ['time_BC', 1],
    'score_SDMT': ['time_SDMT', 1],
    'score_TMT1': ['time_TMT', 2],
    'score_TMT2': ['time_TMT', 2],
    'score_WR1': ['time_WR', 3],
    'score_WR2': ['time_WR', 3],
    'score_WR3': ['time_WR', 3],
    'score_DR': ['time_DR', 1],
    'score_AN': ['time_AN', 1],
    'score_LM1': ['time_LM', 1],
    'score_LM2': ['time_LM', 1],
    'score_WRE': ['time_WRE', 1],
    'score_LMR1': ['time_LMR', 2],
    'score_LMR2': ['time_LMR', 2],
    'score_CP': ['time_CP', 2],
    'score_CP2': ['time_CP', 2],
    'score_NS': ['time_NS', 1]
}

select_col_0 = ['Age','RGender','Edu','Residence','BB000_W3_2']
select_col_1 = ['score_CSI_D2','score_CSID']
select_col_2 = ['score_BLESSED1']
select_col_3 = ['score_TICS1']
select_col_4 = ['score_IQCODE1']
select_col_5 = ['score_SC','score_BC1']
select_col_6 = ['score_SDMT','score_TMT1','score_TMT2']
select_col_7 = ['score_AN']
select_col_8 = ['score_WR1','score_WR2','score_WR3','score_DR','score_WRE']
select_col_9 = ['score_LM1','score_LM2','score_LMR1','score_LMR2']
select_col_10 = ['score_CP','score_CP2']


def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, 'train_data.csv')
    test_path = os.path.join(current_dir, 'test_data.csv')
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_x = train_data.drop('label', axis=1)
    train_y = train_data['label']
    test_x = test_data.drop('label', axis=1)
    test_y = test_data['label']
    train_x = train_x.iloc[:, :27]
    test_x = test_x.iloc[:, :27]
    return train_x, train_y, test_x, test_y
