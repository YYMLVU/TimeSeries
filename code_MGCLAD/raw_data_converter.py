import numpy as np
import pandas as pd
import pickle as pkl
import os


def psm():
    # preprocess for SWaT. SWaT.A2_Dec2015, version 0
    df = pd.read_csv('psm/test_label.csv')
    y = df['label'].to_numpy()
    labels = np.array(y)
    print('labels:', labels.shape)
    #assert len(labels) == 449919
    pkl.dump(labels, open('PSM_test_label.pkl', 'wb'))
    print('PSM_test_label saved')

    df = pd.read_csv('psm/test.csv')
    df = df.drop(columns=['timestamp_(min)'])
    test = df.to_numpy()
    print('test:', test.shape)
    #assert test.shape == (449919, 51)
    pkl.dump(test, open('PSM_test.pkl', 'wb'))
    print('PSM_test saved')

    df = pd.read_csv('psm/train.csv')
    df = df.drop(columns=['timestamp_(min)'])
    train = df.to_numpy()
    print(train.shape)
    #assert train.shape == (496800, 51)
    pkl.dump(train, open('PSM_train.pkl', 'wb'))
    print('PSM_train saved')
    return

def swat():
    # preprocess for SWaT. SWaT.A2_Dec2015, version 0
    df = pd.read_csv('./code_MGCLAD/datasets/swat/SWaT_test.csv', sep=',')
    y = df['label'].to_numpy()
    labels = []
    for i in y:
        if i == 1:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    # # 输出1的个数
    # print(np.sum(labels))
    # # 输出0的个数
    # print(len(labels) - np.sum(labels))
    print('labels:', labels.shape)
    #assert len(labels) == 449919
    labels = labels[4320:]
    print('labels:', labels.shape)
    os.makedirs('./code_MGCLAD/datasets/swat/processed', exist_ok=True)
    pkl.dump(labels, open('./code_MGCLAD/datasets/swat/processed/SWAT_test_label.pkl', 'wb'))
    print('SWaT_test_label saved')

    df = df.drop(columns=['Timestamp', 'label'])
    test = df.to_numpy()
    print('test:', test.shape)
    #assert test.shape == (449919, 51)
    test = test[4320:, :]
    print('test:', test.shape)
    pkl.dump(test, open('./code_MGCLAD/datasets/swat/processed/SWAT_test.pkl', 'wb'))
    print('SWaT_test saved')

    df = pd.read_csv('./code_MGCLAD/datasets/swat/SWaT_train.csv')
    df = df.drop(columns=['Timestamp'])
    train = df.to_numpy()
    print(train.shape)
    # train = train[4320:, :]
    # print(train.shape)
    #assert train.shape == (496800, 51)
    pkl.dump(train, open('./code_MGCLAD/datasets/swat/processed/SWAT_train.pkl', 'wb'))
    print('SWaT_train saved')


def swat_10():
    from scipy import stats

    name = ['Timestamp','FIT101','LIT101','MV101','P101','P102','AIT201','AIT202','AIT203','FIT201',
            'MV201','P201','P202','P203','P204','P205','P206','DPIT301','FIT301','LIT301',
            'MV301','MV302','MV303','MV304','P301','P302','AIT401','AIT402','FIT401','LIT401',
            'P401','P402','P403','P404','UV401','AIT501','AIT502','AIT503','AIT504','FIT501',
            'FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503','FIT601','P601',
            'P602','P603','Normal/Attack']

    # preprocess for SWaT. SWaT.A2_Dec2015, version 0
    df = pd.read_csv('./code_MGCLAD/datasets/swat/SWaT_Dataset_Attack_v0.csv', sep=';')
    y = df['Normal/Attack'].to_numpy()  # 449919
    labels = []
    for i in y:
        if i == 'Attack':
            labels.append(1)
        else:
            labels.append(0)

    # 取众数
    nlabels = []
    for i in range(44991):
        li = labels[i*10:i*10+10]
        # x = stats.mode(li)
        # mod = x[0][0]
        mod = max(set(li), key=li.count)
        #print(mod)
        nlabels.append(mod)
    
    nlabels = np.array(nlabels)    
    print('labels:', nlabels.shape)
    pkl.dump(nlabels, open('./code_MGCLAD/datasets/swat/processed/SWaT_test_label.pkl', 'wb'))
    print('SWaT_test_label saved')

    df = df.drop(columns=['Timestamp', 'Normal/Attack'])
    test = df.to_numpy()
    print('test:', test.shape)
    #assert test.shape == (449919, 51)
    test = test[4320:, :]
    print('test:', test.shape)

    data = []
    length = 44991
    for i in range(51):
        cols = []
        ori = test[:, i]
        #print(ori.shape)
        for j in range(length):
            cur = ori[j*10:(j+1)*10]
            median = np.median(cur)
            cols.append(median)

        data.append(cols)
        print(i, len(cols))

    data = np.asarray(data)
    print(data.shape)
    data = np.transpose(data,(1,0))
    print(data.shape)
    pkl.dump(data, open('./code_MGCLAD/datasets/swat/processed/SWaT_test.pkl', 'wb'))
    print('SWaT_test saved')

    df = pd.read_csv('./code_MGCLAD/datasets/swat/SWaT_Dataset_Normal_v0.csv')
    df = df.drop(columns=['Timestamp', 'Normal/Attack'])
    train = df.to_numpy()
    # print(train.shape)
    # train = train[4320:, :]
    # print(train.shape)
    data = []
    length = 49680
    for i in range(51):
        cols = []
        ori = train[:, i]
        #print(ori.shape)
        for j in range(length):
            cur = ori[j*10:(j+1)*10]
            median = np.median(cur)
            cols.append(median)

        data.append(cols)
        print(i, len(cols))

    data = np.asarray(data)
    print(data.shape)
    data = np.transpose(data,(1,0))
    data = data[4320:,:]
    print(data.shape)
    pkl.dump(data, open('./code_MGCLAD/datasets/swat/processed/SWaT_train.pkl', 'wb'))
    print('SWaT_train saved')




def wadi():

    # metrics :127 - 9(nan_cols) = 118 - 25(constant_cols) = 93
    sensors = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', 
               '1_AIT_005_PV', '1_FIT_001_PV', '1_LT_001_PV', '2_DPIT_001_PV', 
               '2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', 
               '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV', 
               '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', 
               '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', 
               '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV', #27
               '2_FIT_003_PV', '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', #32
               '2_FQ_501_PV', '2_FQ_601_PV', '2_LT_001_PV', '2_LT_002_PV', '2_MCV_101_CO', 
               '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO', #42
               '2_P_003_SPEED', '2_P_004_SPEED', '2_PIC_003_CO', '2_PIC_003_PV', '2_PIT_001_PV', 
               '2_PIT_002_PV', '2_PIT_003_PV', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', 
               '2A_AIT_004_PV', '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV', #57
               '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV', 
               '3_FIT_001_PV', '3_LT_001_PV', 'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW']

    actuators = ['1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS', 
                 '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
                 '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
                 '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
                 '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
                 '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']

    metrics = sensors + actuators

    print(len(metrics))

    # preprocess for wadi
    df = pd.read_csv('./code_MGCLAD/datasets/wadi/WADI_test.csv', sep=',')
    y = df['label'].to_numpy()
    labels = []
    for i in y:
        if i == 1:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    print('labels:', labels.shape)
    # labels = labels[::2]
    print('labels:', labels.shape)
    #assert len(labels) == 449919
    os.makedirs('./code_MGCLAD/datasets/wadi/processed', exist_ok=True)
    pkl.dump(labels, open('./code_MGCLAD/datasets/wadi/processed/WADI_test_label.pkl', 'wb'))
    print('WADI_test_label saved')

    #df = df.drop(columns=['Time', 'label'])
    df = df[metrics]
    test = df.to_numpy()
    print('test:', test.shape)
    # test = test[::2, :]
    print('test:', test.shape)
    #assert test.shape == (449919, 51)
    pkl.dump(test, open('./code_MGCLAD/datasets/wadi/processed/WADI_test.pkl', 'wb'))
    print('WADI_test saved')

    df = pd.read_csv('./code_MGCLAD/datasets/wadi/WADI_train.csv')
    #df = df.drop(columns=['Timestamp', 'Normal/Attack'])
    df = df[metrics]
    train = df.to_numpy()
    print('train:', train.shape)
    # train = train[::2, :]
    print('train:', train.shape)
    #assert train.shape == (496800, 51)
    pkl.dump(train, open('./code_MGCLAD/datasets/wadi/processed/WADI_train.pkl', 'wb'))
    print('WADI_train saved')


if __name__ == '__main__':
    # swat_10()
    #psm()
    swat()
    # wadi()

'''
# preprocess for WADI. WADI.A1
a = str(open('wadi/WADI_14days.csv', 'rb').read(), encoding='utf8').split('\n')[5: -1]
a = '\n'.join(a)
with open('train1.csv', 'wb') as f:
    f.write(a.encode('utf8'))
a = pd.read_csv('train1.csv', header=None)


a = a.to_numpy()[:, 3:]
nan_cols = []
for j in range(a.shape[1]):
    for i in range(a.shape[0]):
        if a[i][j] != a[i][j]:
            nan_cols.append(j)
            break
# len(nan_cols) == 9
train = np.delete(a, nan_cols, axis=1)
assert train.shape == (1209601, 118)
pkl.dump(train, open('WADI_train.pkl', 'wb'))
print('WADI_train saved')

df = pd.read_csv('wadi/WADI_attackdata.csv')
test = df.to_numpy()[:, 3:]
test = np.delete(test, nan_cols, axis=1)
assert test.shape == (172801, 118)
pkl.dump(test, open('WADI_test.pkl', 'wb'))
print('WADI_test saved')

print('WADI_test_label saved')

# WADI labels.pkl are created manually via the description file of the dataset
'''