import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import io

'''
1. 데이터 불러오기
'''
dataDir = 'data/'
rawDir = dataDir + 'raw/'

file_list = os.listdir(rawDir)
print(file_list)

mat_list = []
for i, file_name in enumerate(file_list):
  mat_name = 'mat_{}'.format(i)
  mat_list.append(mat_name)
  globals()[f"{mat_name}"] = io.loadmat(rawDir+file_name)

print(mat_list)  # 각 mat 파일당 하나의 mat

'''
2. 데이터 preprocessing

데이터를 하나로 이어버리면, 각 mat이 연결되는 부분에서 데이터의 값이 크게 변할 수 있다.
TODO:
for mat in mat_list:
  데이터 불러오기 & 데이터 resampling
   데이터 클리닝
        - 현재는 앞부분에 튀는 데이터만 잘라줬다.
        - TODO: 필터링이 필요하다.
              # 차체의 진동으로 인해서 noise가 크게 발생하는 것으로 보임.
              # 같은 시점의 다른 데이터(ex 기어 시프팅 등) 또한 볼 필요가 있어보이나... 귀찮음.
  TODO: 데이터 normalization
        - 일반적 최고 속도와 최고 가속도를 기준으로 normalization을 진행하자.
  데이터 Sequence
  데이터 스플릿
        - train/test /+validation 
        - 5개의 데이터가 있으니까, 학습 4, 테스트 1로 설정해도 무방할 듯.
'''
# 관심 있는 변수 설정.
a_des = 'Veh_Ctrl__DesiredAcceleration__________________________________'
Vx = 'VCU1_LRR__vcuLrr_VEHICLE_SPEED_________________________________'
a_sen = 'Acceleration_Info2__Veh_Acc_Xaxis______________________________'
SWA_des = 'Veh_Ctrl__DesiredSteerAngle____________________________________'

data_name_list = ['a_des', 'Vx', 'a_sen']
data_list = [globals()[name] for name in data_name_list]
print(data_list)


# ## 2.1 데이터 resampling
dt = '10ms'  # TODO: dt를 어떻게 가져갈지 또한 생각해야한다.
mat_dfs = pd.DataFrame()
for mat_name in mat_list:
  mat = globals()[mat_name]

  mat_df = pd.DataFrame()

  for j, data_name in enumerate(data_list):
    data = mat[data_name]
    df = pd.DataFrame(data[:, 1], index=data[:, 0])
    df.columns = [data_name_list[j]]
    df.index = pd.to_datetime(df.index, unit='s')

    df = df.resample(dt).last().fillna(method='ffill')
    if data_name_list[j] == 'a_sen':
      df['a_sen'] = df['a_sen']*9.81

    mat_df = pd.concat([mat_df, df], axis=1).fillna(0)
  break
  # mat_dfs = pd.concat([mat_dfs, mat_df], axis=0, ignore_index=True)

# ## 2.2 데이터 클리닝
data = mat_df[['Vx', 'a_sen', 'a_des']][200:].values


# ## 2.3 데이터 normalization


# ## 2.4 to Sequence
def makeSequence(data, inSeq_len, outSeq_len, label_idx):
  Xs = []
  Ys = []
  for i in range(len(data) + 1 - (inSeq_len+outSeq_len)):
    x = data[i:i+inSeq_len]
    y = data[i+inSeq_len: i+inSeq_len+outSeq_len, label_idx]

    Xs.append(x)
    Ys.append(y)

  Xs = np.array(Xs).reshape(len(Xs), inSeq_len, -1)
  Ys = np.array(Ys).reshape(len(Xs), inSeq_len, -1)

  return Xs, Ys


n = 20  # input seqeunce length (dt(=10ms) * 20 = 200ms?!?!!)
m = 20  # output seqeunce length

Xs, Ys = makeSequence(data, n, m, 1)
print(Xs.shape)
print(Ys.shape)

# ## 2.5 data split
train_data_num = int(len(data)*0.8)
print(train_data_num)
trainX = Xs[:train_data_num]
trainY = Ys[:train_data_num]
testX = Xs[:train_data_num]
testY = Ys[:train_data_num]

'''
3. Data 저장

'''
prcsDir = dataDir + 'resampled/'

try:
  if not os.path.exists(prcsDir):
    os.makedirs(prcsDir)
except OSError:
  print('Error: Creating directory. ' + prcsDir)

data.save()
# 뭘로 저장할까?
# numpy? 가 유력.
