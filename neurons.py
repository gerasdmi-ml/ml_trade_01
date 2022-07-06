
#from cloud_ml.storage.api import Storage
#disk = Storage.ya_disk(application_id='3e85f891293443ea8c9441a1b76c3176', application_secret='3a0ea3c8781948e6953901560e8c39fe')
#disk.get('LTCUSDT.csv', 'USDT.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras
from keras.utils.vis_utils import plot_model
from DNN_model import *

plt.style.use("seaborn")
pd.set_option('display.float_format', lambda x: '%.5f' % x)

## Data preparation


col_names = ['returns','ret_2', 'ret_5', 'ret_10','ret_21','rsi','macd','atr','stoch','ultosc','result','returns_v','ret_2_v', 'ret_5_v', 'ret_10_v','ret_21_v','rsi_v','macd_v','macd_0','macd_1','macd_2']
feature_cols = ['returns','ret_2', 'ret_5', 'ret_10','ret_21','rsi','macd','atr','stoch','ultosc','returns_v','ret_2_v', 'ret_5_v', 'ret_10_v','ret_21_v','rsi_v','macd_v','macd_0','macd_1','macd_2']
#data = pd.read_csv("ml_data_2.csv", sep=';', header=None, names=col_names)

data = pd.read_csv("D:/ml_data/ml_input/ml_data_529.csv", sep=';', header=None, names=col_names)


cols = feature_cols
df = data.copy()



## Test and train datasets
split = int(len(df)*0.66)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()
mu, std = train.mean(), train.std() # train set parameters (mu, std) for standardization
train_s = (train - mu) / std # standardization of train set features
#train_s = train   # local scaling disabled
test_s = (test - mu) / std # standardization of test set features (with train set parameters!!!)
#test_s = test     # local scaling disabled


set_seeds(100)

model = create_model(hl = 5, hu = 150, dropout = False, input_dim = len(cols),rate = 0.2,regularize = True)
model.fit(x = train_s[cols], y = train["result"], epochs = 150, verbose = False, validation_split = 0.01, shuffle = False, class_weight = cw(train), batch_size= 50)

model.evaluate(train_s[cols], train["result"]) # evaluate the fit on the train set
pred = model.predict(train_s[cols]) # prediction (probabilities)

'''
model.evaluate(test_s[cols], test["result"])
pred = model.predict(test_s[cols])
test['result2']=np.round(pred,1)
test.to_csv('test.csv', sep = ';',header=None)
'''


model.save("DNN_model")

# saving mu and std
import pickle
params = {"mu":mu, "std":std}
pickle.dump(params, open("ML_params.pkl", "wb"))



model2 = keras.models.load_model("DNN_model")
model2.evaluate(test_s[cols], test["result"])
pred = model2.predict(test_s[cols])
#print(np.round(pred,1))


'''
test.to_csv('test.csv', sep = ';',header=None)



# Loading mu and std
import pickle
params = pickle.load(open("ML_params.pkl", "rb"))
mu = params["mu"]
std = params["std"]

'''






