import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
import numpy as np
import datetime
import talib as ta
import logging
from sklearn.preprocessing import scale
from shutil import copyfile
import keras
import pyodbc
from keras.utils.vis_utils import plot_model
from DNN_model import *

import pickle

# Date;Open;High;Low;Adj Close;Volume;Close time;Quote asset volume;Number of trades;Taker buy base asset volume;Taker buy quote asset volume;Ignore
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Overall minimum logging level
file_handler = logging.FileHandler('info.log')  # Configure the logging messages written to a file
logger.addHandler(file_handler)


class TradeClient:
    def __init__(self, coin,open_volume,open_volume_max,close_volume,dir,generate,version,mode):
        self.data_dir = dir
        self.generate = generate
        self.version = version
        self.mode = mode
        self.open_volume = open_volume
        self.open_volume_max = open_volume_max
        self.close_volume = close_volume
        self.coin = coin
        self.overall_success = 0

        if self.generate == 1:
            params = pickle.load(open("ML_params.pkl", "rb"))
            self.mu = params["mu"]
            self.std = params["std"]
            self.model2 = keras.models.load_model("DNN_model")

        self.max_mins_in_position = 60
        self.debug = 0
        self.deposit = 1000
        self.leverage = 1
        self.cap_rate = 0.0
        self.open_counter = 0

        self.ml_counter = 0
        self.ml_df = pd.DataFrame(columns=['returns','ret_2', 'ret_5', 'ret_10','ret_21','rsi','macd','atr','stoch','ultosc','result',
                                           'returns_v','ret_2_v', 'ret_5_v', 'ret_10_v','ret_21_v','rsi_v','macd_v','macd_0','macd_1','macd_2','result_mean'])
        self.cols = cols = ['returns','ret_2', 'ret_5', 'ret_10','ret_21','rsi','macd','atr','stoch','ultosc','returns_v','ret_2_v', 'ret_5_v', 'ret_10_v','ret_21_v','rsi_v','macd_v','macd_0','macd_1','macd_2','result_mean']



        #self.coin_setup(coin)
        self.data_path = f'{self.data_dir}_{coin}USDT.csv'
        self.price_change_allowed = 3.2

        self.tech_zeros()

        '''
        self.volume_mean = round(self.df_cut['Volume'].to_numpy().mean(),2)

        self.large_position_treshhold_open = open_volume * self.volume_mean
        self.large_position_treshhold_open_max = open_volume_max * self.volume_mean
        self.large_position_treshhold_close = close_volume * self.volume_mean
        '''



        self.update_volume_first_time()


        self.df_cut["ema_vn"] = self.df_cut['Volume'].ewm(span=20).mean()
        self.df_cut["rav"] = self.df_cut['Volume'].rolling(window=20).mean()
        self.df_cut['ret_1'] = self.df_cut['Adj Close'].pct_change(1) * 100
        self.df_cut['ret_2'] = self.df_cut['Adj Close'].pct_change(2) * 100
        self.df_cut['ret_21'] = self.df_cut['Adj Close'].pct_change(21) * 100

        self.df_cut['ret_5'] = self.df_cut['Adj Close'].pct_change(5) * 100
        self.df_cut['atr'] = ta.ATR(self.df_cut.High, self.df_cut.Low, self.df_cut['Adj Close'])
        self.df_cut['macd_0'], self.df_cut['macd_1'], self.df_cut['macd_2'] = ta.MACD(self.df_cut['Adj Close'], 12, 26, 9)
        self.df_cut['macd'] = ta.MACD(self.df_cut['Adj Close'])[1]
        self.df_cut['macd_v'] = ta.MACD(self.df_cut['Volume'])[1]
        self.df_cut['ret_21_v'] = self.df_cut['Volume'].pct_change(21)
        self.df_cut['ret_2_v'] = self.df_cut['Volume'].pct_change(2)
        self.df_cut['ret_5_v'] = self.df_cut['Volume'].pct_change(5)

        # and volume
        message = f'preprocess volume_mean/open/% {self.data_path}= {self.volume_mean} {open_volume} {open_volume_max} {close_volume}'
        print (message)
        #logger.info(message)


        for self.index, self.row in self.df_cut.iterrows():

            if self.ongoing_position == False:
                self.cond_open_1 = self.row['Volume'] > self.large_position_treshhold_open and self.row['Volume'] < self.large_position_treshhold_open_max and self.index >20
                self.cond_open_2 =  self.row['ret_1']>0


                if self.cond_open_1  and self.cond_open_2    :


                    if self.generate == 1:
                        self.create_ML_dataframe(self.df_cut.iloc[:self.index + 1])

                        self.open_position()

                    if self.generate ==0:
                        self.open_position()

            if self.ongoing_position == True:


                self.cond_exit_1 = (self.row['Volume'] <= self.large_position_treshhold_close)

                if (self.cond_exit_1)  : self.close_position()

        if self.negative + self.positive !=0:
            target_function = round(self.positive / (self.negative + self.positive) *0.7 + self.price_difference_ytd/self.deposit*0.3, 2)

        else:
            target_function =0
            self.positive = 1

        message = f'PNL TOTAL {coin}  {target_function} {round(self.positive / (self.negative + self.positive), 2)} {round(self.price_difference_ytd, 2)} volume= {round(self.large_position_treshhold_open,-2)} '
        logger.info(message)
        print(message)

        if self.generate==1:
            self.ml_df["result_mean"] = self.ml_df['result'].rolling(window=20).mean()
            self.ml_df.to_csv(f'D:/ml_data/ml_input/ml_data_{self.version}.csv', sep=';', header=None, mode='a')

        #return (target_function)

    def coin_setup(self,coin):
        if coin == 'ATOM':
            self.data_path = f'{self.data_dir}_ATOMUSDT.csv'
            self.large_position_treshhold_open = 100000  # 45000  4-7% from average per minute !!!
            self.max_rsi = 70
            self.price_change_allowed = 3.2
        if coin == 'AVAX':
            self.data_path = f'{self.data_dir}_{coin}USDT.csv'
            self.large_position_treshhold_open = 95000  # 45000  4-7% from average per minute !!!
            self.max_rsi = 70
            self.price_change_allowed = 3.2
        if coin == 'AXS':
            self.data_path = f'{self.data_dir}_{coin}USDT.csv'
            self.large_position_treshhold_open = 200000  # 45000  4-7% from average per minute !!!
            self.max_rsi = 70
            self.price_change_allowed = 3.2
        if coin == 'BCH':
            self.data_path = f'{self.data_dir}_{coin}USDT.csv'
            self.large_position_treshhold_open = 7000  # 45000  4-7% from average per minute !!!
            self.max_rsi = 70
            self.price_change_allowed = 3.2
        if coin == 'XMR':
            self.data_path = f'{self.data_dir}_{coin}USDT.csv'
            self.large_position_treshhold_open = 3000  # 45000  4-7% from average per minute !!!
            self.max_rsi = 70
            self.price_change_allowed = 3.2


        if coin == 'LTC':
            self.data_path = f'{self.data_dir}_LTCUSDT.csv'
            self.large_position_treshhold_open = 45000  # 45000  4-7% from average per minute !!!
            self.max_rsi = 70
            self.price_change_allowed = 3.2
        elif coin == 'ETH':
            self.data_path = f'{self.data_dir}_ETHUSDT.csv'
            self.large_position_treshhold_open = 35000  #  50000
            self.max_rsi = 70
            self.price_change_allowed = 3.2
        elif coin == 'BNB':
            self.data_path = f'{self.data_dir}_BNBUSDT.csv'
            self.large_position_treshhold_open = 50000 # 55000 32
            self.max_rsi = 70
            self.price_change_allowed = 3.2   # 1.6
        elif coin == 'AAVE':   #
            self.data_path = f'{self.data_dir}_AAVEUSDT.csv'
            self.large_position_treshhold_open = 7500  #10000
            self.price_change_allowed = 2.2
            self.max_rsi = 70
        elif coin == 'SOL':   #
            self.data_path = f'{self.data_dir}_SOLUSDT.csv'
            self.large_position_treshhold_open = 150000 #200000
            self.max_rsi = 70
            self.price_change_allowed = 2.2
        elif coin == 'ETC':   #
            self.data_path = f'{self.data_dir}_ETCUSDT.csv'
            self.large_position_treshhold_open = 150000  #200000
            self.max_rsi = 50
            self.price_change_allowed = 2.2
        elif coin == 'LINK':
            self.data_path = f'{self.data_dir}_LINKUSDT.csv'
            self.large_position_treshhold_open = 200000  # 200000
            self.max_rsi = 50
            self.price_change_allowed = 3.2

    def tech_zeros(self):
        self.positive_pnl = 0
        self.negative_pnl = 0
        self.dict_pnl = {}
        self.dict_alpha_lib = {}
        self.dict_alpha_lib_index = 0
        self.position_pnl = 0
        self.price_difference_ytd = 0
        self.ongoing_position = False
        self.transactions_count = 0
        self.open_entry=0
        self.open_price=0
        self.position =0
        self.comission_dynamic = 0
        self.comission_dynamic_ytd = 0
        self.transactions_count = 0

        self.df_initial= pd.read_csv(self.data_path, sep=';')  # minute positions for ...
        self.cnt = self.df_initial.count()[0]
        self.from_record = 0 #
        self.to_record = self.cnt - 1  #
        split = int(self.cnt * 0.66)
        if self.mode == 'train':
            self.from_record = 0
            self.to_record = split
        if self.mode =='test':
            self.from_record = split
            self.to_record = self.cnt - 1
        if self.mode =='full':
            self.from_record = 0
            self.to_record = self.cnt - 1


        self.df_cut = self.df_initial.iloc[self.from_record:self.to_record,]
        self.df_cut = self.df_cut.reset_index(drop=True)

        self.negative = 0
        self.positive = 0

    def open_position(self):

        #self.panic_index = round(self.row['btc_ret_21'],2)

        if self.index == 0: self.prev_price = self.row['Adj Close']
        else:  self.prev_price = self.df_cut.iloc[self.index - 1,]['Adj Close']
        self.open_price = self.row['Adj Close']

        self.open_date = self.row['Date']

        self.open_entry = self.index
        self.ongoing_position = True
        self.transactions_count = self.transactions_count + 1

        if self.prev_price > self.open_price:
            self.position = 1
        else:
            self.position = -1

        self.position = 1

    def close_position(self):

        self.close_price = self.row['Adj Close']
        self.close_date = self.row['Date']
        self.ongoing_position = False

        # capitalization with rate self.cap_rate
        self.new_deposit = self.deposit + self.cap_rate*self.price_difference_ytd
        # no capitalization
        #self.new_deposit = self.deposit

        self.price_difference = self.position * (
                    self.close_price - self.open_price) * self.new_deposit / self.open_price * self.leverage

        # post comissions at the moment of initialization
        self.comission_dynamic = round(abs(self.new_deposit * 2 * 0.04 / 100 * self.leverage),2)
        self.price_difference = self.price_difference - self.comission_dynamic

        self.comission_dynamic_ytd = round(
            self.comission_dynamic_ytd + abs(self.new_deposit * 2 * 0.04 / 100 * self.leverage), 2)
        self.price_difference_ytd = self.price_difference_ytd + self.price_difference
        self.max_allowed_position_loss = - round(self.new_deposit * self.leverage / 100 * 30,2)  # see 368010 - price change from 175.20 to 229.30 in a few minutes = 31%

        if self.price_difference > 0:
            self.positive_pnl = self.positive_pnl + self.price_difference
        else:
            self.negative_pnl = self.negative_pnl + self.price_difference


        if round(self.price_difference, 2) < 0:
            self.warning = '!!!'
            self.negative = self.negative + 1
        else:
            self.warning = '___'
            self.positive = self.positive + 1

        self.sql_analysis()

        if self.price_difference < 1000000:
            print(
                f'{self.warning} {self.open_entry}...{self.index}, {int(self.open_date)}..{int(self.close_date)} APY/Comm =  {round(self.price_difference_ytd, 2)}/{self.comission_dynamic_ytd}  deal= {round(self.price_difference, 2)},'
                f' open/close= {self.open_price}/{self.close_price}, pos/mins={self.position}/{self.index - self.open_entry}'
                f'     ')

        if self.generate==1:
            self.update_ML_dataframe(round(self.price_difference, 2))

    def create_ML_dataframe(self,data):


        data2 = data.copy()
        #print(data2)


        data2.drop('Date', axis=1, inplace=True)
        data2.drop('Open', axis=1, inplace=True)
        data2.drop('Ignore', axis=1, inplace=True)
        data2.drop('Number of trades', axis=1, inplace=True)
        data2.drop('Taker buy base asset volume', axis=1, inplace=True)
        data2.drop('Taker buy quote asset volume', axis=1, inplace=True)
        data2.drop('Quote asset volume', axis=1, inplace=True)
        data2.drop('Close time', axis=1, inplace=True)
        data2 = data2.rename(columns={'High': 'high', 'Low': 'low'})
        data2 = data2.rename(columns={'Adj Close': 'close', 'Volume': 'volume'})
        data2 = data2[['close', 'volume', 'low', 'high']]

        data2['returns'] = data2.close.pct_change()                     # and volume
        data2['ret_2'] = data2.close.pct_change(2)                      # and volume
        data2['ret_5'] = data2.close.pct_change(5)                      # and volume
        data2['ret_10'] = data2.close.pct_change(10)                    # and volume
        data2['ret_21'] = data2.close.pct_change(21)                    # and volume
        data2['rsi'] = ta.STOCHRSI(data2.close)[1]                      # and volume
        data2['macd'] = ta.MACD(data2.close)[1]                         # and volume
        data2['atr'] = ta.ATR(data2.high, data2.low, data2.close)
        slowk, slowd = ta.STOCH(data2.high, data2.low, data2.close)
        data2['stoch'] = slowd - slowk
        data2['atr'] = ta.ATR(data2.high, data2.low, data2.close)
        data2['ultosc'] = ta.ULTOSC(data2.high, data2.low, data2.close)

        data2['returns_v'] = data2.volume.pct_change()                     # and volume
        data2['ret_2_v'] = data2.volume.pct_change(2)                      # and volume
        data2['ret_5_v'] = data2.volume.pct_change(5)                      # and volume
        data2['ret_10_v'] = data2.volume.pct_change(10)                    # and volume
        data2['ret_21_v'] = data2.volume.pct_change(21)                    # and volume
        data2['rsi_v'] = ta.STOCHRSI(data2.volume)[1]                      # and volume
        data2['macd_v'] = ta.MACD(data2.volume)[1]                         # and volume
        data2['macd_0'],  data2['macd_1'],  data2['macd_2'] = ta.MACD( data2['close'], 12, 26, 9)


        data2 = (data2.replace((np.inf, -np.inf), np.nan).drop(['high', 'low', 'close', 'volume'], axis=1).dropna())
        r = data2.returns.copy()
        data2 = pd.DataFrame(scale(data2), columns=data2.columns, index=data2.index)
        #data2['coin'] = self.coin_number()
        features = data2.columns.drop('returns')
        data2['returns'] = r  # don't scale returns
        data2 = data2.loc[:, ['returns'] + list(features)]

        data3 = data2.iloc[-1,].copy()
        data3['result'] = 0

        #print(data3)
        self.ml_counter = self.ml_counter + 1
        self.ml_df.loc[self.ml_counter] = data3.copy()

    def update_ML_dataframe(self,result):
        if result>=0 :
            self.ml_df.iloc[-1,]['result'] = 1
        else:
            self.ml_df.iloc[-1,]['result'] = 0

    def update_volume(self,index):
        #print (f'update volume with index {index}')

        from_record = index - 239999
        to_record = index - 1
        df_volume = self.df_cut.iloc[from_record:to_record,]

        self.volume_mean = round(df_volume['Volume'].to_numpy().mean(),2)

        self.large_position_treshhold_open = self.open_volume * self.volume_mean
        self.large_position_treshhold_open_max = self.open_volume_max * self.volume_mean
        self.large_position_treshhold_close = self.close_volume * self.volume_mean

        print(f'update volume with index = {index} new value= {self.large_position_treshhold_open}')

    def update_volume_first_time(self):

        dir = 'D:/Data_2020/'
        print (f'update volume with INITIAL index ')
        data_path = f'{dir}_{self.coin}USDT.csv'
        df_cut2 = pd.read_csv(data_path, sep=';')

        # replace Volume with Quote asset volume
        #df_cut2.drop(['Volume'], axis=1, inplace=True)
        #df_cut2['Volume']= df_cut2['Quote asset volume']


        cnt = df_cut2.count()[0]
        from_record = cnt - 240000
        to_record = cnt - 1
        df_volume = df_cut2.iloc[from_record:to_record,]

        self.volume_mean = round(df_volume['Volume'].to_numpy().mean(),2)

        self.large_position_treshhold_open = self.open_volume * self.volume_mean
        self.large_position_treshhold_open_max = self.open_volume_max * self.volume_mean
        self.large_position_treshhold_close = self.close_volume * self.volume_mean

        print(f'update volume with INITIAL VALUE new value= {self.large_position_treshhold_open}')

    def coin_number(self):
        if self.coin =='ETH':
            return (1)
        if self.coin =='BNB':
            return (2)
        if self.coin =='SOL':
            return (3)
        if self.coin =='AVAX':
            return (4)
        if self.coin =='ADA':
            return (5)

    def sql_analysis(self):

        from_i = str(self.open_date)
        to_i = str(self.close_date)
        case = str(self.price_difference)
        from_p = str(float(from_i) + 60000) #+ 1 минута

        from_before = str(float(from_i) - 120 * 60000) # minus 2 hour
        to_before = str(float(from_i) + 60000)


        cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                              "Server=DESKTOP-LB73L9Q;"
                              "Database=Binance;"
                              "Trusted_Connection=yes;")

        df_front = pd.read_sql(
            f"select top 1120000 *,dateadd(S, [time]/1000, '1970-01-01') as time2 from all_trades where time >= {from_i} and time <= {from_p} order by time",
            cnxn)
        df_wave = pd.read_sql(
            f"select top 1120000 *,dateadd(S, [time]/1000, '1970-01-01') as time2 from all_trades where time >= {from_p} and time <= {to_i} order by time",
            cnxn)
        df_before = pd.read_sql(
            f"select   *,dateadd(S, [time]/1000, '1970-01-01') as time2 from all_trades where time >= {from_before} and time <= {to_before} order by time",
            cnxn)



        # prepare TA KPIs
        df_wave['ret_1'] = df_wave['price'].pct_change(1) * 100
        df_wave['ret_2'] = df_wave['price'].pct_change(2) * 100
        df_wave['ret_5'] = df_wave['price'].pct_change(5) * 100
        df_wave['ret_10'] = df_wave['price'].pct_change(10) * 100
        df_wave['ret_21'] = df_wave['price'].pct_change(21) * 100
        df_wave['ret_1_v'] = df_wave['quantity'].pct_change(1)
        df_wave['ret_2_v'] = df_wave['quantity'].pct_change(2)
        df_wave['ret_5_v'] = df_wave['quantity'].pct_change(5)
        df_wave['ret_10_v'] = df_wave['quantity'].pct_change(10)
        df_wave['ret_21_v'] = df_wave['quantity'].pct_change(21)
        #df_wave['rsi'] = ta.STOCHRSI(df_wave['price'])[1]  # and volume
        #df_wave['macd'] = ta.MACD(df_wave['price'])[1]  # and volume

        # df1=df1.iloc[200:,]
        df_wave['ret_1_v'] = df_wave['ret_1_v'].fillna(0)
        df_wave['ret_2_v'] = df_wave['ret_2_v'].fillna(0)
        df_wave['ret_5_v'] = df_wave['ret_5_v'].fillna(0)
        df_wave['ret_10_v'] = df_wave['ret_10_v'].fillna(0)
        df_wave['ret_21_v'] = df_wave['ret_21_v'].fillna(0)

        figure, axis = plt.subplots(2, 2)

        axis[0, 0].plot(df_front.loc[:, 'price'] * 1, color='r', lw=1)
        axis[0, 0].set_title("front price")
        axis[1, 0].plot(df_before.loc[:, 'price'] * 1, color='y', lw=1)
        axis[1, 0].set_title("before front price")
        axis[0, 1].plot(df_wave.loc[:, 'price'] * 1, color='y', lw=1)
        axis[0, 1].set_title("wave price")
        axis[1, 1].plot(df_wave.loc[:, 'quantity'] * 1, color='y', lw=1)
        axis[1, 1].set_title("wave quantity")



        figure.suptitle(case, fontsize=16)
        plt.savefig(f'C:/Python/full_cycle_02/SQL_pic2/{from_i}.png')

# set 20 60 3 to prod

generation_file_version='702'
comment_text=' CLEAR (lev = 1. cap = 0.0)'
const_v_open=20
const_v_max=60
const_v_close=3
mode = 'full'   # full/train/test
generate=0      # generate or not ML data
dir = 'D:/Data_2022/'

copyfile('generate_ml.py',f'D:/source_ver/{generation_file_version}.py')
list_coins=['AAVE','ADA','ALGO','ATOM','AVAX','AXS','BCH','BNB','BTC','DOGE','EGLD','EOS','ETC','ETH','FIL','GRT','KSM','LINK','LTC','LUNA','MATIC','MKR','NEO','SOL','THETA','TRX','UNI','VET','WAVES','XLM','XMR','XRP'] ## in tests
list_coins=['ETH','BNB','SOL','XRP','AVAX','ADA','LTC','ETC','LINK','MATIC','XRP'] ## in tests

list_coins=['ETC'] ## in tests


logger.info(f'run with version = {generation_file_version} {comment_text} {const_v_open} {const_v_max} {const_v_close} {dir} ')
for i in list_coins:
    TradeClent =TradeClient(i,const_v_open,const_v_max,const_v_close,dir,generate,generation_file_version,mode)




