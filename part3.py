import random
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal, stats
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as st
import pandas as pd
import statsmodels.api as sm
import statsmodels
from neuralprophet import NeuralProphet


# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符
    num_examples = (len(corpus_indices) - 1) // num_steps  # 下取整，得到不重叠情况下的样本个数
    example_indices = [i * num_steps for i in range(num_examples)]  # 每个样本的第一个字符在corpus_indices中的下标
    random.shuffle(example_indices)

    def _data(i):
        # 返回从i开始的长为num_steps的序列
        return corpus_indices[i: i + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0, num_examples, batch_size):
        # 每次选出batch_size个随机样本
        batch_indices = example_indices[i: i + batch_size]  # 当前batch的各个样本的首字符的下标
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)

def one_hot(x, n_class, dtype=torch.float32):
    result = torch.zeros(len(x), n_class, dtype=dtype, device=x.device)  # shape: (n, n_class)
    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result
    
def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, inputs, state):
        # inputs.shape: (batch_size, num_steps)
        X = to_onehot(inputs, self.vocab_size)
        X = torch.stack(X)  # X.shape: (num_steps, batch_size, vocab_size)
        hiddens, state = self.rnn(X, state)
        hiddens = hiddens.view(-1, hiddens.shape[-1])  # hiddens.shape: (num_steps * batch_size, hidden_size)
        output = self.dense(hiddens)
        return output, state

# 实现一个预测函数，与前面的区别在于前向计算和初始化隐藏状态
def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y.argmax(dim=1).item())
    return '=>'.join([idx_to_char[i] for i in output])

def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
    corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )
    batch_num = (indices.shape[1] - 1) // num_steps
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        l_sum, n = 0.0, 0
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        state = None
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state[0].detach_()
                    state[1].detach_()
                else: 
                    state.detach_()
            (output, state) = model(X, state) # output.shape: (num_steps * batch_size, vocab_size)
            y = torch.flatten(Y.T)
            l = loss(output, y.long())
            loss_list = [l]
            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]


        if (epoch + 1) % pred_period == 0:
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))

def FRT(raw, k, part, path):
    fft_size = 1024*4
    peak_num =3 #找前几个周期；
    figure_flag = 1 #傅里叶变换是否画图，0：不画图 1：画图
    #获取指数同比序列
    seq = raw[k]
    seq_log = np.array(seq)
    dseq_log = signal.detrend(seq_log) #去趋势项
    freq_index = list(range(fft_size))
    freq_index[: int(fft_size / 2 + 1)] = [i / fft_size for i in range(int(fft_size / 2 + 1))]

    # freq_index(fft_size/2+2:end) = [-fft_size/2+1:1:-1]/fft_size
    freq_index[int(fft_size/2+1):] = [i / fft_size for i in range(int(-fft_size / 2 + 1), 0)]
    # data_fft = abs(fftshift(fft(dseq_log, fft_size)));
    data_fft = abs(np.fft.fftshift(np.fft.fft(dseq_log, fft_size)))
    #freq_index = fftshift(freq_index);
    freq_index = np.fft.fftshift(freq_index)
    peak_num = 3

    loc_raw = list(signal.find_peaks(data_fft[int(fft_size/2):])[0])
    peak_raw = data_fft[int(fft_size/2):][loc_raw]
    data_dict = dict(zip(loc_raw,peak_raw))
    temp = sorted(data_dict.items(),key=lambda x:x[1],reverse=True)
    loc_peaks = temp[:peak_num]

    plt.figure(figsize=(20,10))
    plt.plot(freq_index[int(fft_size/2+1):],data_fft[int(fft_size/2+1):])
    plt.grid(True)
    plt.xlabel('频率(Hz)', fontsize=20)
    plt.ylabel('幅度', fontsize=20)
    title = f'{part} {k}幅度频率图'
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for i in range(peak_num):
        plt.text(freq_index[loc_peaks[i][0] + int(fft_size / 2 - 1)], loc_peaks[i][1], f'[{str(round(1 / freq_index[loc_peaks[i][0] + int(fft_size / 2 - 1)], 2))},{str(round(loc_peaks[i][1]))}]', fontsize=20)

    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100) 

    # dseq_log,fft_size,peak_num,figure_flag
    def period_mean_fft(data,nfft,peak_num,figure_flag):
        data_n = np.array(data)
        if len(data_n.shape)>1 and data_n.shape[-1]>data_n.shape[0]:
            data_n = data_n.T
        Y_fft = np.fft.fft(data_n,nfft)
        nfft = len(Y_fft)
        Y_fft = np.delete(Y_fft,0)
        #power = abs(Y(1:floor(nfft/2))).^2;  %求功率谱
        power = np.abs(np.square(Y_fft[:int(np.floor(nfft/2))]))  #求功率谱
        #Amplitude = abs(Y(1:floor(nfft/2)));  %求振幅
        amplitude = np.abs(Y_fft[:int(np.floor(nfft/2))])  #求振幅
        nyquist = 1/2
        #freq = (1:floor(nfft/2))/(floor(nfft/2))*nyquist; 
        # %求频率，从低频到高频，最高的频率是nyquist,最低的频率是2/N
        freq = [nyquist*i/np.floor(nfft/2) for i in range(1,int(np.floor(nfft/2)+1))]
        freq = np.array(freq)
        period = 1/freq
        #[peaks,posi] = findpeaks(power,'NPeaks',peak_num,'SortStr','descend');
        loc_raw = list(signal.find_peaks(power)[0])
        peak_raw = power[loc_raw]
        data_dict = dict(zip(loc_raw,peak_raw))
        temp = sorted(data_dict.items(),key=lambda x:x[1],reverse=True)
        loc_peaks = temp[:peak_num]
        posi = [loc_peaks[i][0] for i in range(peak_num)]
        T_fft = period[posi]
        if figure_flag == 1:
            plt.figure(figsize=(20,10))
            plt.plot(period,power)
            plt.grid(True)
            plt.xlabel('周期', fontsize=20)
            plt.ylabel('功率', fontsize=20)
            title = f'{part} {k}的周期-振幅图'
            plt.title(title, fontsize=20)
            for i in range(peak_num):
                plt.text(period[loc_peaks[i][0]], loc_peaks[i][1], f'[{str(round(period[loc_peaks[i][0]], 2))},{str(round(loc_peaks[i][1]))}]', fontsize=20)

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim(12,300)
            file = os.path.join(path, f'{title}.png')
            plt.savefig(file, dpi=100) 

        return T_fft

    return period_mean_fft(dseq_log,fft_size,peak_num,figure_flag)



#ARIMA
class ARIMA_auto:
    def __init__(self, data, col, part, path):
        self.col = col
        self.data = data
        self.path = path
        self.part = part
        self.result_ARMA = self.fit()
        self.predictions_ARMA = self.pre()
        self.arch_p = self.arch_diag()

    def cum_sum(self, data, n=1):
        return data.cumsum() if n == 1 else self.cum_sum(data, n - 1).cumsum()

    def diff(self, data, n):
        return data.diff() if n == 1 else self.diff(data, n - 1).diff()

    # Test seasonality and stationarity 
    def diff_stat(self):
        nor_p = stats.shapiro(self.data)
        if nor_p[1] > 0.05:
            print(f'{self.col} can be assumed to be normally distributed')
        else:
            print(f'{self.col} cannot be assumed to be normally distributed')

        # ACF
        fig, ax = plt.subplots(figsize=(20, 10))
        fig = plot_acf(self.data, alpha=0.05, ax=ax, lags=10, unbiased=True)
        title = f'{self.part} {self.col}的ACF图'
        plt.xlabel('Order', fontsize=20)
        plt.ylabel('ACF', fontsize=20)
        plt.title(title, fontsize=20)   
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100) 

        # PACF
        fig, ax = plt.subplots(figsize=(20, 10))
        fig = plot_pacf(self.data, alpha=0.05, ax=ax, lags=10)
        title = f'{self.part} {self.col}的PACF图'
        plt.xlabel('Order', fontsize=20)
        plt.ylabel('PACF', fontsize=20)
        plt.title(title, fontsize=20)   
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100) 

        # 时间序列分解
        ts = seasonal_decompose(self.data, freq=12)
        plt.figure(figsize=(20, 10))
        ts.plot()
        title = f'{self.part} {self.col}的时间序列分解图'
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100)

        p_value = adfuller(self.data)[1]
        if p_value < 0.05:
            print(f'{self.col} is stationary')
            return 0
        print(f'{self.col} is not stationary')
        i = 1 
        p_value = adfuller(self.diff(self.data, i).dropna())[1]
        while p_value >= 0.05:
            i += 1
            p_value = adfuller(self.diff(self.data, i).dropna())[1]
        return i


    # find the order for ARMA 
    @staticmethod
    def find_order(series):
        order = st.arma_order_select_ic(
                series, 
                max_ar=3,
                max_ma=3, 
                ic=['aic', 'bic', 'hqic']
            )
        order_min = order.aic_min_order
        p = order_min[0]
        q = order_min[1]
        return p, q

    def _fit(self, series, order):
        p, d, q = order
        if d == 0:
            model = ARMA(
                    series, 
                    order=(p, q)
            )
        else:
            model = ARMA(
                    self.diff(series, d).dropna(), 
                    order=(p, q)
            )
        return model.fit(disp=-1)

    def fit(self):
        # find the order for differentiation
        d = self.diff_stat()
        if d > 0:
            p, q = self.find_order(self.diff(self.data, d).dropna())
        else:
            p, q = self.find_order(self.data)

        self.arima_order = p, d, q
        # fit the ARMA        
        try:
            return self._fit(self.data, self.arima_order)
        except:
            d += 1
            p, q = self.find_order(self.diff(self.data, d).dropna())
            self.arima_order = p, d, q
            return self._fit(self.data, self.arima_order)

    # predict
    def pre(self):
        predictions_ARMA_diff = pd.Series(
            self.result_ARMA.fittedvalues, 
            copy=True,
            index=self.data.index
        )
        plt.figure(figsize=(20, 10))
        if self.arima_order[1] > 0:
            predictions_ARMA = self.cum_sum(predictions_ARMA_diff, self.arima_order[1])
        else:
            predictions_ARMA = predictions_ARMA_diff

        temp = pd.concat(
            [self.data, predictions_ARMA], axis=1
        )
        temp.columns = ['original data', 'fitted data']
        colormaps = {
                'original data':'blue',
                'fitted data':'red'
            }
        for col in temp.columns:
            temp[col].plot(
                style=colormaps.get(col)
            )
        plt.legend(loc=1, fontsize=20)
        title = f'{self.part} {self.col} ARIMA({self.arima_order[0]}, {self.arima_order[1]}, {self.arima_order[2]})'
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Duration', fontsize=20)
        plt.title(title, fontsize=20)   
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100)
        return predictions_ARMA

    # test ARCH effect
    def arch_diag(self):
        # ARIMA残差正态检验
        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(121)
        resid = (self.data - self.predictions_ARMA).dropna()
        resid.hist(ax=ax1, bins=100)
        resid.plot(ax=ax1, kind='kde')
        self._extracted_from_arch_diag_8('Square of residuals', 'Frequency')
        ax2 = plt.subplot(122)
        stats.probplot(resid, dist='norm', plot=ax2, fit=True)  # test normality
        self._extracted_from_arch_diag_8('Theoritical quantitles', 'Real quantitles')
        plt.title('Probability Plot', fontsize=20)
        title = f'{self.part} {self.col}的ARIMA残差正态检验图'
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100)


        #test ARCH effect
        resid2 = pd.DataFrame(resid**2)
        resid2.plot(figsize=(20, 10))
        title = f'{self.part} {self.col} Time Series of Residual squares'
        self._extracted_from_arch_diag_8('Date', 'Residual squares')
        plt.legend([])
        plt.title(title, fontsize=20)
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100)

        # LB test
        q, p = acorr_ljungbox(resid2)
        with plt.style.context('ggplot'):
            fig = plt.figure(figsize=(20, 10))
            axes = fig.subplots(1, 2)
            axes[0].plot(q, label='Q')
            axes[0].tick_params(axis='x',labelsize=20)
            axes[0].tick_params(axis='y',labelsize=20)
            axes[0].set_ylabel('Q', fontsize=20)
            axes[0].set_xlabel('Qrder', fontsize=20)
            axes[1].plot(p, label='p')
            axes[1].tick_params(axis='x',labelsize=20)
            axes[1].tick_params(axis='y',labelsize=20)
            axes[1].set_ylabel('P', fontsize=20)
            axes[1].set_xlabel('Qrder', fontsize=20)
            axes[0].legend(fontsize=20)
            axes[1].legend(fontsize=20)
            plt.tight_layout()
            title = f'{self.part} {self.col}的GARCH检验图'
            file = os.path.join(self.path, f'{title}.png')
            plt.savefig(file, dpi=100) 

        if max(p) < .05:
            print(f'There exists strong ARCH effect for {self.col}')
        else:
            print(f'There does not exist strong ARCH effect for {self.col}')
        return max(p)

    # TODO Rename this here and in `arch_diag`
    def _extracted_from_arch_diag_8(self, arg0, arg1):
        plt.xlabel(arg0, fontsize=20)
        plt.ylabel(arg1, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

class garch_auto:
    def __init__(self, data, col, order, part, path, dist='normal', n=20):
        self.data = data
        self.col = col
        self.part, self.path = part, path
        self.arch = self.fit(order, dist=dist)
        self.n = n
        self.arch_pred = self.predict_garch(self.arch, n=self.n)

    def fit(self, order, dist):
        garch = arch_model(
            self.data, 
            mean='AR', 
            lags=order[0], 
            vol='GARCH', 
            p=1, 
            q=1, 
            dist=dist
        ).fit()
        
        self.condition_vol = pd.Series(
            garch.conditional_volatility,
            index=self.data.index
        )
        temp = pd.concat(
            [self.condition_vol, self.data], axis=1
        )
        temp.columns = ['conditional volatility', 'real values']
        print(garch.summary)

        # show the plot
        plt.figure(figsize=(20, 10))
        colormaps = {
                'conditional volatility':'blue',
                'real values':'red'
            }
        for col in temp.columns:
            temp[col].plot(
                style=colormaps.get(col)
            )
        plt.legend(loc=1, fontsize=20)
        title = f'{self.part} {self.col}波动率分解图'
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Volatility', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(title, fontsize=20)
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100) 
        return garch

    #predict 
    def predict_garch(self, garch, n=100, ar_p=1):
        params = garch.params.values
        conditional_volatility = garch.conditional_volatility
        garch_ret = []
        bef_ret = np.array(self.data[-(n + ar_p):])
        a = np.array(params[1:ar_p + 1])
        w = a[::-1]
        for i in range(n):  
            fit = params[0] + w.dot(bef_ret[i:ar_p + i])  
            garch_ret.append(fit)

        bef2 = [conditional_volatility.values[-1]]
        for _ in range(n):
            alpha0 = params[ar_p + 1]
            alpha1 = params[ar_p + 2]
            beta1 = params[ar_p + 3]
            new = alpha0 + (alpha1 + beta1) * (bef2[-1])
            bef2 = np.append(bef2, new)
        garch_vol_pre = bef2[-n:]

        # summary the data into one df
        index = pd.date_range(
            start=pd.to_datetime(self.data.index[0]),
            periods=n + len(self.data)
        )

        # add NA so that the length matches
        condition_vol = self.condition_vol.to_list() + [np.nan] * n
        real_value = self.data.to_list() + [np.nan] * n
        predicted_volatility = [np.nan] * len(self.data) + garch_vol_pre.tolist()

        # summarize the data
        predict = pd.concat(
            [
                pd.Series(condition_vol),
                pd.Series(real_value),
                pd.Series(predicted_volatility),
            ], axis=1
        )
        predict.index = index
        predict.columns = [
                'conditional volatility', 'real values', 'predicted_volatility'
            ]

        # plot the dataset
        plt.figure(figsize=(20, 10))
        colormaps = {
                'conditional volatility':'blue',
                'real values':'red',
                'predicted_volatility':'green'
            }
        for col in predict.columns:
            predict[col].plot(
                style=colormaps.get(col)
            )
        plt.legend(loc=1, fontsize=20)
        title = f'{self.part} {self.col}向前{n}步波动率预测图'
        plt.xlabel('Date', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(title, fontsize=20)
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100)
        return predict

class VAR:
    def __init__(self, df, arima_dict, part, path):
        self.df = df.astype(float)
        self.cols = ','.join(list(self.df.columns))
        self.arima_dic = arima_dict
        self.part, self.path = part, path
        self.model, self.model_res = self.fit()

    def fit(self):
        cols = self.df.columns
        varLagNum = [self.arima_dic.get(col).arima_order[0] for col in cols]
        orgMod = sm.tsa.VARMAX(self.df, order=varLagNum,trend='c',exog=None)
        fitMod = orgMod.fit(maxiter=200,disp=True)
        print(fitMod.summary())
        resid = fitMod.resid
        result = {'fitMod':fitMod,'resid':resid}
        return fitMod, result

    def Test_Cusum(self):
        result = statsmodels.stats.diagnostic.\
            breaks_cusumolsresid(self.model_res.get('resid'))
        if result[1] > .05:
            print('There is no drifting')
        else:
            print('There is drifting')
        return result

    def impulse_responses(self):
        title = f'{self.part} {self.cols} VAR模型的脉冲响应分析'
        ax = self.model.impulse_responses(20, orthogonalized=True).\
            plot(figsize=(20, 10))
        plt.xlabel('Prediction length', fontsize=20)
        plt.ylabel('Sum of duration', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(title, fontsize=20)
        plt.legend(loc=1, fontsize=20)
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100) 


    def variance_decompositon(self):
        md = sm.tsa.VAR(self.df.values)
        re = md.fit(2)
        fevd = re.fevd(10)
        print(fevd.summary())
        fevd.plot(figsize=(20, 10))
        plt.legend(list(self.df.columns), fontsize=20)
        title = f'{self.part} {self.cols} VAR模型的方差分解'
        file = os.path.join(self.path, f'{title}.png')
        plt.savefig(file, dpi=100)

def MyNeuralProphet(data, col, NP_dict, test_length, part, path):
    nprophet_model = NeuralProphet()
    temp = data.rename(columns={'date':'ds', col: 'y'})
    metrics = nprophet_model.fit(temp[['ds', 'y']], freq="D")
    future_df = nprophet_model.make_future_dataframe(
        temp[['ds', 'y']], 
        periods = test_length, 
        n_historic_predictions=len(temp[['ds', 'y']])
    )
    preds_df_2 = nprophet_model.predict(future_df)
    title = f'{part} {col}的NeuralProphet时间序列预测图'
    nprophet_model.plot(
        preds_df_2, 
        ylabel=col, 
        figsize=(20, 10),
    );
    plt.xlabel('Date', fontsize=20)
    plt.ylabel(col, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=20)
    file = os.path.join(path, f'{title}.png')
    plt.savefig(file, dpi=100)
    NP_dict[col] = nprophet_model, preds_df_2