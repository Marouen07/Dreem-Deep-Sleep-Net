"""Pytorch Models USed """
import torch
import torch.nn as nn
class Feature_extractor(nn.Module):
  def __init__(self, n_measures,fs, drop=.5,seq_l=30,features=False):
    super().__init__()
    d=64
    self.measures=n_measures
    self.features=features
    #Extracting Temporal Features
    self.temporal = nn.Sequential(
        nn.Conv1d(self.measures,64, fs//2+1,fs//8),nn.BatchNorm1d(64),nn.ReLU(inplace=True),nn.MaxPool1d(8,stride=8),nn.Dropout(drop),
        nn.Conv1d(64,d, 8,stride=1),nn.BatchNorm1d(d),nn.ReLU(inplace=True),nn.Dropout(drop),
        nn.Conv1d(d,d, 8,stride=1),nn.BatchNorm1d(d),nn.ReLU(inplace=True),nn.Dropout(drop),
        nn.Conv1d(d,d, 8,stride=1),nn.BatchNorm1d(d),nn.ReLU(inplace=True),
        nn.Dropout(drop),nn.MaxPool1d(4,stride=4))
    #following the guidelines in Pytorch Docs To calculate Length
    self.temp_len=int(int(((seq_l*fs-fs//2-1)/(fs//8))+1)/8) #first layer of Conv
    self.temp_len= int((self.temp_len-(8-1)*3)/4)
    #Extracting Frequency Related Features
    self.freq = nn.Sequential(
        nn.Conv1d(self.measures,64, fs*4,stride=fs//3),nn.BatchNorm1d(64),nn.ReLU(inplace=True),nn.MaxPool1d(4,stride=4),nn.Dropout(drop),
        nn.Conv1d(64,d, 6,stride=1),nn.BatchNorm1d(d),nn.ReLU(inplace=True),nn.Dropout(drop),
        nn.Conv1d(d,d, 6,stride=1),nn.BatchNorm1d(d),nn.ReLU(inplace=True),nn.Dropout(drop),
        nn.Conv1d(d,d, 6,stride=1),nn.BatchNorm1d(d),nn.ReLU(inplace=True),
        nn.Dropout(drop),nn.MaxPool1d(2,stride=2))
    #following the guidelines in Pytorch Docs To calculate Length
    self.freq_len=int(int(((seq_l*fs-fs*4)/(fs//3))+1)/4) #first layer of Conv
    self.freq_len= int((self.freq_len-(6-1)*3)/2)
    self.n_features=(self.freq_len+self.temp_len)*d
    self.lin=nn.Sequential(nn.Linear(self.n_features,5))
  def forward(self, input):
    x_t = self.temporal(input)
    x_f=self.freq(input)
    #print(x_t.shape,'  ',x_f.shape)
    x_t=x_t.view(x_t.size(0), -1)
    x_f=x_f.view(x_f.size(0), -1)
    input=torch.cat([x_f, x_t], dim=1)
    if self.features:
      return(input)
    else:
      return self.lin(input)



"""----------------------------------------------------------------------------------------------------------------------------"""



class Seq_learn(nn.Module):
  def __init__(self,hidden_size,Residual=True,n_layers=2,lookback=3, drop=.5):
    super().__init__()
    self.Residual=Residual
    self.lookback=lookback
    self.n_layers=n_layers
    self.hidden_size=hidden_size
    self.lstm=nn.LSTM(input_size=self.hidden_size,hidden_size=self.hidden_size,
                num_layers=self.n_layers,dropout=drop,batch_first=True,
                bidirectional =True)
    self.lin=nn.Sequential(nn.Dropout(drop),nn.Linear(3*self.hidden_size*self.lookback,500),
                           nn.BatchNorm1d(500),nn.ReLU(inplace=True),nn.Dropout(drop),
                           nn.Linear(500,5)
                           )
    self.post_lstm=nn.Sequential(nn.Linear(2*self.hidden_size*self.lookback,5))
  def init_hidden(self, batch_size,device):
    # even with batch_first = True this remains same as docs
    hidden_state = torch.zeros(self.n_layers*2,batch_size,self.hidden_size)
    cell_state = torch.zeros(self.n_layers*2,batch_size,self.hidden_size)
    self.hidden = (hidden_state.to(device), cell_state.to(device))
  def forward(self, input):
    #print(input.shape)
    lstm_out,self.hidden=self.lstm(input,self.hidden)
    #print(lstm_out.shape)
    lstm_out=lstm_out.contiguous().view(lstm_out.size(0), -1)
    if self.Residual :
      residual=torch.cat([input.reshape(input.shape[0],-1),lstm_out],dim=1)
      #print(residual.shape)
      return self.lin(residual)
    else:
      return self.post_lstm(lstm_out)
