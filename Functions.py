"""Auxiliary Functions Used in our model"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def cuda():
  # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
  is_cuda = torch.cuda.is_available()
  # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
  if is_cuda:
      device = torch.device("cuda")
      print("GPU is available")
  else:
      device = torch.device("cpu")
      print("GPU not available, CPU used")
  return device


"""----------------------------------------------------------------------------------------------------------------------------"""



def Create_dataset(measures,f):
  """getting data from h5py file
    Parameters:
    -------------
    measures: List of sensors to take data from
    f: H5py file containing the dataset ( memory efficient)

    Output :
    -------------
    df : DataFrame containing the formatted data
    y  : Dataframe of corrresponding Labels
    """
  keys=list(f.keys())
  indexes=[keys[7]]+[keys[9]]
  df=pd.concat([pd.DataFrame(f[i][:]) for i in measures],ignore_index=True)
  #Standardizing Input
  df=(df.sub(df.mean(axis=1),axis=0)).div(df.std(axis=1),axis=0)#(df.max(axis=1)-df.min(axis=1))
  #reformatting the data so that we can feed it to our Neural Network Later on
  n_measures=len(measures)
  n_samples=df.shape[0]//n_measures
  df['id']=df.index%n_samples
  df['channel']=[measures[i] for i in df.index//n_samples]
  df_indexes=pd.concat([pd.DataFrame(f[i][:]) for i in indexes],ignore_index=True,axis=1)
  df[indexes]=df_indexes.loc[df['id']].reset_index(drop=True)
  df=df.set_index(['id','channel']+indexes)
  df=df.sort_index()
  return df,n_measures,n_samples

"""----------------------------------------------------------------------------------------------------------------------------"""


def create_loader(X,y,batch_size,shuffle=True):
  """Creating DataLoaders to feed our NN with Data
  """
  data=TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
  loader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
  return loader


"""----------------------------------------------------------------------------------------------------------------------------"""



def Oversample(df,y,shape,split=.2):
  """Oversampling Data to avoid Class inbalance issue
  Parmaeters :
  ------------
  df: dataset
  y: Labels
  shape: for 1d Conv (n_samples,N_channels(=N_measures),Length(Sampling_Frequency*Duration))
         for 2d Conv (n_samples,N_channels(=1),N_measures,Length(Sampling_Frequency*Duration))
  Output:
  -----------
  oversampled splitted data (Train and Validation)
  """
  X=df.values.reshape(shape[0],-1)
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split)
  X_resampled, y_resampled = SMOTE(n_jobs=-1).fit_resample(X_train, y_train)
 #Adapting Data Shape
  X_resampled=X_resampled.reshape((X_resampled.shape[0],)+shape[1:])
  X_val=X_val.reshape((X_val.shape[0],)+shape[1:])
  return X_resampled,y_resampled,X_val,y_val

"""----------------------------------------------------------------------------------------------------------------------------"""



def Train_model(model,criterion,optimizer,loader,scheduler,device,mode='train',hidden=False):
  """Model Training/Validation
  ------------------------------
  Output:
  -----------
  mean_loss: Mean loss over Epoch"""

  if mode=='train':
    model.train()
  elif mode=='val':
    model.eval()
  else:
    print('mode should be Train or Val')
    return
  losses=[]
  Preds=[]
  Actual=[]
  correct = 0
  total = 0
  for inputs, labels in loader:
    inputs, labels = inputs.to(device), labels.to(device)
    if hidden:
      model.init_hidden(inputs.size(0),device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(mode=='train'):
      output = model(inputs)
      loss = criterion(output, labels)
      _,pred=torch.max(output, 1)
      if mode=='train':
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    Preds=np.append(Preds,pred.cpu().long().numpy())
    Actual=np.append(Actual,labels.cpu().long().numpy())
    correct+=float(torch.sum(pred==labels.data))
    total+=float(inputs.shape[0])

  if mode =='train':
    scheduler.step()
  mean_loss=np.mean(losses)
  print('{} Loss {:.4f}  Acc : {:.2%}  f1 score {:.2f}'.format(mode,mean_loss,correct/total,f1_score(Actual,Preds,average='weighted')))
  return mean_loss


"""----------------------------------------------------------------------------------------------------------------------------"""


def  seperate(data,shape_eeg,shape_pul,lookback):
  """Auxilary Function used to seperate eeg data from
  pulsometer/accelerometer data due to the difference in
  sampling frequencies (10Hz/50Hz)"""
  n1=shape_eeg[1]*shape_eeg[2]
  n2=shape_pul[1]*shape_pul[2]
  data_pul=data[:,:,n1:n1+n2].reshape((-1,lookback)+shape_pul[1:])
  data_eeg=data[:,:,:n1].reshape((-1,lookback)+shape_eeg[1:])
  return data_eeg,data_pul

"""----------------------------------------------------------------------------------------------------------------------------"""


def Create_Sequential_data(df,lookback,n_measures,labels=np.array([])):
  dataX=[]
  dataY=[]
  for i in df.index.get_level_values('index').unique():
    df_patient=df.xs(i,level='index')
    patient_samples=df_patient.shape[0]//n_measures
    sample_id=df_patient.index.get_level_values('id').unique()
    if len(labels)>0 :
      dataY=np.append(dataY,labels[sample_id][lookback-1:])
    else:
      dataY=np.append(dataY,sample_id[lookback-1:])
    X_train=df_patient.values.reshape(patient_samples,n_measures,-1)
    for t in range(lookback,patient_samples+1):
      dataX+=[X_train[t-lookback:t]]
  dataX=np.array(dataX)
  return dataX,dataY


"""----------------------------------------------------------------------------------------------------------------------------"""



def Train_seq_model(models,criterion,optimizer,loader,scheduler,shape_eeg,shape_pul,device,lookback,mode='train'):
  """Model Training/Validation
  ------------------------------
  Output:
  -----------
  mean_loss: Mean loss over Epoch"""
  cl_eeg,cl_pul,seq=models
  cl_eeg.features=True
  cl_pul.features=True
  if mode=='train':
    for model in models:
      model.train()
  elif mode=='val':
    for model in models:
      model.eval()
  else:
    print('mode should be Train or Val')
    return
  losses=[]
  Preds=[]
  Actual=[]
  correct = 0
  total = 0
  for inputs, labels in loader:
    input_eeg,input_pul=seperate(inputs,shape_eeg,shape_pul,lookback)
    input_eeg=input_eeg.reshape((-1,)+shape_eeg[1:])
    input_pul=input_pul.reshape((-1,)+shape_pul[1:])
    input_eeg,input_pul,labels = input_eeg.to(device),input_pul.to(device),labels.to(device)
    seq.init_hidden(inputs.size(0),device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(mode=='train'):
      output_eeg = cl_eeg(input_eeg).reshape(inputs.size(0),lookback,-1)
      #print(output_eeg.shape)
      ouput_pul=cl_pul(input_pul).reshape(inputs.size(0),lookback,-1)
      #print(ouput_pul.shape)
      output=seq(torch.cat([output_eeg,ouput_pul],dim=2))
      #print(output.shape)
      loss = criterion(output, labels)
      _,pred=torch.max(output, 1)
      if mode=='train':
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    Preds=np.append(Preds,pred.cpu().long().numpy())
    Actual=np.append(Actual,labels.cpu().long().numpy())
    correct+=float(torch.sum(pred==labels.data))
    total+=float(inputs.shape[0])

  if mode =='train':
    scheduler.step()
  mean_loss=np.mean(losses)
  print('{} Loss {:.4f}  Acc : {:.2%}  f1 score {:.2f}'.format(mode,mean_loss,correct/total,f1_score(Actual,Preds,average='weighted')))
  return mean_loss


"""----------------------------------------------------------------------------------------------------------------------------"""

def Test_seq_model(models,batch_size,f,measures_eeg,measures_pul,F_s,F_s_pul,lookback,w_eeg,w_pul,device):
  df_eeg,n_measures_eeg,n_samples=Create_dataset(measures_eeg,f)
  df_pul,n_measures_pul,n_samples=Create_dataset(measures_pul,f)
  dataX_eeg,dataY=Create_Sequential_data(df_eeg,lookback=lookback,n_measures=n_measures_eeg)
  dataX_pul,dataY=Create_Sequential_data(df_pul,lookback=lookback,n_measures=n_measures_pul)
  data=np.concatenate((dataX_eeg.reshape(dataX_eeg.shape[0:2]+(-1,)),dataX_pul.reshape(dataX_pul.shape[0:2]+(-1,))),axis=2)
  Seq_test_loader=create_loader(data,dataY,batch_size=batch_size,shuffle=False)

  shape_eeg=(n_samples,n_measures_eeg,F_s*30)
  shape_pul=(n_samples,n_measures_pul,F_s_pul*30)
  Preds=pd.DataFrame(index=df_eeg.index.get_level_values('id').unique().rename('index'),columns=['sleep_stage'])
  cl_eeg,cl_pul,seq=models
  cl_eeg.features=True
  cl_pul.features=True
  for model in models:
    model.eval()

  for inputs,indices in Seq_test_loader:
    input_eeg,input_pul=seperate(inputs,shape_eeg,shape_pul,lookback)
    input_eeg=input_eeg.reshape((-1,)+shape_eeg[1:])
    input_pul=input_pul.reshape((-1,)+shape_pul[1:])
    input_eeg,input_pul = input_eeg.to(device),input_pul.to(device)
    seq.init_hidden(inputs.size(0),device)
    with torch.no_grad():
      output_eeg = cl_eeg(input_eeg).reshape(inputs.size(0),lookback,-1)
      #print(output_eeg.shape)
      ouput_pul=cl_pul(input_pul).reshape(inputs.size(0),lookback,-1)
      #print(ouput_pul.shape)
      output=seq(torch.cat([output_eeg,ouput_pul],dim=2))
      #print(output.shape)
      _,pred=torch.max(output, 1)
    Preds.loc[indices]=pred.cpu().reshape(-1,1)

  #save model parameters
  params=[model.state_dict() for model in models]
  #Completing the missing predictions here: predicting from features only
  cl_eeg.features=False
  #cl_pul.features=False
  cl_eeg.load_state_dict(w_eeg)
  #cl_pul.load_state_dict(w_pul)

  for indice in Preds[Preds.isna()['sleep_stage']].index :
    input=torch.from_numpy(df_eeg.xs(indice,level='id').values[np.newaxis,:,:]).to(device)
    with torch.no_grad():
      output = cl_eeg(input)
      _,pred=torch.max(output, 1)
      Preds.loc[indice]=pred.item()
  Preds.index+=24688
  #getting back the old model state
  cl_eeg.features=True
  cl_pul.features=True
  for i,model in enumerate(models):
    model.load_state_dict(params[i])
  return Preds

"""----------------------------------------------------------------------------------------------------------------------------"""