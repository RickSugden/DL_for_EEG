import torch
from glob import glob
import glob
from torch.utils.data import Dataset, TensorDataset
import os
import pandas as pd
from sys import platform

class EEGDataset(Dataset):

  def __init__(self, data_path, chunk_size=2500):
    """
        Args:
            data_path (string): Directory with the EEG training data. Filenames in this dir must begin with "PD" or "Control" to assign labels correctly. Must be in .csv files. 
            chunk_size (int): Number of datapoints from EEG time series to be included in a single non-overlapping epoch. Note that UNM data was collected at 500Hz.
    """
    #create containers for the data and labels respectively
    self.df_list = []
    self.label_list = []
    self.chunk_size=chunk_size
    #create a list of datafields to keep. The electrodes given here are those in common to both the UI and UNM datasets.
    self.common_electrodes = ['time', 'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'POz', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
    if data_path[-4:] != '.csv':

      #create the datapaths
      self.data_path = data_path  
      self.data_list = glob.glob(self.data_path + "*.csv")
      self.files = glob.glob(self.data_path+ '*.csv')


    else:
      self.files = [data_path]

    for file in self.files:
      
      if os.path.isfile(file):
        csv = pd.read_csv(file,sep=',', index_col=[0]) #load in single session as a csv
        csv.drop(index=0, inplace=True, axis=0) #drop first row because it's usually noisy
        csv = csv[self.common_electrodes]             #select the subset of electrodes defined by self.common_electrodes
        csv.drop('time', inplace=True, axis=1) #drop time so it is not considered as a variable

        #csv is then segmented into epochs. Each epochs is added as a df to the list of data with a corresponding list of labels (at this point the whole filename is given as the label).
        for chunk in range(1,csv.shape[0]//chunk_size +1):
            start = (chunk-1)*chunk_size
            stop = chunk*chunk_size
            self.df_list.append(csv.iloc[start:stop])
            self.label_list.append(file)

    print('there are this many items in the list of data ' ,len(self.df_list))  
    print('there are this many items in the list of labels ' , len(self.label_list))

    #define the labels as vectors to match training
    self.class_map = {"CTL" : [0, 1], "PD": [1, 0]} 
    
    #Normalize each channel of each epoch to a mean of 0 and std of 1.
    self.all_data = self.df_list[0]
    
    self.normalized_df_list = []
    
    #iterate through each epoch
    for df_index in range(0,len(self.df_list)):
      
      temp_df = self.df_list[df_index]
      mean_by_channel = []
      std_by_channel = []
      
      #determine normalization parameters by column (i.e. for each channel)
      for column in temp_df:
        mean_by_channel.append(temp_df[column].mean())
        std_by_channel.append(temp_df[column].std())

      #apply normalization
      temp_df = temp_df.sub(mean_by_channel, axis='columns')
      temp_df = temp_df.div(std_by_channel, axis='columns')
      self.normalized_df_list.append(temp_df)

    assert (len(std_by_channel)== len(mean_by_channel)), 'length of mean normalization and std normalization are not same length'
    print('The length of the lists of channels means and stds is ', len(mean_by_channel))
    assert ((self.normalized_df_list[0].shape)==(self.df_list[0].shape) and (len(self.normalized_df_list)==len(self.df_list))), 'Normalization changed the shape of the df_list'
      

  #this is a required function that tells you how many data points you have in the dataset
  def __len__(self):
      return len(self.normalized_df_list) 

  #this is a required function that allows you to obtain a single data point according to its index
  def __getitem__(self, idx):
    
    #each dataframe represents one block id
    eeg_dataframe = self.normalized_df_list[idx]
    
    #determine whether that subject is control or PD
    from sys import platform
    if platform == 'win32':
        #filename = os.path.normpath(self.label_list[idx]).split(os.path.sep)[-1]
        filename = self.label_list[idx].split('\\')[-1]
    elif platform=="linux" or platform =="linux2":
        filename = self.label_list[idx].split('/')[-1]
    elif platform.startswith('darwin'):
        filename = self.label_list[idx].split('/')[-1]
    
    
    # this is the string containing the filename. index first 2 chars to see if PD.
    if filename[0:2]=='PD':
      PD_label = 'PD'
    # then see if its control
    elif filename[0:3]=='CTL':
      PD_label = 'CTL'
    # if neither, throw an error
    else:
      print(filename[0:5])
      print(filename[0:8])
      assert False, 'there is a problem finding the label'

    #convert label to tensor using class map
    PD_label = torch.tensor(self.class_map[PD_label], dtype=torch.float32)   
    #reformat the eeg data
    eeg_tensor = torch.tensor(eeg_dataframe[0:self.chunk_size].values) #you can artificially shorten epochs here
    eeg_tensor = torch.permute(eeg_tensor,(1, 0))
    
    return eeg_tensor.to(torch.float32), PD_label.to(torch.float32), filename


def make_data_into_tensor(data_path, chunk_size=2500, device='cpu'):
    """
        Args:
            data_path (string): Directory with the EEG training data. Filenames in this dir must begin with "PD" or "Control" to assign labels correctly. Must be in .csv files. 
            chunk_size (int): Number of datapoints from EEG time series to be included in a single non-overlapping epoch. Note that UNM data was collected at 500Hz.
    """
    #create containers for the data and labels respectively
    df_list = []
    label_list = []
    chunk_size=chunk_size
    #create a list of datafields to keep. The electrodes given here are those in common to both the UI and UNM datasets.
    common_electrodes = ['time', 'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'POz', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
    if data_path[-4:] != '.csv':

      #create the datapaths
      data_list = glob.glob(data_path + "*.csv")
      files = glob.glob(data_path+ '*.csv')


    else:
      files = [data_path]

    for file in files:
      
      if os.path.isfile(file):
        csv = pd.read_csv(file,sep=',', index_col=[0]) #load in single session as a csv
        csv.drop(index=0, inplace=True, axis=0) #drop first row because it's usually noisy
        csv = csv[common_electrodes]             #select the subset of electrodes defined by common_electrodes
        csv.drop('time', inplace=True, axis=1) #drop time so it is not considered as a variable

        #csv is then segmented into epochs. Each epochs is added as a df to the list of data with a corresponding list of labels (at this point the whole filename is given as the label).
        for chunk in range(1,csv.shape[0]//chunk_size +1):
            if chunk ==0:
              start = 0
              stop = chunk_size
            else:
              start = (chunk-1)*chunk_size
              stop = chunk*chunk_size

            #start = (chunk-1)*chunk_size
            #stop = chunk*chunk_size
            df_list.append(csv.iloc[start:stop])
            if platform=='win32':
              label_list.append(file.split('\\')[-1])
            else:
              label_list.append(file.split('/')[-1])

    print('there are this many items in the list of data ' ,len(df_list))  
    print('there are this many items in the list of labels ' , len(label_list))

    #define the labels as vectors to match training
    class_map = {"CTL" : [0], "PD_": [1]} 
    
    #Normalize each channel of each epoch to a mean of 0 and std of 1.
    all_data = df_list[0]
    
    normalized_df_list = []
    
    #iterate through each epoch
 
    for df_index in range(0,len(df_list)):
      
      temp_df = df_list[df_index]
      mean_by_channel = []
      std_by_channel = []
      
      #determine normalization parameters by column (i.e. for each channel)
      for column in temp_df:
        mean_by_channel.append(temp_df[column].mean())
        std_by_channel.append(temp_df[column].std())

      #apply normalization
      temp_df = temp_df.sub(mean_by_channel, axis='columns')
      temp_df = temp_df.div(std_by_channel, axis='columns')
      normalized_df_list.append(temp_df)

    assert (len(std_by_channel)== len(mean_by_channel)), 'length of mean normalization and std normalization are not same length'
    print('The length of the lists of channels means and stds is ', len(mean_by_channel))
    assert ((normalized_df_list[0].shape)==(df_list[0].shape) and (len(normalized_df_list)==len(df_list))), 'Normalization changed the shape of the df_list'
      
    # unpack normalized_df_list into a tensor where the first dimension represents sample number
    for i, (data, label) in enumerate(zip(normalized_df_list, label_list)):

      #add the array to the tensor and the label to the label tensor
      if i == 0:
        assert data.shape[0] == chunk_size, 'The chunk size is not correct on the first sample'
        dataset_tensor = torch.tensor(data.values)
        dataset_tensor = torch.unsqueeze (dataset_tensor, 0)
        label_tensor = torch.tensor(class_map[label[0:3]]).float()
        subject_tensor = torch.tensor([float(label.split('.')[0].split('_')[-1])], dtype=torch.float32)
        subject_tensor = torch.unsqueeze(subject_tensor,0)
      else:
        data_tensor = torch.unsqueeze(torch.tensor(data.values), 0)

        #confirm the shape is correct before adding:
        if data_tensor.shape[1::] != dataset_tensor.shape[1::]:
          print('skipped a sample because the shape was wrong')
        else:
          dataset_tensor = torch.cat(tensors=(dataset_tensor, data_tensor), dim=0)
          label_tensor = torch.cat((label_tensor, torch.tensor(class_map[label[0:3]], dtype=torch.float32)), dim=0)
          subject_tensor = torch.cat((subject_tensor, torch.unsqueeze(torch.tensor([float(label.split('.')[0].split('_')[-1])]),dim=0)), dim=0)

    print('The final shape of the tensor dataset is ',  dataset_tensor.shape)
    print('The final shape of the tensor labels is ',  label_tensor.shape)
    
    ##permute the last two dims of the dataset_tensor
    dataset_tensor = torch.permute(dataset_tensor, (0, 2, 1)).float()
    label_tensor = label_tensor.float()
    subject_tensor = subject_tensor.float()
    #print(dataset_tensor)
    #create a tensor dataset
    if device.startswith('cuda'): print('moving tensor dataset to gpu')
    #this was the line causing everything to go to zero #tensor_dataset = TensorDataset(dataset_tensor.to(device), label_tensor.to(device), subject_tensor.to(device))
    tensor_dataset = TensorDataset(dataset_tensor, label_tensor, subject_tensor)
    return tensor_dataset, dataset_tensor, label_tensor, subject_tensor