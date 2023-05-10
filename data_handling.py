'''
This file is where we will put dataset and dataloader classes for the EEG data. note that the current dataset class is only for clinical EEG
so we will need to write a new one for wearable EEG or make this one dynamic--your choice. 

specs:
- dataset class should be a subclass of torch.utils.data.Dataset
- dataloader class should be a subclass of torch.utils.data.DataLoader
- dataset class should have a __len__ method that returns the length of the dataset
- dataset class should have a __getitem__ method that returns a tuple of (data, label) for the index given
- note that datasets can return more than just data and label, if needed to keep track of things like subject name
- we are assuming that the input is a folder containing all the files and that the filenames start with either PD or CTL 
so that is what gets used for the label
- for the 60-channel EEG call the class "sixty_channel_EEG_dataset" and for the 4-channel EEG call the class "four_channel_EEG_dataset"
when they were originally written, no thought was given and EEGDataset is too vague to be useful going forward. Don't worry about making the 4channel just yet
we will get there eventually. for I just want the 60 channel working from github.
- 
'''


#Load In the Dataset
class EEGDataset(Dataset):

  def __init__(self, data_path, chunk_size=1000):
    """
        Args:
            data_path (string): Directory with the EEG training data. Filenames in this dir must begin with "PD" or "Control" to assign labels correctly. Must be in .csv files. 
            chunk_size (int): Number of datapoints from EEG time series to be included in a single non-overlapping epoch. Note that UNM data was collected at 500Hz.
    """
    #create containers for the data and labels respectively
    self.df_list = []
    self.label_list = []

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
    filename = self.label_list[idx].split('/')[-1]
    
    # this is the string containing the filename. index first 2 chars to see if PD.
    if filename[0:2]=='PD':
      PD_label = 'PD'
    # then see if its control
    elif filename[0:3]=='CTL':
      PD_label = 'CTL'
    # if neither, throw an error
    else:
      print(filename[0:2])
      print(filename[0:7])
      assert False, 'there is a problem finding the label'

    #convert label to tensor using class map
    PD_label = torch.tensor(self.class_map[PD_label], dtype=torch.long)
    
    #reformat the eeg data
    eeg_tensor = torch.tensor(eeg_dataframe[0:chunk_size].values) #you can artificially shorten epochs here
    eeg_tensor = torch.permute(eeg_tensor,(1, 0))
    
    return eeg_tensor.float(), PD_label.float(), filename