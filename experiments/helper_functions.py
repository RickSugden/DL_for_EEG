#Imports
#imports
import importlib
import glob
import importlib
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from experiments.models import PD_CNN
import torch
import experiments.training_and_validation as training_and_validation
from experiments.training_and_validation import loso_cross_validation, train, validate, cross_train, UNM_train, UNM_validate, UNM_testing_subject
from experiments.training_and_validation import with_leak_validate, testing_epoch_validate, with_leak_train
from experiments.result_visualization import run_chi_squared_test, plot_confusion_matrix, plot_roc_auc, plot_confusion_matrix, make_combo_plot
from experiments.result_visualization import waterfall_plot, sequence_plot, wise_roc_curve, plot_roc_matrix, plot_both
from torch.utils.data import DataLoader, random_split
import experiments.data_handling as data_handling
import experiments.models as models
import experiments.result_visualization as result_visualization

#Data Loader Function
def load_dataset(data_src, batch_size = 8, num_workers = 2, chunk_size=2500):
  #locate the raw data

  ############ create list of subject numbers to leave out ###############################
  files = glob.glob(data_src + '*.csv')
  leave_one_out_list = []
  for file in files:  
    leave_one_out_list.append(file.split('/')[-1])#.split('_')[1]) #remove hashtags to return to UNM dataset

  ############# create dataset of all data ############################
  EEG_whole_Dataset = data_handling.EEGDataset(data_path=data_src, chunk_size=chunk_size)
  return EEG_whole_Dataset, leave_one_out_list


def data_loader(EEG_whole_Dataset, num_workers=2, batch_size=8):
    # DATA LOADER
    ################ CREATE DATALOADER  ############################################
    #define the train test split
    train_size = int(0.90 * len(EEG_whole_Dataset))
    val_size = len(EEG_whole_Dataset) - train_size
    train_dataset, val_dataset = random_split(EEG_whole_Dataset, [train_size, val_size],generator=torch.Generator().manual_seed(402))

    #create a respective dataloader out of the test/train split
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    print(next(iter(trainloader))[0].size())
    print('there are this many batches in the training dataloader:',len(trainloader))
    print(next(iter(valloader))[0].size())
    print('there are this many batches in the validation dataloader: ',len(valloader))
    #the trainlaoder has 91 batches, the valloader has 16 batches
    #1.5 Minute runtime
    return trainloader, valloader


def data_loader_val_only(EEG_whole_Dataset, num_workers=2, batch_size=8):
    # DATA LOADER
    ################ CREATE DATALOADER  ############################################
    #define the train test split
    train_size = int(0.90 * len(EEG_whole_Dataset))
    val_size = len(EEG_whole_Dataset) - train_size
    train_dataset, val_dataset = random_split(EEG_whole_Dataset, [train_size, val_size],generator=torch.Generator().manual_seed(402))

    #create a respective dataloader out of the test/train split
    # trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    # print(next(iter(trainloader))[0].size())
    # print('there are this many batches in the training dataloader:',len(trainloader))
    # print(next(iter(valloader))[0].size())
    # print('there are this many batches in the validation dataloader: ',len(valloader))
    #the trainlaoder has 91 batches, the valloader has 16 batches
    #1.5 Minute runtime
    return valloader


def confirm_model(device='cpu'):
    input_tensor = torch.rand([8,60,2500]).to(device) #of the form [batch_size, channels, epoch_length]
    print(input_tensor.size())
    network = PD_CNN().to(device)
    output_tensor = network(input_tensor)
    print((output_tensor.shape))


def visualize():
    plot_confusion_matrix(TP=790, FP=259, TN=752, FN=321, filename='subjectwise_matrix_epochs')
    x2, p = run_chi_squared_test(TP=790, FP=259, TN=752, FN=321)
    print(p)
def name_model(device='cpu', chunk_size=2500):
  ############# choose name for model--Load or create model ######################
  experiment_name = 'epoch_reduced_training' 
  model_folder = '/Users/rakan/ResearchPD/DL_for_EEG/saved_models/'
  PATH = model_folder + experiment_name 

  if (os.path.exists(PATH )):
    model = torch.load(PATH).to(device)
    print(' model has been successfully loaded')
    return model, PATH
  else:
    model = PD_CNN(chunk_size=chunk_size).to(device)
    log_containing_train_loss = []
    log_containing_val_loss = []
    print(' model has been successfully created')
    return model, PATH
def run_replicates(EEG_Dataset, num_workers=2, batch_size=8, chunk_size=2500, device='cpu'):
  # This ccell can be used to train a set of replicates they will all be saved to this model folder
  model_folder = '/Users/rakan/ResearchPD/DL_for_EEG/saved_models/'
  replicates = 10

  for i in range(replicates):
    #define the train test split
    train_size = int(0.90 * len(EEG_Dataset))
    val_size = len(EEG_Dataset) - train_size
    train_dataset, val_dataset = random_split(EEG_Dataset, [train_size, val_size],generator=torch.Generator().manual_seed(i))

    #create a respective dataloader out of the test/train split
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    ####################### CREATE OR LOAD MODEL ############################3
    experiment_name = 'epoch_reduced_training' + str(i) 
    PATH = model_folder + experiment_name
    if (os.path.exists(PATH)):
      model = torch.load(PATH).to(device)
      print(' model has been successfully loaded')
    else:
      model = PD_CNN(chunk_size=chunk_size).to(device)
      log_containing_train_loss = []
      log_containing_val_loss = []
      print('model ', i, ' has been successfully created')
    ##################################################################33
    model.train()
    model, log_containing_train_loss, log_containing_val_loss = UNM_train(model=model,train_dataloader=trainloader, val_dataloader=valloader, epochs=30, learning_rate=0.0001, training_loss_tracker=[], val_loss_tracker=[])  

    ##############################################################
    torch.save(model, PATH)
def UNM_val(model, valloader, batch_size=8):
  #run validation
  model.eval()

  #validate using a single threshold
  #validate(model=model,valloader=valloader,threshold=0.5,batch_size=batch_size)

  #validate using a list of thresholds
  for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    print('Validating using threshold = ', threshold)
    UNM_validate(model=model,valloader=valloader,threshold=threshold,batch_size=batch_size)
def UNM_visualize_1(log_containing_train_loss, log_containing_val_loss):
    #Print Learning Curves
    training_loss = sns.lineplot(x = np.arange(0,len(log_containing_train_loss)), y = log_containing_train_loss) 
    _ = training_loss.set_xlabel('Epoch',size=14)
    _ = training_loss.set_ylabel('Training Loss',size=14)
    plt.show()
    validation_loss = sns.lineplot(x = np.arange(0,len(log_containing_val_loss)), y = log_containing_val_loss) 
    _ = validation_loss.set_xlabel('Epoch',size=14)
    _ = validation_loss.set_ylabel('Validation Loss',size=14)
    plt.show()
def UNM_visualize_2(log_containing_train_loss, log_containing_val_loss):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(np.arange(0,len(log_containing_train_loss)), log_containing_train_loss, linestyle='--',color='blue')

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(np.arange(0,len(log_containing_val_loss)), log_containing_val_loss, linestyle='--',color='red')

    # Save the full figure...
    fig.savefig('full_figure.png')
def subject_list():
  ############ create list of subject_numbers ###############################
  chunk_size=2500
  data_src = '/Users/rakan/ResearchPD/DL_for_EEG/Data/UI/all_data_reref_bandpass_1_to_45/' 
  files = glob.glob(data_src + '*.csv')
  subject_list = []
  filename_list = []
  for file in files:  
    filename = file.split('/')[-1] #remove all preceeding directories
    
    filebasename = filename.split('.')[0] #drop the .csv
    
    subject_number = filebasename[-4:] #last four will be the subject number
    
    subject_list.append(subject_number)
    filename_list.append(filename)
  return subject_list, filename_list # fix this later
def epoch_level_model(device='cpu', chunk_size=2500):
  ############# choose name for model--Load or create model ######################
  #choose experiment name to match filename. Choose model folder to the model files location
  experiment_name = 'UNM_training_replicates_redo_3' 
  #model_folder = '/content/drive/MyDrive/UTOR-MSc/colab_notebooks/model_weights/'
  model_folder = '/Users/rakan/ResearchPD/DL_for_EEG/saved_models/'
  PATH = model_folder + experiment_name

  if (os.path.exists(PATH)):
    model = torch.load(PATH).to(device)
    print(' model has been successfully loaded')
    return model, PATH
    
  else:
    print('model was not found')
    model = PD_CNN(chunk_size=chunk_size).to(device)
    log_containing_train_loss = []
    log_containing_val_loss = []
    print('new model has been successfully created')
    return model, PATH
def testing_epoch_func(model, valloader, batch_size=8, device='cpu'):
  #run validation
  model.eval()
  #validate(model=model,valloader=valloader,threshold=0.2,batch_size=batch_size)
  sensitivities = []
  specificities = []

  for threshold in [0,0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.50, 0.60, 0.7, 0.8, 0.9,0.99,0.999, 1]:
    print('threshold: ', threshold)
    _, sensitvity, specificity = testing_epoch_validate(model=model,valloader=valloader,threshold=threshold,batch_size=batch_size, device=device)
    sensitivities.append(sensitvity)
    specificities.append(specificity)
    # 1m runtime
  return sensitivities , specificities
def testing_epoch_viz():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    plt.plot([0, 5], [0, 5])

    plt.ylabel("Y-axis ")
    plt.xlabel("X-axis ")

    image_format = 'jpeg' # e.g .png, .svg, etc.
    image_name = 'myimage.jpeg'

    fig.savefig(image_name, format=image_format, dpi=1200)
def train_test_split(EEG_Dataset, batch_size=8, num_workers=2, chunk_size=2500, device='cpu'):
  '''
    This cell will create a new train/test split based {folds} number of times. It will train and validate for each train/test split. 
  '''
  N_samples = len(EEG_Dataset)
  folds = 52 #how many iterations of training and validation do you want to run. 52 is chosen to be comparable to our CV codes
  K = N_samples/folds 
  val_size = int(np.round(K))
  train_size= int(N_samples - val_size)

  true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
  acc_list, f1_list, AUC_list, sensitivity_list, specificity_list = [], [], [], [], []

  for i in range(folds):
    print(' Starting Fold # ', i)
    #define the train test split
    train_dataset, val_dataset = random_split(EEG_Dataset, [train_size, val_size],generator=torch.Generator().manual_seed(i))

    #create a respective dataloader out of the test/train split
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

    #initialize model
    model = PD_CNN(chunk_size=chunk_size).to(device)
    model.train()

    #run train which will in turn run validate. they return the confusion matrix which will get saved here. 
    model = with_leak_train(model=model,train_dataloader=trainloader, epochs=1, learning_rate=0.0001, training_loss_tracker=[], val_loss_tracker=[])
    model.eval()
    TP, FP, TN, FN, AUC = with_leak_validate(model, valloader=valloader,threshold=0.5,batch_size=batch_size)

    print('true positives: ', TP)
    print('false positives: ',FP)
    print('true negatives: ',TN)
    print('false negatives', FN)  
    print('AUC', AUC)

    true_positives += TP
    false_positives += FP
    true_negatives += TN
    false_negatives += FN

    acc = (TP + TN)/(TP+FP+TN+FN)
    acc_list.append(acc) 
    f1 = TP/(TP+0.5*(FP+FN))
    f1_list.append(f1)
    AUC_list.append(AUC)
    sensitivity = TP/(TP+FN)
    sensitivity_list.append(sensitivity)
    specificity = TN/(TN+FP)
    specificity_list.append(specificity)
    print('----------------------------------------------------------')

    

  print('total True Positives: ', true_positives)
  print('total False Positives: ', false_positives)
  print('total True Negatives: ', true_negatives)
  print('total False Negatives: ', false_negatives)

  print('Mean Accuracy = ', np.mean(acc_list))
  print('Accuracy std = ', np.std(acc_list))
  print('Mean F1 = ', np.mean(f1_list))
  print('F1 std = ', np.std(f1_list))
  print('Mean AUC', np.mean(AUC_list))
  print('AUC std', np.std(AUC_list))
  print('Mean Sensitivity = ', np.mean(sensitivity_list))
  print('Sensitivity std = ', np.std(sensitivity_list))
  print('Mean Specificity = ', np.mean(specificity_list))
  print('Specificity std = ', np.std(specificity_list))
  return model, valloader
def CV_leak_visualize():
  plot_confusion_matrix(TP=1092, FP=38, TN=965, FN=37)
  x2, p = run_chi_squared_test(TP=1092, FP=38, TN=965, FN=37)
  print(p)








# using a copy of this model instead of importing to avoid the pickling error
import torch.nn as nn
import torch

class PD_CNN(nn.Module):

    def __init__(self,chunk_size=2500):
        super(PD_CNN, self).__init__()
        self.chunk_size = chunk_size

        self.conv1 = nn.Conv1d(in_channels=60, out_channels=21, kernel_size=20,stride=1)
        self.norm1 = nn.BatchNorm1d(num_features=21)
        self.maxpool1 = nn.MaxPool1d(kernel_size=4,stride=4)

        self.conv2 = nn.Conv1d(in_channels=21, out_channels=42, kernel_size=10,stride=1)
        self.norm2 = nn.BatchNorm1d(num_features=42)
        self.maxpool2 = nn.MaxPool1d(kernel_size=4,stride=4)

        self.conv3 = nn.Conv1d(in_channels=42, out_channels=42, kernel_size=10,stride=1)
        self.norm3 = nn.BatchNorm1d(num_features=42)
        self.maxpool3 = nn.MaxPool1d(kernel_size=4,stride=4)

        self.conv4 = nn.Conv1d(in_channels=42, out_channels=64, kernel_size=5,stride=1)
        self.norm4 = nn.BatchNorm1d(num_features=64)
        self.maxpool4 = nn.MaxPool1d(kernel_size=4,stride=4)

        
        self.relu = nn.LeakyReLU(0.1)

        
        self.fc1 = nn.Linear(in_features=448,out_features=256)#in_features=4*(self.chunk_size-8)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(in_features=64, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.relu(self.maxpool1(self.norm1(self.conv1(x))))

        x = self.relu(self.maxpool2(self.norm2(self.conv2(x))))
        
        x = self.relu(self.maxpool3(self.norm3(self.conv3(x))))
        
        x = self.relu(self.maxpool4(self.norm4(self.conv4(x))))
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimensi
        
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.fc3(x)

        x = self.softmax(self.fc4(x))
        return x