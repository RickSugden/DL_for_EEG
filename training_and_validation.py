'''
This is a method to train the model. It is not a part of the model itself, but it is a part of the training process.
it takes in a model (nn.module), a training dataloader (torch.utils.dataloader.Dataloader), a validation dataloader (torch.utils.dataloader.Dataloader),
epochs and learning rate are hyperparameters that you can adjust. The training_loss_tracker and val_loss_tracker are lists that will be updated with the loss
so that we can visualize the training and validation loss over training iterations. that all works out of the box. 

for now manage libraries, connect this to the main method and make the training loop compatible with the validation method below it, you might have 
to change a couple things because i used multiple versions of the validation method, so not all training loops are yet compatible with the validation method. 
'''
from torch.utils.data import Dataset, DataLoader, random_split, Subset, RandomSampler
from data_handling import EEGDataset
import torch.nn as nn # basic building block for neural neteorks
import torch
import torch.optim as optim # optimzer
import pandas as pd
import models
from models import PD_CNN, PD_LSTM
from tqdm import tqdm

def train_with_validation(model,train_dataloader, val_dataloader, epochs=30, learning_rate=0.0001, training_loss_tracker=[], val_loss_tracker=[], device="cpu"):
    '''
    INPUTS:
      model(nn.Module): here we will pass PDNet to the training loop.
      train_dataloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the training batches. 
      val_dataloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the validation batches. 
      epochs(int): the total number of epochs to train for
      learning_rate(float): training hyperparameter defines the rate at which the optimizer will learn
      training_loss_tracker(list): list containing float values of training loss from previous training. The length of this list
      will be taken as the current epoch number.
      val_loss_tracker(list):list containing float values of validation loss from previous training. 
    
    OUTPUTS:
      model(nn.Module): updated network after being trained.
      training_loss_tracker(list): updated list containing float values of training loss from previous training. 
      val_loss_tracker(list): updated list containing float values of validation loss from previous training. 
    '''
    assert epochs > len(training_loss_tracker), 'Loss tracker is already equal to or greater than epochs'

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]).to(device=device)) #you can adjust weights. first number is applied to PD and the second number to Control
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #start a timer
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)
    #start.record()

    current_epoch = len(training_loss_tracker)
    counter = 0

    #here, we take Epoch in a deep learning sense (i.e. iteration of training over the whole dataset)
    for epoch in range(current_epoch, epochs): 
        
        running_loss = 0.0
        counter = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, filename = data
            inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
            
            batch_size = inputs.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward 
            outputs = model(inputs)

            #Apply L2 Regularization. Replace pow(2.0) with abs() for L1 regularization
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                  
            #loss + backward + optimize
            loss = criterion(outputs,labels) + l2_lambda*l2_norm
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            counter += 1
          
        # perform validation every N epochs (adjust val_every)
        val_every = 1
        if epoch % val_every == 0:    
          print('[epoch: %d, batch: %5d] average training loss: %.3f' % (epoch + 1, i + 1, running_loss / counter ))
          training_loss_tracker.append(running_loss/counter)
          running_loss = 0.0
          counter = 0
          current_val_loss = 0
          current_val_loss = validate(model,val_dataloader,batch_size=batch_size, device=device)
          val_loss_tracker.append(current_val_loss)

    # whatever you are timing goes here
    #end.record()

    # Waits for everything to finish running
   #  torch.cuda.synchronize()

    print('Finished Training Session')
    #print('Time elapsed in miliseconds: ', start.elapsed_time(end))  # milliseconds
    print('The training loss at the end of this session is: ',loss.item())

    return model, training_loss_tracker, val_loss_tracker


#testing the performance
def validate(model, valloader,threshold=0.5,supress_output=False, device='cpu'):
  '''
    INPUTS:
      model(nn.Module): here we will pass PDNet to the training loop. 
      valloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the validation batches. 
      threshold(float): give a threshold above which a classification will be binarized to PD and below to CTL
      batch_size(int): batch size of the valloader
      supress_output(bool): False to get output print statements
    
    OUTPUTS:
      true_positives, false_positives, true_negatives, false_negatives (int):confusion matrix
      vote (str): correct/incorrect at the subject level
      sequence (list): binary list of which epochs were classified correctly or incorrectly, for the waterfall plot
  '''
  sequence=[]
  total_loss = 0
  true_positives = 0
  true_negatives = 0
  false_positives = 0
  false_negatives = 0

  counter = 0

  #This loads a batch at time
  for i, data in enumerate(valloader, 0):
    #read in data
    inputs, labels, filename = data
    inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
    
    #forward
    output = model(inputs)

    #calculate loss using L2 regularization and CE loss
    criterion = nn.CrossEntropyLoss() 
    l2_lambda = 0.0001
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    total_loss += criterion(output,labels) + l2_lambda*l2_norm 
    counter += 1
  
    #binarize output according to the threshold you set
    output[output>threshold] = 1
    output[output<=threshold] = 0

    #determine whether each classification was true/false and postive/negative
    for j in range(0,len(output)):
      if (output[j-1,0] == 1) and (labels[j-1,0] == 1):
          true_positives += 1
          sequence.append(1)
      elif(output[j-1,0]==1) and (labels[j-1,0] == 0):
          false_positives += 1
          sequence.append(0)
      elif(output[j-1,0]==0) and (labels[j-1,0]== 1):
          false_negatives += 1
          sequence.append(0)
      elif(output[j-1,0]==0) and (labels[j-1,0]== 0):
          true_negatives += 1
          sequence.append(1)

  #aggregate epochs via majority vote
  if (true_positives + true_negatives) > (false_negatives + false_positives):
    vote = 'Correct'
    
  elif (true_positives + true_negatives) < (false_negatives + false_positives):
    vote = 'Incorrect'
    
  else:
    vote = 'Unsure'

  if supress_output == False:
    print('true positives: ', true_positives)
    print('false positives: ',false_positives)
    print('true negatives: ',true_negatives)
    print('false negatives', false_negatives)  
    print('The vote was: ', vote)

  return true_positives, false_positives, true_negatives, false_negatives, vote, sequence

#cross train function for loso cross validation
def cross_train(model, train_dataloader, val_dataloader, epochs=23, learning_rate=0.0001, num_workers=2,  threshold=0.5, chunk_size=2500, device='cpu', supress_output=False):
    '''
      INPUTS:
        model(nn.Module): here we will pass PDNet to the training loop.
        train_dataloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the training batches. 
        val_dataloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the validation batches. 
        epochs(int): the total number of epochs to train for
        learning_rate(float): training hyperparameter defines the rate at which the optimizer will learn
        
      OUTPUTS:
        TP, FP, TN, FN (int): confusion matrix
        vote (str): correct/incorrect
    '''
    

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #start a timer
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)

    #start.record()
    counter = 0

    ########################## Training ########################################
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        counter = 0
        
        for i, data in enumerate(train_dataloader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, filename = data
            inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
            
            batch_size = inputs.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            #labels = int(labels)
            # forward 
            outputs = model(inputs)

            #Regularization Replaces pow(2.0) with abs() for L1 regularization
    
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                  
            #loss + backward + optimize
            loss = criterion(outputs,labels) + l2_lambda*l2_norm
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            counter += 1
        
    ################################ Validation ##############################
    model.eval()
    TP, FP, TN, FN = 0, 0, 0, 0

    for i, data in enumerate(val_dataloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, filename = data
        inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
              
        batch_size = inputs.shape[0]

        #forward
        output = model(inputs)

        #binarize output
        output[output>threshold] = 1
        output[output<=threshold] = 0        

        #designate each sample to a confusion matrix label
        for j in range(0,len(output)):
          if (output[j-1,0] == 1) and (labels[j-1,0] == 1):
              TP += 1
          elif(output[j-1,0]==1) and (labels[j-1,0] == 0 ):
              FP += 1
          elif(output[j-1,0]==0) and (labels[j-1,0]== 1):
              FN += 1
          elif(output[j-1,0]==0) and (labels[j-1,0]== 0):
              TN += 1

    #determine whether this subject was predicted correct or incorrect by majority vote
    if (TP + TN) > (FP + FN):
      vote = 'Correct'
    elif (TP + TN) < (FP + FN):
      vote = 'Incorrect'
    else:
      vote ='Unsure'

    # whatever you are timing goes here
    #end.record()

    # Waits for everything to finish running
    #torch.cuda.synchronize()
    if supress_output == False:
      print('The vote was: ', vote)
      print('true positives: ', TP)
      print('false positives: ',FP)
      print('true negatives: ',TN)
      print('false negatives', FN)
   

    return TP, FP, TN, FN, vote, model


def calculate_metrics(TP, FP, TN, FN):

    if (TP + TN + FP + FN) == 0:
      assert ValueError('No samples were predicted')
    else:
      #calculate accuracy
      accuracy = (TP + TN) / (TP + TN + FP + FN)

    if (TP + FP) == 0:
      precision = 0
    else:
      #calculate precision
      precision = TP / (TP + FP)

    if (TP + FN) == 0:
      recall = 0
    else: 
      #calculate recall
      recall = TP / (TP + FN)

    #confirm f1 can be calculated
    if (precision + recall) == 0:
      f1_score = 0
    else:
       #calculate f1 score
      f1_score = 2 * ((precision * recall) / (precision + recall))

    if (TP + FN) == 0:
      sensitivity = 0
    else:
      #calculate sensitivity
      sensitivity = TP / (TP + FN)

    if (TN + FP) == 0:
      specficity = 0
    else:
      #calculate specificity
      specficity = TN / (TN + FP)

    return accuracy, f1_score, sensitivity, specficity

def loso_cross_validation(filename_list, EEG_whole_Dataset, model_type='CNN', epochs=1, batch_size=1, num_workers=1, learning_rate=0.0001, chunk_size=2500, device='cpu', supress_output=False):
   ##################### CROSS VALIDATION ##############
  '''
  Here, a for loop will iterate through every object in the whole dataset. Using the filename, it will determine the
  subject number for the sample and make two subsets: validation using only the one subject number, and training using all other
  subject numbers. It will repeat this for each subject number.
  Epochs and Learning rate are adjustable below
    '''
  correct_votes, incorrect_votes, unsure_votes = 0,0,0
  true_positives, false_positives, true_negatives, false_negatives = 0,0,0,0
  
  log = []
  print('tracker for the subjects')
  #leave_out will be the subject we validate on
  #with tqdm(total=len(filename_list), desc='Subjects completed:', position=1) as inner_pbar:

  for leave_out in tqdm(filename_list, total=len(filename_list)) : 
    
    #print('Running a fold while leaving out: ', leave_out)
    
    #make a training and validation dataset
    train_dataset, val_dataset = loso_split(EEG_whole_Dataset, leave_out)
    #convert datasets to dataloaders and delete the datasets to free up memory
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    del train_dataset, val_dataset #free up memory

    #create a model
    if model_type == 'CNN':
      model = PD_CNN(chunk_size=chunk_size).to(device)
    elif model_type == 'LSTM':
      model = PD_LSTM(device=device).to(device)
    else:
      assert ValueError('Model not recognized')
    model.train()

    #perform one fold of training and validation
    TP, FP, TN, FN, vote, model = cross_train(model, train_dataloader, val_dataloader, epochs=epochs, learning_rate=learning_rate, threshold=0.5, chunk_size=chunk_size, device=device, supress_output=supress_output)

    #add the metrics to the lists
    true_positives += TP
    false_positives += FP
    true_negatives += TN
    false_negatives += FN

    #determine whether this subject was predicted correct or incorrect by majority vote
    if vote == 'Correct':
      correct_votes += 1
    elif vote == 'Incorrect':
      incorrect_votes += 1
    else:
      unsure_votes +=1

    log.append((leave_out, TP, FP, TN, FN, vote))
    #inner_pbar.update(1)  
      

  #print the results 
  print('total correct subject classifications: ', correct_votes)
  print('total incorrect subject classifications: ', incorrect_votes)
  print('total unsure subject classifications: ',unsure_votes)
  print('total true postives (epochs)', true_positives)
  print('total false postives (epochs)', false_positives)
  print('total true negatives (epochs)', true_negatives)
  print('total false negatives (epochs)', false_negatives)
  print('----------------------------------------------------------------')
  acc, f1, sensitivity, specificity = calculate_metrics(true_positives, false_positives, true_negatives, false_negatives)
  print('accuracy',acc,' f1 ', f1, 'sensitivity', sensitivity, 'specificity', specificity)

  total_results = [correct_votes, incorrect_votes, unsure_votes, true_positives, false_positives, true_negatives, false_negatives, acc, f1, sensitivity, specificity]
 
  

  return log, total_results


  
def loso_split(EEG_whole_Dataset, leave_out):

  # we will make a list of all the indices to be removed
  to_be_removed = []

  #iterate through each object in the dataset to see if it belongs to the subject we are leaving out
  for index in range(len(EEG_whole_Dataset)):

    #get list of all indecies so we can compare to the to_be_removed list
    complete_list = range(len(EEG_whole_Dataset))

    #build a dataloader with only one sample
    subset_ds = Subset(EEG_whole_Dataset, [index])
    sample_sampler = RandomSampler(subset_ds)
    subset_dataloader = DataLoader(subset_ds, sampler=sample_sampler, batch_size=1)

    #get the filename of the sample. this allows us to determine the subject number
    _, _, filename = next(iter(subset_dataloader))
    
    #get the subject number and determine if this index is from the subject to leave out
    subj_id = filename[0]#.split('_')[1] #remove hashtags to return to UNM dataset
    if subj_id == leave_out:
      to_be_removed.append(index)

  #now we have a list of all the indices to be removed. We can use this to make a list of all the indices to be kept
  to_be_kept = [x for x in complete_list if x not in to_be_removed]

  #split the dataset into training and validation based on the one subject we are leaving out
  train_dataset = Subset(EEG_whole_Dataset, to_be_kept)
  val_dataset = Subset(EEG_whole_Dataset, to_be_removed)

  return train_dataset, val_dataset

def train(model,train_dataloader, epochs=30, learning_rate=0.0001, training_loss_tracker=[], device="cpu"):
    ''' Method to perform a training session with no validation feedback. meant to be a followup to a CV/hyperparameter search
    INPUTS:
      model(nn.Module): here we will pass PDNet to the training loop.
      train_dataloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the training batches. 
      epochs(int): the total number of epochs to train for
      learning_rate(float): training hyperparameter defines the rate at which the optimizer will learn
      training_loss_tracker(list): list containing float values of training loss from previous training. The length of this list
      will be taken as the current epoch number.

    OUTPUTS:
      model(nn.Module): updated network after being trained.
      training_loss_tracker(list): updated list containing float values of training loss from previous training. 

    '''
    assert epochs > len(training_loss_tracker), 'Loss tracker is already equal to or greater than epochs'

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]).to(device=device)) #you can adjust weights. first number is applied to PD and the second number to Control
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    current_epoch = len(training_loss_tracker)
    counter = 0

    #here, we take Epoch in a deep learning sense (i.e. iteration of training over the whole dataset)
    for epoch in range(current_epoch, epochs): 
        
        running_loss = 0.0
        counter = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, filename = data
            inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
            
            batch_size = inputs.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward 
            outputs = model(inputs)

            #Apply L2 Regularization. Replace pow(2.0) with abs() for L1 regularization
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                  
            #loss + backward + optimize
            loss = criterion(outputs,labels) + l2_lambda*l2_norm
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            counter += 1
          
        

    print('Finished Training Session')
    #print('Time elapsed in miliseconds: ', start.elapsed_time(end))  # milliseconds
    print('The training loss at the end of this session is: ',loss.item())

    return model, training_loss_tracker



def test_subjectwise(trained_model, test_data_src, filename_list):
   
  trained_model.eval()
  correct_votes, incorrect_votes, unsure_votes = 0,0,0
  true_positives, false_positives, true_negatives, false_negatives = 0,0,0,0
  subject_TP, subject_FP, subject_TN, subject_FN = 0, 0, 0, 0
  
  


  #leave_out will be 
  for filename in filename_list:
    
    
    print('Performing testing on subject number: ', filename.split('.')[0])

    #make dataset
    single_subject_dataset = EEGDataset(data_path=(test_data_src+filename))
    single_subject_dataloader = DataLoader(single_subject_dataset, shuffle=False,  batch_size=1)

    
    TP, FP, TN, FN, vote, seq = validate(model=trained_model, valloader=single_subject_dataloader,threshold=0.5, supress_output=True)
    #list_of_sequences.append(seq)  

    #add to the total counters
    true_positives += TP
    false_positives += FP
    true_negatives += TN
    false_negatives += FN

    #see what the vote was
    if vote=='Correct':
      correct_votes += 1
      #determine if the vote was positive or negative
      if TP > 0:
        subject_TP += 1
      elif TN > 0:
        subject_TN += 1
      else:
        ValueError('Vote is correct but no TP or TN')

    elif vote=='Incorrect':
      incorrect_votes += 1
      #detemine if the vote was positive or negative
      if FP > 0:
        subject_FP += 1
      elif FN > 0:
        subject_FN += 1
      else:
        ValueError('Vote is incorrect but no FP or FN')
    elif vote=='Unsure':
      unsure_votes += 1
    else:
       ValueError('Vote is not one of the three options')
    

  #aggregate the results
  epoch_conf_mat = [true_positives, false_positives, true_negatives, false_negatives]
  sub_conf_mat = [subject_TP, subject_FP, subject_TN, subject_FN]
  votes = [correct_votes, incorrect_votes, unsure_votes]

  #return the aggregated results
  return  epoch_conf_mat, sub_conf_mat, votes


def save_testing_results(epoch_conf_mat, sub_conf_mat, votes, results_dir='./testing_results/', experiment_name='testing_results', replicate=0):
   
   #assert that the last character of results_dir is a '/'
  assert results_dir[-1] == '/', 'results_dir must end with a /'

  #confirm the data is there
  print(epoch_conf_mat)

  #open a new csv file
  csv = open(results_dir+experiment_name+'_rep'+str(replicate)+'.csv', 'w')

  #write final metrics to the first line of the csv
  csv.write('true_positives, false_positives, true_negatives, false_negatives \n')
  for result in epoch_conf_mat:
      csv.write(str(result)+',')
  csv.write('\n')

  for result in sub_conf_mat:
      csv.write(str(result)+',')
  csv.write('\n')
  
  csv.write('correct_votes, incorrect_votes, unsure_votes \n')
  for result in votes:
      csv.write(str(result)+',')
  csv.write('\n')

  return csv
  


def initialize_model(model_type='CNN', device='cpu'):
  if model_type == 'CNN':
    model = PD_CNN().to(device)   
  elif model_type == 'LSTM':
    model = PD_LSTM().to(device)
  else:
    ValueError('Model type not recognized')
  

  return model