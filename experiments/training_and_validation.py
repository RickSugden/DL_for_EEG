from torch.utils.data import Dataset, DataLoader, random_split, Subset, RandomSampler
from experiments.data_handling import EEGDataset
import torch.nn as nn # basic building block for neural neteorks
import torch
import torch.optim as optim # optimzer
import pandas as pd
from experiments.models import PD_CNN
import numpy as np
from sklearn.metrics import auc

def train(model,train_dataloader, val_dataloader, epochs=30, learning_rate=0.0001, training_loss_tracker=[], val_loss_tracker=[], device="cpu"):
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
def validate(model, valloader,threshold=0.5,batch_size=8, supress_output=False, device='cpu'):
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
def cross_train(train_dataloader, val_dataloader, epochs=23, learning_rate=0.0001, num_workers=2,  threshold=0.5, chunk_size=2500, device='cpu'):
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
    #create a model
    model = PD_CNN(chunk_size=chunk_size).to(device)
    model.train()

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

    print('The vote was: ', vote)
    print('True Positives: ', TP)
    print('False Positives: ',FP)
    print('True Negatives: ', TN)
    print('False Negatives: ', FN)

    return TP, FP, TN, FN, vote, model


def calculate_metrics(TP, FP, TN, FN):

    #calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    #calculate precision
    precision = TP / (TP + FP)

    #calculate recall
    recall = TP / (TP + FN)

    #calculate f1 score
    f1_score = 2 * ((precision * recall) / (precision + recall))

    #calculate sensitivity
    sensitivity = TP / (TP + FN)

    #calculate specificity
    specficity = TN / (TN + FP)

    return accuracy, f1_score, sensitivity, specficity
def UNM_testing_subject(model, filename_list, EEG_whole_Dataset, batch_size=1, num_workers=2, chunk_size=2500, device='cpu'):
   

##################### TESTING ###########################################
  '''
  Here, a for loop will iterate through every object in the whole dataset. Using the filename, it will determine the
  subject number for the sample and make a test set using only the one subject number. It will repeat this for each subject number.
  Epochs and Learning rate are adjustable below.
  '''
  model.eval()
  correct_votes, incorrect_votes, unsure_votes = 0,0,0
  true_positives, false_positives, true_negatives, false_negatives = 0,0,0,0
  subject_TP, subject_FP, subject_TN, subject_FN = 0, 0, 0, 0
  correct_epochs_list, incorrect_epochs_list = [], []
  list_of_sequences = []

  #leave_out will be 
  for filename in filename_list:
    sequence = []
    to_test = []
    print('Performing testing on subject number: ', filename.split('.')[0])

    for index in range(len(EEG_whole_Dataset)):
      
      #this is a verbose way of accessing a single data point from the dataset in order
      #then if the file associated with that data sample is the same as the filename for the testing subject
      complete_list = range(len(EEG_whole_Dataset))
      subset_ds = Subset(EEG_whole_Dataset, [index])
      sample_sampler = RandomSampler(subset_ds)
      subset_dataloader = DataLoader(subset_ds, sampler=sample_sampler, batch_size=1)
      data = next(iter(subset_dataloader))
      eeg_data, label, file = data
      
      if file[0] == filename:
        
        to_test.append(index)

    #to_be_kept = [x for x in complete_list if x not in to_be_removed]
    #train_dataset = Subset(EEG_whole_Dataset, to_be_kept)
    
    val_dataset = Subset(EEG_whole_Dataset, to_test)

    
    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    del val_dataset, subset_ds, subset_dataloader, data  # ,train_dataset


    TP, FP, TN, FN, vote, sequence = validate(model=model, valloader=val_dataloader,threshold=0.5,batch_size=1, supress_output=False)
    correct_epochs_list.append(TP+TN) 
    incorrect_epochs_list.append(FP+FN) 
    print(len(val_dataloader))
    true_positives += TP
    false_positives += FP
    true_negatives += TN
    false_negatives += FN
    if vote == 'Correct':
      correct_votes += 1
      if TP >0:
        subject_TP += 1
      elif TN >0:
        subject_TN += 1
      
      sequence.append(1)
      
    elif vote == 'Incorrect':
      incorrect_votes += 1
      if FP >0:
        subject_FP += 1
      elif FN >0:
        subject_FN += 1
    else:
      unsure_votes +=1
      sequence.append(0)

    list_of_sequences.append(sequence)
    print('-----------------------------------------------------------------------')
  print('total True Positives: ', true_positives)
  print('total False Positives: ', false_positives)
  print('total True Negatives: ', true_negatives)
  print('total False Negatives: ', false_negatives)
  print('total correct subject classifications: ', correct_votes)
  print('total incorrect subject classifications: ', incorrect_votes)
  print('total unsure subject classifications: ',unsure_votes)
  print('total subject level True positives: ', subject_TP)
  print('total subject level False Positives: ', subject_FP)
  print('total subject level True Negatives: ', subject_TN)
  print('total subject level False Negatives: ', subject_FN)
  denom = (subject_TP + subject_FP +subject_TN+subject_FN)
  if denom != 0: subject_accuracy = (subject_TP + subject_TN) / denom
  else: subject_accuracy = 0
  print('total subject accuracy: ', subject_accuracy)
  F1_denom = (2*subject_TP + subject_FP + subject_FN)
  if F1_denom != 0: subject_F1 = 2*subject_TP / (2*subject_TP + subject_FP + subject_FN)
  else: subject_F1 = 0
  print('total subject F1: ', subject_F1)
  if (subject_TP + subject_FN) != 0: subject_sensitvity = subject_TP / (subject_TP + subject_FN)
  else: subject_sensitvity = 0
  print('total subject sensitivity: ',subject_sensitvity)
  if (subject_FP + subject_TN) != 0: subject_specificity = subject_TN / (subject_FP + subject_TN)
  else: subject_specificity = 0
  print('total subject specificity: ', subject_specificity)

  print('----------------------------------------------------------------')
  return correct_epochs_list, incorrect_epochs_list, list_of_sequences



def loso_cross_validation(filename_list, EEG_whole_Dataset, epochs=1, batch_size=1, num_workers=2, chunk_size=2500, device='cpu'):
   ##################### CROSS VALIDATION ##############
  '''
  Here, a for loop will iterate through every object in the whole dataset. Using the filename, it will determine the
  subject number for the sample and make two subsets: validation using only the one subject number, and training using all other
  subject numbers. It will repeat this for each subject number.
  Epochs and Learning rate are adjustable below
    '''
  correct_votes, incorrect_votes, unsure_votes = 0,0,0
  true_positives, false_positives, true_negatives, false_negatives = 0,0,0,0
  acc_list, f1_list, AUC_list, sensitivity_list, specificity_list = [], [], [], [], []

  #leave_out will be the subject we validate on
  for leave_out in filename_list:
    print('Running a fold while leaving out: ', leave_out)
    to_be_removed = []

    #
    for index in range(len(EEG_whole_Dataset)):
      complete_list = range(len(EEG_whole_Dataset))

      subset_ds = Subset(EEG_whole_Dataset, [index])
      sample_sampler = RandomSampler(subset_ds)
      subset_dataloader = DataLoader(subset_ds, sampler=sample_sampler, batch_size=1)
      data = next(iter(subset_dataloader))
      eeg_data, label, filename = data
      subj_id = filename[0]#.split('_')[1] #remove hashtags to return to UNM dataset

      if subj_id == leave_out:
        
        to_be_removed.append(index)

    to_be_kept = [x for x in complete_list if x not in to_be_removed]

    train_dataset = Subset(EEG_whole_Dataset, to_be_kept)
    val_dataset = Subset(EEG_whole_Dataset, to_be_removed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    del train_dataset, val_dataset #free up memory

    TP, FP, TN, FN, vote, model = cross_train(train_dataloader, val_dataloader, epochs=epochs, learning_rate=0.0001, threshold=0.5, chunk_size=chunk_size, device=device)
    true_positives += TP
    false_positives += FP
    true_negatives += TN
    false_negatives += FN

    if vote == 'Correct':
      correct_votes += 1
    elif vote == 'Incorrect':
      incorrect_votes += 1
    else:
      unsure_votes +=1

  print('total correct subject classifications: ', correct_votes)
  print('total incorrect subject classifications: ', incorrect_votes)
  print('total unsure subject classifications: ',unsure_votes)
  print('total true postives (epochs)', true_positives)
  print('total false postives (epochs)', false_positives)
  print('total true negatives (epochs)', true_negatives)
  print('total false negatives (epochs)', false_negatives)
  print('----------------------------------------------------------------')
  acc, f1, sensitivity, specificity = calculate_metrics(true_positives, false_positives, true_negatives, false_negatives)
  print(acc, f1, sensitivity, specificity)



def UNM_train(model,train_dataloader, val_dataloader, epochs=30, learning_rate=0.0001, training_loss_tracker=[], val_loss_tracker=[], device='cpu'):
    assert epochs > len(training_loss_tracker), 'Loss tracker is already equal to or greater than epochs'

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]).to(device)) #you can adjust weights. first number is applied to PD and the second number to Control
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
            # inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
            
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
          current_val_loss = UNM_validate(model,val_dataloader,batch_size=batch_size)
          val_loss_tracker.append(current_val_loss)

    # whatever you are timing goes here
    # end.record()

    # Waits for everything to finish running
    #torch.cuda.synchronize()

    print('Finished Training Session')
    #print('Time elapsed in miliseconds: ', start.elapsed_time(end))  # milliseconds
    print('The training loss at the end of this session is: ',loss.item())

    return model, training_loss_tracker, val_loss_tracker



#testing the performance
def UNM_validate(model, valloader,threshold=0.5,batch_size=8, device='cpu'):
  '''
    INPUTS:
      model(nn.Module): here we will pass PDNet to the training loop. 
      valloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the validation batches. 
      threshold(float): give a threshold above which a classification will be binarized to PD and below to CTL
      batch_size(int): batch size of the valloader
    
    OUTPUTS:
      avg_loss(float): for validation tracking purposes
  '''
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

    #calculate loss for tracking purposes. Ensure regularization matches that in the training loop.
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
      elif(output[j-1,0]==1) and (labels[j-1,0] == 0 ):
          false_positives += 1
      elif(output[j-1,0]==0) and (labels[j-1,0]== 1):
          false_negatives += 1
      elif(output[j-1,0]==0) and (labels[j-1,0]== 0):
          true_negatives += 1
  
  print('true positives: ', true_positives)
  print('false positives: ',false_positives)
  print('true negatives: ',true_negatives)
  print('false negatives', false_negatives)    
  avg_loss = total_loss/counter 
  print("the average validation loss value is: ", avg_loss.item())
  print('--------------------------------------------------')

  return avg_loss.item()



#testing the performance
def testing_epoch_validate(model, valloader,threshold=0.5,batch_size=8, device='cpu'):
  '''
    INPUTS:
      model(nn.Module): here we will pass PDNet to the training loop. 
      valloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the validation batches. 
      threshold(float): give a threshold above which a classification will be binarized to PD and below to CTL
      batch_size(int): batch size of the valloader
    
    OUTPUTS:
      avg_loss(float): for validation tracking purposes
  '''
  
  total_loss = 0
  true_positives = 0
  true_negatives = 0
  false_positives = 0
  false_negatives = 0


  outputs = []
  true_labels = []
  counter = 0

  #This loads a batch at time
  for i, data in enumerate(valloader, 0):
    #read in data
    inputs, labels, _ = data
    
    inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
    
    #forward
    output = model(inputs)

    #calculate loss for tracking purposes. Ensure regularization matches that in the training loop.
    criterion = nn.CrossEntropyLoss() 
    l2_lambda = 0.0001
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    total_loss += criterion(output,labels) + l2_lambda*l2_norm #,labels
    counter += 1
  
    #binarize output according to the threshold you set
    output[output>threshold] = 1
    output[output<=threshold] = 0
    
    #determine whether each classification was true/false and postive/negative
    for j in range(0,len(output)):
      if (output[j-1,0] == 1) and (labels[j-1,0] == 1):
          true_positives += 1
      elif(output[j-1,0]==1) and (labels[j-1,0] == 0 ):
          false_positives += 1
      elif(output[j-1,0]==0) and (labels[j-1,0]== 1):
          false_negatives += 1
      elif(output[j-1,0]==0) and (labels[j-1,0]== 0):
          true_negatives += 1

  print('true positives: ', true_positives)
  print('false positives: ',false_positives)
  print('true negatives: ',true_negatives)
  print('false negatives', false_negatives)    
  
  avg_loss = total_loss/counter 
  print("the average validation loss value is: ", avg_loss.item())
  
  #calculate the F1 score
  if ((true_positives + false_positives) != 0) and ((true_positives + false_negatives)!= 0):
    recall = true_positives/(true_positives + false_negatives)
    precision = true_positives/(true_positives + false_positives)
    if (precision + recall)!=0:
      F1_Score= (2*precision *recall)/(precision + recall)
    print('The F1 Manual Calculation score was ',F1_Score )

  #Calculate sensitivity specificity
  if ((true_positives + false_negatives) != 0) and ((true_negatives+false_positives)!= 0):
    sensitivity = true_positives/(true_positives + false_negatives)
    specificity = true_negatives/(true_negatives + false_positives)  
    print('sensitivity = ', sensitivity)
    print('specificity = ', specificity)
  #manually code sensitivity and specificity to zero for their respective failure cases
  else: 
    if ((true_positives + false_negatives) == 0):
      sensitivity = 0
      specificity = true_negatives/(true_negatives + false_positives)
    elif (true_negatives + false_positives) == 0:
      specificity = 0
      sensitivity = true_positives/(true_positives + false_negatives)
    else:
      assert False, 'error in manual assignment of sensitivity/specificity'

  accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_negatives + false_positives)
  print('accuracy = ', accuracy)
  
  #print('The F1 Auto Calculation score was ',f1 )
  print('--------------------------------------------------')
  return avg_loss.item(), sensitivity, specificity
  

#testing the performance
def roc_validate(model, valloader,threshold=0.5,batch_size=8, device='cpu'):
  '''
    INPUTS:
      model(nn.Module): here we will pass PDNet to the training loop. 
      valloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the validation batches. 
      threshold(float): give a threshold above which a classification will be binarized to PD and below to CTL
      batch_size(int): batch size of the valloader
    
    OUTPUTS:
      avg_loss(float): for validation tracking purposes
  '''
  
  total_loss = 0
  true_positives = 0
  true_negatives = 0
  false_positives = 0
  false_negatives = 0


  outputs = []
  true_labels = []
  counter = 0

  #This loads a batch at time
  for i, data in enumerate(valloader, 0):
    #read in data
    inputs, labels, _ = data
    
    inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
    
    #forward
    output = model(inputs)
  
    #binarize output according to the threshold you set
    output[output>threshold] = 1
    output[output<=threshold] = 0
    
    #determine whether each classification was true/false and postive/negative
    for j in range(0,len(output)):
      if (output[j-1,0] == 1) and (labels[j-1,0] == 1):
          true_positives += 1
      elif(output[j-1,0]==1) and (labels[j-1,0] == 0 ):
          false_positives += 1
      elif(output[j-1,0]==0) and (labels[j-1,0]== 1):
          false_negatives += 1
      elif(output[j-1,0]==0) and (labels[j-1,0]== 0):
          true_negatives += 1

  

  #Calculate sensitivity specificity
  if ((true_positives + false_negatives) != 0) and ((true_negatives+false_positives)!= 0):
    sensitivity = true_positives/(true_positives + false_negatives)
    specificity = true_negatives/(true_negatives + false_positives)  
    
  #manually code sensitivity and specificity to zero for their respective failure cases
  else: 
    if ((true_positives + false_negatives) == 0):
      sensitivity = 0
      specificity = true_negatives/(true_negatives + false_positives)
    elif (true_negatives + false_positives) == 0:
      specificity = 0
      sensitivity = true_positives/(true_positives + false_negatives)
    else:
      assert False, 'error in manual assignment of sensitivity/specificity'


  return 0, sensitivity, specificity
  
  


 #testing the performance
def with_leak_validate(model, valloader,threshold=0.5,batch_size=8, device='cpu'):

  total_loss = 0
  true_positives = 0
  true_negatives = 0
  false_positives = 0
  false_negatives = 0


  outputs = []
  true_labels = []
  counter = 0

  #This loads a batch at time
  for i, data in enumerate(valloader, 0):

    #read in data
    inputs, labels, _ = data
    inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
    
    #forward
    output = model(inputs)

    #calculate loss + regularization
    criterion = nn.CrossEntropyLoss() 
    l2_lambda = 0.0001
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    total_loss += criterion(output,labels) + l2_lambda*l2_norm
    counter += 1

    #binarize output based on threshold
    output[output>threshold] = 1
    output[output<threshold] = 0

    #determine whether each classification was true/false and positive/negative
    for j in range(0,len(output)):
      if (output[j-1,0] == 1) and (labels[j-1,0] == 1):
          true_positives += 1
      elif(output[j-1,0]==1) and (labels[j-1,0] == 0):
          false_positives += 1
      elif(output[j-1,0]==0) and (labels[j-1,0]== 1):
          false_negatives += 1
      elif(output[j-1,0]==0) and (labels[j-1,0]== 0):
          true_negatives += 1
  
  model.eval()
  #validate(model=model,valloader=valloader,threshold=0.2,batch_size=batch_size)
  sensitivities = []
  specificities = []

  for threshold in [0,0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.50, 0.60, 0.7, 0.8, 0.9,0.99,0.999, 1]:
    #print('threshold: ', threshold)
    _, sensitvity, specificity = roc_validate(model=model,valloader=valloader,threshold=threshold,batch_size=batch_size)
    sensitivities.append(sensitvity)
    specificities.append(specificity)
  AUC = auc((1-np.array(specificities)),sensitivities)
    
  avg_loss = total_loss/counter 
  print("the average validation loss value is: ", avg_loss.item())
  

  return true_positives, false_positives, true_negatives, false_negatives, AUC 


def with_leak_train(model,train_dataloader, epochs=30, learning_rate=0.0001, num_workers=2, training_loss_tracker=[], val_loss_tracker=[], device='cpu'):
    '''
    INPUTS:
      model(nn.Module): here we will pass PDNet to the training loop. 
      valloader(Dataloader): here we will pass the torch.utils.dataloader.Dataloader containing all the validation batches. 
      threshold(float): give a threshold above which a classification will be binarized to PD and below to CTL
      batch_size(int): batch size of the valloader
      supress_output(bool): False to get output print statements
    
    OUTPUTS:
      model(nn.Module): updated, trained model.
    '''
    assert epochs > len(training_loss_tracker), 'Loss tracker is already equal to or greater than epochs'

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #start a timer
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)

    #start.record()

    current_epoch = 0 
    counter = 0
    

    for epoch in range(current_epoch, epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        counter = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs, labels = torch.permute(inputs,(0,1,2)).to(device), labels.to(torch.float32).to(device) #send them to the GPU
            
            batch_size = inputs.shape[0]

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward 
            outputs = model(inputs)

            
            #regularization #Replace pow(2.0) with abs() for L1 regularization
            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                  
            # calculate loss
            loss = criterion(outputs,labels) + l2_lambda*l2_norm

            #backward + optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            counter += 1
          
    # whatever you are timing goes here
    #end.record()

    # Waits for everything to finish running
    # torch.cuda.synchronize()

    return model 


'''
This file contains the deep learning models used in the project.
- All models are implemented using PyTorch and are subclasses of nn.Module.
- Note that the model architecture essentially has to be hard-coded so that means for different datatypes, we need to write different models.
for clinical EEG with ~60 channels, we have one architecture and for wearable EEG with 4 channels, we need a new architecture (adapted from the other one).

I've put the existing model for the ~60 channels below, but I haven't formatted or managed the libraries for you. 
'''
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
'''
This is a method to train the model. It is not a part of the model itself, but it is a part of the training process.
it takes in a model (nn.module), a training dataloader (torch.utils.dataloader.Dataloader), a validation dataloader (torch.utils.dataloader.Dataloader),
epochs and learning rate are hyperparameters that you can adjust. The training_loss_tracker and val_loss_tracker are lists that will be updated with the loss
so that we can visualize the training and validation loss over training iterations. that all works out of the box. 

for now manage libraries, connect this to the main method and make the training loop compatible with the validation method below it, you might have 
to change a couple things because i used multiple versions of the validation method, so not all training loops are yet compatible with the validation method. 
'''