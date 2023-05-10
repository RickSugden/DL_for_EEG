'''
This is a method to train the model. It is not a part of the model itself, but it is a part of the training process.
it takes in a model (nn.module), a training dataloader (torch.utils.dataloader.Dataloader), a validation dataloader (torch.utils.dataloader.Dataloader),
epochs and learning rate are hyperparameters that you can adjust. The training_loss_tracker and val_loss_tracker are lists that will be updated with the loss
so that we can visualize the training and validation loss over training iterations. that all works out of the box. 

for now manage libraries, connect this to the main method and make the training loop compatible with the validation method below it, you might have 
to change a couple things because i used multiple versions of the validation method, so not all training loops are yet compatible with the validation method. 
'''

def train(model,train_dataloader, val_dataloader, epochs=30, learning_rate=0.0001, training_loss_tracker=[], val_loss_tracker=[]):
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
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]).cuda()) #you can adjust weights. first number is applied to PD and the second number to Control
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #start a timer
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    current_epoch = len(training_loss_tracker)
    counter = 0

    #here, we take Epoch in a deep learning sense (i.e. iteration of training over the whole dataset)
    for epoch in range(current_epoch, epochs): 
        
        running_loss = 0.0
        counter = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
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
          current_val_loss = validate(model,val_dataloader,batch_size=batch_size)
          val_loss_tracker.append(current_val_loss)

    # whatever you are timing goes here
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training Session')
    print('Time elapsed in miliseconds: ', start.elapsed_time(end))  # milliseconds
    print('The training loss at the end of this session is: ',loss.item())

    return model, training_loss_tracker, val_loss_tracker


#testing the performance
def validate(model, valloader,threshold=0.5,batch_size=8, supress_output=False):
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

