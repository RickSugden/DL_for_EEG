'''
This will be a library for functions that visualize the results of the DL models. 
there should be cells among the notebooks, you won't need to create anything new for right now.
 examples of the graphs needed are:
 - ROC curve
 - confusion matrix
 - waterfall plot
 - sequence plot
 - combo plot
 - training curves
'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib as mpl
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split, Subset, RandomSampler
from data_handling import EEGDataset
from training_and_validation import validate
#Calculate AUC
#testing the performance
#set the device to cuda:0 
if torch.cuda.is_available():  
  device = "cuda:0" 
else:  
  device = "cpu"  
# Chi Squared Test
from scipy.stats import chisquare

def run_chi_squared_test(TP, FP, TN, FN):
  chisq, p_value = chisquare([TP, FP, TN, FN])

  return chisq, p_value

#ROC Validate
def roc_validate(model, valloader,threshold=0.5,batch_size=8):
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
    inputs, labels = data
    
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
#Calculate AUC
def calculate_AUC(model, val_dataloader, threshold=0.5, batch_size=8):

  sensitivities, specificities = [], []

  #calculate AUC by testing different thresholds and calculating the area under the curve
  for threshold in [0,0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.50, 0.60, 0.7, 0.8, 0.9,0.99,0.999, 1]:

    #print('threshold: ', threshold)
    _, sensitvity, specificity = roc_validate(model=model,valloader=val_dataloader,threshold=threshold,batch_size=batch_size)
    sensitivities.append(sensitvity)
    specificities.append(specificity)

  #calculate auc
  AUC = auc((1-np.array(specificities)),sensitivities)


  return AUC
#ROC curve
def plot_roc_auc(specificities, sensitivities):
  plt.figure(figsize=(10,8))
  plt.plot((1-np.array(specificities)),sensitivities);
  #plt.title('ROC Curve', size= 20);
  plt.xlabel('1-Specificity', size = 14);
  plt.ylabel('Sensitivity', size = 14);
  plt.legend(['CNN'])
  AUC = auc((1-np.array(specificities)),sensitivities)
  plt.text(0.7,0.2, 'AUC = %f'%AUC, fontsize=10)
  plt.show()
#Confusion matrix
def plot_confusion_matrix(TP, FP, TN, FN, filename='confusion_matrix_image'):
  cm = 1/2.54 #define inch to cm
  fig, (ax1, axcb) = plt.subplots(1,2,gridspec_kw={'width_ratios':[0.8,0.08]}, sharey=False, figsize = (6*cm,4*cm))
  
  #plt.subplots_adjust(wspace=0.15)
  
  pred = []
  true = []

  for i in range(TP):
    pred.append(1)
    true.append(1)
  
  for i in range(FP):
    pred.append(1)
    true.append(0)
  
  for i in range(TN):
    pred.append(0)
    true.append(0)

  for i in range(FN):
    pred.append(0)
    true.append(1)

  
  conf_matrix = confusion_matrix(true, pred)
  # Using Seaborn heatmap to create the plot
  ax1 = sns.heatmap(conf_matrix,cmap='Blues',vmin=0,vmax=max([TP,FP,TN,FN]),ax=ax1,cbar_ax=axcb, annot=True, annot_kws={'fontsize':7},fmt='g')
  
  # labels the title and x, y axis of plot
  #fx.set_title('Insert title here');
  ax1.set_xlabel('Predicted Values',size=7)
  ax1.set_ylabel('Actual Values ', size=7);

  # labels the boxes
  ax1.xaxis.set_ticklabels(['Control','PD'],size=7)
  ax1.yaxis.set_ticklabels(['Control','PD'],size=7)
  #ax2.title.set_text('CV by Subject')
  #fig.tight_layout()
  
  
  plt.savefig((filename+'.png'), dpi=300, format='png',bbox_inches='tight')

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
# waterfall plot
def waterfall_plot(correct_epochs_list, incorrect_epochs_list, filename_list, color1='#E55451', color2='228b22'):
    differential = np.array(correct_epochs_list) - np.array(incorrect_epochs_list)
    total_epochs = np.array(correct_epochs_list) + np.array(incorrect_epochs_list)
    normalized_differential = differential/total_epochs
    #print(differential)
    diff_list,subject_list = zip(*sorted(zip(normalized_differential, filename_list)))
    diff_list = np.array(diff_list)
    subject_list = np.array(subject_list)
    mask1 = diff_list < 0
    mask2 = diff_list >= 0

    x = np.arange(len(diff_list))

    plt.plot(figsize=(20,12))
    plt.barh(x, diff_list, color=color1,tick_label=subject_list) #, tick_label=subject_list[mask1]
    plt.barh(x[mask2], diff_list[mask2], color=color2)
    plt.ylabel('Subject ID',size=14);
    plt.xlabel('Correct Epochs - Incorrect Epochs', size=14);
    plt.title('Correct Epochs by Subject', size=20);
    plt.gcf().set_size_inches(8, 8)
    plt.show()
# sequnce plot
##################### TESTING ###########################################
'''
Here, a for loop will iterate through every object in the whole dataset. Using the filename, it will determine the
subject number for the sample and make a test set using only the one subject number. It will repeat this for each subject number.
Epochs and Learning rate are adjustable below.
'''
def sequence_plot(model, filename_list, data_src, chunk_size):
    model.eval()
    correct_votes, incorrect_votes, unsure_votes = 0,0,0
    true_positives, false_positives, true_negatives, false_negatives = 0,0,0,0
    subject_TP, subject_FP, subject_TN, subject_FN = 0, 0, 0, 0
    correct_epochs_list, incorrect_epochs_list = [], []
    list_of_sequences = []


    #leave_out will be 
    for filename in filename_list:
        to_test = []
        print('Performing testing on subject number: ', filename.split('.')[0])

        #make dataset
        single_subject_dataset = EEGDataset(data_path=(data_src+filename),  chunk_size=chunk_size)
        single_subject_dataloader = DataLoader(single_subject_dataset, shuffle=False,  batch_size=1)

        
        TP, FP, TN, FN, vote, seq = validate(model=model, valloader=single_subject_dataloader,threshold=0.5,batch_size=1, supress_output=True)
        list_of_sequences.append(seq)  

# combo plot
def make_combo_plot(sequence_list=[], filename_list=[], correct_color='#228B22', incorrect_color='#E55451'):
  
  ############################################################
  #create list of normalized correctness and sort it
  diff_list=[]
  sorted_diffs = []
  sorted_sequences = []
  sorted_filenames = []
  sorted_labels = []
  sorted_pvals = []
  for sequence in sequence_list:
    sequence = [i for i in sequence if i != -1]
    correct = sequence.count(1)
    incorrect = sequence.count(0)
    empty = sequence.count(-1)
    diff = (correct-incorrect)/len(sequence)
    diff_list.append(diff)

  zipped = list(zip(diff_list, sequence_list, filename_list))
  zipped.sort()
  sorted_diffs, sorted_sequences, sorted_filenames = map(list, zip(*zipped))

  sorted_diffs = np.flip(np.array(sorted_diffs))
  #percent_list, subject_list = zip(*sorted())
  ################################################################
  #set up a subplot to fill in
  mask1 = sorted_diffs < 0
  mask2 = sorted_diffs >= 0

  x = np.arange(len(sorted_diffs))

  fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, gridspec_kw={'width_ratios': [1, 1, 0.05, 0.05]})
  plt.plot(figsize=(20,12), ax=ax[0])
  ax[0].barh(x, sorted_diffs, color=incorrect_color,tick_label=sorted_filenames,align='edge') #, tick_label=subject_list[mask1]
  ax[0].barh(x[mask2], sorted_diffs[mask2], color=correct_color,align='edge')
  ax[0].set_ylabel('Subject ID',size=14);
  ax[0].set_xlabel('Normalized Epoch Accuracy', size=14);
  pop_c = mpatches.Patch(color=incorrect_color, label='Incorrect Subject') # used to be default red
  pop_d = mpatches.Patch(color=correct_color, label='Correct Subject') # used to be default green
  ax[0].legend(handles=[pop_c,pop_d], bbox_to_anchor=(0.47,1.1))

  ################################################################


  #determine the maximum length of all the sequences
  max_len = 0
  for sequence in sorted_sequences:

    if len(sequence) > max_len:
      max_len = len(sequence)

  #make all lists the same length by appending -1 to the short ones 
  for i, sequence in enumerate(sorted_sequences):

    if len(sequence) < max_len:
      short_by = max_len - len(sequence)

      for j in range(short_by):
        sorted_sequences[i].append(-1)

  sorted_sequences = np.vstack([np.array(i) for i in sorted_sequences])
  flipped_sequences = np.flip(sorted_sequences, axis=0)

 # for i, sequence in enumerate(flipped_sequences):
 #  q, p = statsmodels.sandbox.stats.runs.runstest_1samp(sequence, cutoff='mean', correction=True)
    #print('for subject ', i, ' the p = ', p)
 #    sorted_pvals.append(p) 
 # need statsmodels version 0.15.0 to use this
  #reshap pvals into two dimensional shape with a singleton dimension
  """ sorted_pvals = np.asarray(sorted_pvals).reshape(len(sorted_pvals), 1)

  #mask sorted pvals to less than 0.01 and 0.001 
  for i in range(len(sorted_pvals)):
    
      if sorted_pvals[i] < 0.01:
        if sorted_pvals[i]< 0.001:
          sorted_pvals[i] = 2
        else:
          sorted_pvals[i] = 1
      else:
        sorted_pvals[i] = 0

    """
  
  # define the colors
  cmap = mpl.colors.ListedColormap(['#FFFFFF', incorrect_color, correct_color])

  # create and normalize object the describes the limits of each color
  bounds = [-1, -0.25, 0.5, 1.]
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

  # use filenames to deduce the PD or CTLs
  for i in range(len(sorted_filenames)):
    sorted_filenames[i] = sorted_filenames[i].split('.')[0]
    
    if sorted_filenames[i].split('_')[0]== 'CTL':
      sorted_labels.append(0)
    elif sorted_filenames[i].split('_')[0] == 'PD':
      sorted_labels.append(1)
    else:
      assert False, 'error assigning label'
  #reverse the order of the list and make into two dimensional array
  sorted_labels.reverse()
  sorted_labels = np.asarray(sorted_labels).reshape(len(sorted_labels), 1)

  #sequence plot heatmap
  sns.heatmap(flipped_sequences, cmap=cmap, norm=norm, yticklabels=np.flip(sorted_filenames), cbar=False, ax = ax[1]);
  pop_a = mpatches.Patch(color=incorrect_color, label='Incorrect Epoch')
  pop_b = mpatches.Patch(color=correct_color, label='Correct Epoch')
  ax[1].legend(handles=[pop_a,pop_b], bbox_to_anchor=(0.45,1.1))
  ax[1].set_xlabel('Epoch Number', size=14);
  
  # first colorbar for significance
  # define the colors
  """
  cmap2 = mpl.colors.ListedColormap(['#B9BBB6','#FEE12B','#6F2DB8'])
  pop_non = mpatches.Patch(color='#B9BBB6', label='ns')
  pop_sig = mpatches.Patch(color='#FEE12B', label='*')
  pop_sig2 = mpatches.Patch(color='#6F2DB8', label='**')
  ax[2].legend(handles=[pop_non, pop_sig2], bbox_to_anchor=(0.85,1.1)) #add the pop_sig if there are any examples
  
  # create and normalize object the describes the limits of each color
  bounds = [-1, -0.25, 0.5, 1.]
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  sns.heatmap(sorted_pvals, ax = ax[2], cbar=False, cmap=cmap2)
  """
 

  # define the colors
  cmap3 = mpl.colors.ListedColormap(['#0000FF','#FFA500'])
  pop_ctl = mpatches.Patch(color='#0000FF', label='CTL')
  pop_pd = mpatches.Patch(color='#FFA500', label='PD')
  ax[3].legend(handles=[pop_ctl,pop_pd], bbox_to_anchor=(2.85,1.1))

  # create and normalize object the describes the limits of each color
  bounds = [-1, 0.5, 1.]
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  sns.heatmap(sorted_labels, ax = ax[3], cbar=False, cmap=cmap3)



  
  ax[0].set_yticks(np.arange(len(sorted_filenames)))
  ax[0].set_yticklabels(labels=np.flip(sorted_filenames), va='center')
  plt.gcf().set_size_inches(10, 10)
  plt.savefig('subject_inference_waterfall_combo.png', dpi=300, format='png',bbox_inches='tight')
  return ax,fig
#training curves (Paramters=(log containing training loss, log containing val loss, color1, color2, and labels possibly))
def training_cruve(log_containing_train_loss, log_containing_val_loss, color1='#0000FF', color2='#E55451', label1='training',
                    label2='validation'):
    fig, ax = plt.subplots()
    #ax1 = fig.add_subplot(2,1,1)

    ax.plot(np.arange(0,len(log_containing_train_loss)), log_containing_train_loss, linestyle='--',color=color1,label=label1)
    ax.plot(np.arange(0,len(log_containing_val_loss)), log_containing_val_loss, linestyle='--',color=color2,label=label2)
    plt.legend()
    plt.xlabel('Epoch Number', size=14)
    plt.ylabel('Cross Entropy Loss',size=14)
    # Save the full figure...
    fig.savefig('training_curve.png', dpi=300)