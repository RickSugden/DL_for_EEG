import os
import random
from training_and_validation import loso_cross_validation




def make_csv_from_log(log, final_metrics, experiment_title='CV', save_path='./training_results'):
    ''' 
    log is a list of tuples of the form (subject, TP, FP, TN, FN)
    total_metrics is a list = [correct_votes, incorrect_votes, unsure_votes, true_positives, false_positives, true_negatives, false_negatives, acc, f1, sensitivity, specificity]
    I want a csv with the total metrics at the top row and then the log below it for each subject. 
    '''
    #open a new csv file
    csv = open(save_path+'/'+experiment_title+'.csv', 'w')

    #write final metrics to the first line of the csv
    csv.write('correct_votes, incorrect_votes, unsure_votes, true_positives, false_positives, true_negatives, false_negatives, acc, f1, sensitivity, specificity\n')
    for metric in final_metrics:
        csv.write(str(metric)+',')
    csv.write('\n')

    #write the log to the csv
    for tuple in log:
        for item in tuple:
            csv.write(str(item)+',')
        csv.write('\n')
    
    return csv


def perform_random_hyperparameter_search(EEG_dataset, leave_one_out_list,  sample_size=60, search_title='CNN_hyperparameter_search/', batch_min_max = (1,32), epoch_min_max=(1,5),learning_rate=(0.00001,0.1), model_type='CNN', supress_output=True, device='cpu'):
    save_path = './training_results/'+search_title
    if os.path.exists(save_path)==False:
        #make folder to store results
        os.makedirs(save_path)

    for i in range(sample_size):

        #set hyperparameters
        batch_size_min, batch_size_max = batch_min_max
        epochs_min, epochs_max = 1, 5
        learning_rate_min, learning_rate_max = 0.00001, 0.1

        #select random hyperparameters
        batch_size = random.randint(batch_size_min, batch_size_max)
        epochs = random.randint(epochs_min, epochs_max)
        learning_rate = random.uniform(learning_rate_min, learning_rate_max)
        
        #set experiment title
        experiment_title = model_type+'_batch_size_'+str(batch_size)+'_epochs_'+str(epochs)+'_learning_rate_'+str(round(learning_rate,5))

        print('experiment title: ', experiment_title)
        #perform cross validation
        cv_log, total_metrics = loso_cross_validation(filename_list=leave_one_out_list, EEG_whole_Dataset=EEG_dataset, model_type=model_type, 
                                                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                                    device=device, supress_output=True)

        #save results
        make_csv_from_log(cv_log, total_metrics, experiment_title=experiment_title, save_path=save_path)


def find_best_hyperparmeter_combo(dir='./training_results/CNN_hyperparameter_search/'):
    #find the best hyperparameter options based on average of the accuracy, f1 and (sensitivity+specificity)/2
    #return the filename with the best hyperparameter combo
    
    #get all the files in the directory
    files = os.listdir(dir)
    
    #get the average of the accuracy, f1 and (sensitivity+specificity)/2 for each file
    #store the results in a dictionary
    results = {}
    for file in files:
        if file.endswith('.csv'):
            #open the file
            csv = open(dir+file, 'r')
            #get the first line
            first_line = csv.readline()
            #get the second line
            second_line = csv.readline()
           
            #split the second line into a list
            second_line = second_line.split(',')
            # store the accuracy, f1 and (sensitivity+specificity)/2 as a tuple
            result = (float(second_line[7]), float(second_line[8]), (float(second_line[9])+float(second_line[10]))/2)          
            #store the result in the results dictionary
            results[file] = result
    
    #find the best hyperparameter combo
    best_result = 0
    best_file = ''
    for file in results:
        #get the average of the accuracy, f1 and (sensitivity+specificity)/2
        result = results[file]
        average = (result[0]+result[1]+result[2])/3
        
        if average > best_result:
            best_result = average
            best_file = file

    if best_result == 0:
        print('No hyperparameter combo found accuracy above zero percent.')
    
    return best_file
