# Deep Learning Classifiers to Mimic Diagnosis of Parkinson's from EEG Data

Code used to generate the results for my paper: "Generalizable electroencephalographic classification of Parkinson's disease using deep learning"
doi: https://doi.org/10.1016/j.imu.2023.101352
The code base is built to be modular and allow for a variety of experiements to be performed, including the development of an attention-based transformer for EEG diagnosis (WIP).

### Contents
### 1. main.ipynb
Use this as the main control panel for all the subroutines throughout the repo.
### 2. various .py files
These contain the subroutines to perform:
- Data handling
- Hyperparameter tuning
- models used for benchmarking
- training/testing loops
### 3. Final editions
These are the colab notebooks that I used to generate the data for the paper. This is the easiest way to quickly reproduce an experiment if you're not trying to develop too much. 


### How to use
If you are interested in working with this pipeline, here is what I recommend:
1. Download everything. Including the resting state data from Anjum and Cavanaugh. http://predict.cs.unm.edu/downloads.php
2. Preprocess this data according to the details in the paper. Store in repository folder e.g. ./Data/UI/PD_1661.csv or ./Data/UNM/CTL_1081.csv
3. In main.ipynb, get each of the cells under the miscellaneous header (at the bottom) to run using any of the provided models.
you're now ready to start.
4. develop a model, place in models.py
5. Define the parameters you want to optimize (batch size, training epochs, learning rate) and run hyperparameter training header
6. Use hyperparameter selection to determine which output file from the hyperparameter training was the most successful
7. Continue under that header to train replicates of your chosen hyperparameter combination. This will automatically generate results files under ./testing_results
8. The last few cells from this header allow you to extract and summarize your results, comparing them to any other results you have access to (such as the ones from the paper).

### Implementation details:
- final editions .zip contains colab notebooks, compatible with default colab environment
- local notebooks are optimized for GPU usage and python 3.9.6
- requirements.txt provided
- If you are trying to develop new models, and test rigorously, I recommend doing it locally with the GPU repo.

Here is the architecture used in the publication
![PD_CNN_architecture](https://github.com/RickSugden/DL_for_EEG/assets/41484082/6d63e8ca-f0ba-4af3-aff5-acd3be35360f)

This was the first open-source demonstration of diagnosing **Unseen** Patients with Parkinson's disease, results show balanced classification using Leave-one-subject out Cross validation. 
![result_3](https://github.com/RickSugden/DL_for_EEG/assets/41484082/e772b939-62a6-435c-81c5-d8d62a11ccba)
