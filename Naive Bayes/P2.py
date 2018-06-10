#Name - Amarjit Jadhav

import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Section to read, shuffle and pre-process data.
print ("Reading spambase data set")
inputDataSet = genfromtxt('spambase.data', delimiter=',')
inputDataLength = len(inputDataSet)
targetValues = inputDataSet[:, -1]
print ("Shuffling spambase data set")
np.random.shuffle(inputDataSet)
attributes_values = preprocessing.scale(inputDataSet[:, 0:-1])
targetValues = inputDataSet[:, -1]

#split data into training and testing data_set(50%)
X_cor_train, X_cor_test, y_cor_train, y_cor_test = train_test_split(attributes_values, targetValues, test_size=0.5, random_state=17)


#The total probabilities for the spam and non- spam should be around 40% and 60 %
Total_Spam = 0
train_dataset_length = len(X_cor_train)
for eachrow in range(train_dataset_length):
    if y_cor_train[eachrow] == 1:
        Total_Spam += 1
prob_spam_mails = float(Total_Spam) / train_dataset_length
prob_non_spam_mails = 1 - prob_spam_mails
print("Spam mail probability is: \t",prob_spam_mails)
print("Non-spam mail probability is: \t",prob_non_spam_mails)

#calculating mean and Standard Deviation for all the attributes_values i.e nothing but the features.
mean_spam_mails,stan_dev_spam_mails,mean_non_spam_mails,stan_dev_non_spam_mails  = [], [],[],[]
for attributes_values in range(0,57):
    spam_values,nonspam_values = [],[]
    for eachrow in range(0, train_dataset_length):
        if (y_cor_train[eachrow] == 1):
            spam_values.append(X_cor_train[eachrow][attributes_values])
        else :
           nonspam_values.append(X_cor_train[eachrow][attributes_values])
    mean_spam_mails.append(np.mean(spam_values))
    mean_non_spam_mails.append(np.mean(nonspam_values))
    stan_dev_spam_mails.append(np.std(spam_values))
    stan_dev_non_spam_mails.append(np.std(nonspam_values))

#section for replacing 0 standard deviation with .0001
for feature in range(0,57):
    if(stan_dev_spam_mails[feature]==0):
        stan_dev_spam_mails[feature] = .0001
    if(stan_dev_non_spam_mails[feature]==0):
        stan_dev_non_spam_mails[feature]=.0001

#Section to calculate precision, Recall and accuracy
def calculate_accuracy_precision_recall(tar_values, predicted_values, threshold_value):
    true_pos,false_pos,true_neg,false_neg = 0,0,0,0
    for eachrow in range(len(predicted_values)):
        if (predicted_values[eachrow] > threshold_value and tar_values[eachrow] == 1)  :
            true_pos += 1
        elif (predicted_values[eachrow] > threshold_value and tar_values[eachrow] == 0 )  :
            false_pos += 1
        elif (predicted_values[eachrow] <= threshold_value and tar_values[eachrow] == 1 )  :
            false_neg += 1
        elif (predicted_values[eachrow] <= threshold_value and tar_values[eachrow] == 0 )  :
            true_neg += 1
    result_accuracy = float(true_pos + true_neg) / len(predicted_values)
    result_recall = float(true_pos) / (true_pos + false_neg)
    result_precision = float(true_pos) / (true_pos + false_pos)
    return  result_accuracy, result_recall, result_precision
	
# Making use of the Gaussian Naive Bayes algorithm to calculate the probability and predict classes.
probability_spam,probability_non_spam = 0,0
pred = []
for eachrows in range(0,len(X_cor_test)):
    NB_final_spam_prob,NB_final_nonspam_prob = [],[]
    NB_part_cal_1,NB_part_cal_2,NB_part_cal_3,NB_part_cal_4 = 0,0,0,0
    for attributes_values in range(0,57):
        NB_part_cal_1 = float(1)/ (np.sqrt(2 * np.pi) * stan_dev_spam_mails[attributes_values])
        NB_part_cal_2 = (np.e) ** - (((X_cor_test[eachrows][attributes_values] - mean_spam_mails[attributes_values]) ** 2) / (2 * stan_dev_spam_mails[attributes_values] ** 2))
        NB_final_spam_prob.append(NB_part_cal_1 * NB_part_cal_2)
        NB_part_cal_3 = float(1)/ (np.sqrt(2 * np.pi) * stan_dev_non_spam_mails[attributes_values])
        NB_part_cal_4 = (np.e) ** - (((X_cor_test[eachrows][attributes_values] - mean_non_spam_mails[attributes_values]) ** 2) / (2 * stan_dev_non_spam_mails[attributes_values] ** 2))
        NB_final_nonspam_prob.append(NB_part_cal_3 * NB_part_cal_4)
		
    probability_spam = np.log(prob_spam_mails) + np.sum(np.log(np.asarray(NB_final_spam_prob)))
    probability_non_spam = np.log(prob_non_spam_mails) + np.sum(np.log(np.asarray(NB_final_nonspam_prob)))
    output = np.argmax([probability_non_spam, probability_spam])
    pred.append(output)
acc,rec,pre = calculate_accuracy_precision_recall(y_cor_test, pred, 0)
print("Confusion matrix:\n",metrics.confusion_matrix(y_cor_test, pred))
print ("Accuracy Value: \t",acc)
print ("Precision Value: \t", pre)
print ("Recall Value: \t",rec)
