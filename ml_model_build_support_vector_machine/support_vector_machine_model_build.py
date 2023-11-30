# -*- coding: utf-8 -*-


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

from ml_functions.ml_model_eval import pred_proba_plot
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

plt.close('all')

#------------------------------- ML MODEL BUILD -------------------------------

#importing the data and creating the feature dataframe and target series

with open('../2019_prem_generated_clean/2019_prem_df_for_ml_5.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('../2019_prem_generated_clean/2019_prem_df_for_ml_10.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)


x_10 = df_ml_10.drop(['Target Fixture ID', 'Team Av Shots', 'Opponent Av Shots'], axis=1)
y_10 = df_ml_10['Team Av Corners']

x_5 = df_ml_5.drop(['Target Fixture ID', 'Team Av Shots', 'Opponent Av Shots'], axis=1)
y_5 = df_ml_5['Team Av Corners']

# --------------- TESTING C PARAM ---------------

expo_iter = np.square(np.arange(0.1, 10, 0.1))


def testing_c_parms(df, iterable):
    training_score_li = []
    test_score_li = []
    for c in iterable:
        x = df.drop(['Target Fixture ID', 'Team Av Shots', 'Opponent Av Shots'], axis=1)
        y = df['Team Av Corners']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        clf = svm.SVC(kernel='rbf', C=c)
        clf.fit(x_train, y_train)
        train_data_score = round(clf.score(x_train, y_train) * 100, 1)
        test_data_score = round(clf.score(x_test, y_test) * 100, 1)
        training_score_li.append(train_data_score)
        test_score_li.append(test_data_score)
    return training_score_li, test_score_li


training_score_li, test_score_li = testing_c_parms(df_ml_10, expo_iter)

# from the plot below we can see that a c of around 3 is likely to be more optimal than 1
fig, ax = plt.subplots()
ax.plot(expo_iter, test_score_li)

print('\nSUPPORT VECTOR MACHINES\n')
#--------------------------- SUPPORT VECTOR MACHINE ---------------------------

from sklearn.preprocessing import LabelEncoder

def svm_train(df):

    # create features matrix
    x = df.drop(['Target Fixture ID', 'Team Av Shots', 'Opponent Av Shots'], axis=1)

    le = LabelEncoder()
    df['Team Av Corners'] = le.fit_transform(df['Team Av Corners'])  # Encode categorical labels to numerical

    # Target variable for classification
    y = df['Team Av Corners']

    # split into training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Testing C Parameters
    expo_iter = np.square(np.arange(0.1, 10, 0.1))
    training_score_li, test_score_li = testing_c_parms(df, expo_iter)

    # Identify the optimal C value (you can customize this logic)
    optimal_c_value = expo_iter[np.argmax(test_score_li)]

    # Instantiate the SVM class with the optimal C value
    clf = svm.SVC(kernel='rbf', C=optimal_c_value, probability=True)

    # train the model
    clf.fit(x_train, y_train)

    # training data
    train_data_score = round(clf.score(x_train, y_train) * 100, 1)
    print(f'Training data score = {train_data_score}%')

    # test data
    test_data_score = round(clf.score(x_test, y_test) * 100, 1)
    print(f'Test data score = {test_data_score}% \n')

    return clf

ml_10_svm = svm_train(df_ml_10)
ml_5_svm = svm_train(df_ml_5)

with open('ml_model_build_support_vector_machine/ml_models/svm_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_svm, myFile)

with open('ml_model_build_support_vector_machine/ml_models/svm_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_svm, myFile)


#---------- MODEL EVALUATION ----------

#cross validation
cv_score_av = round(np.mean(cross_val_score(ml_10_svm, x_10, y_10, cv=5))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_svm, x_5, y_5, cv=5))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')

#prediction probability plots
#fig = pred_proba_plot(ml_10_svm, x_10, y_10, no_iter=50, no_bins=35, x_min=0.3, classifier='Support Vector Machine (ml_10)')
#fig.savefig('figures/svm_pred_proba_ml10_50iter.png')

#fig = pred_proba_plot(ml_5_svm, x_5, y_5, no_iter=50, no_bins=35, x_min=0.3, classifier='Support Vector Machine (ml_5)')
#fig.savefig('figures/svm_pred_proba_ml5_50iter.png')



    
# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')

