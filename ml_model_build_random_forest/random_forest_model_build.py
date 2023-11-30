# -*- coding: utf-8 -*-


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

from ml_functions.ml_model_eval import pred_proba_plot
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict


plt.close('all')

#------------------------------- ML MODEL BUILD -------------------------------

#importing the data and creating the feature dataframe and target series

with open('../2019_prem_generated_clean/2019_prem_df_for_ml_5.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('../2019_prem_generated_clean/2019_prem_df_for_ml_10.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)


x_10 = df_ml_10.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
y_10 = df_ml_10['Team Av Corners']

x_5 = df_ml_5.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
y_5 = df_ml_5['Team Av Corners']


print('\nRANDOM FOREST\n')
#------------------------------- RANDOM FOREST --------------------------------

def rand_forest_train(df, is_classifier=True):

    #create features matrix
    x = df.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
    y = df['Team Av Corners']
    
    #instantiate the random forest class
    if is_classifier:
        # Instantiate the random forest classifier
        clf = RandomForestClassifier()
    else:
        # Instantiate the random forest regressor
        clf = RandomForestRegressor()

    #split into training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    #train the model
    clf.fit(x_train, y_train)
    
    #training data
    train_data_score = round(clf.score(x_train, y_train) * 100, 1)
    print(f'Training data score = {train_data_score}%')
    
    #test data
    test_data_score = round(clf.score(x_test, y_test) * 100, 1)
    print(f'Test data score = {test_data_score}% \n')
    
    return clf


ml_10_rand_forest = rand_forest_train(df_ml_10, is_classifier=False)  # Use regressor for ml_10
ml_5_rand_forest = rand_forest_train(df_ml_5, is_classifier=False)  # Use regressor for ml_5

with open('ml_models/random_forest_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_rand_forest, myFile)

with open('ml_models/random_forest_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_rand_forest, myFile)


#---------- MODEL EVALUATION ----------


#cross validation
cv_score_av = round(np.mean(cross_val_score(ml_10_rand_forest, x_10, y_10, cv=5))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_rand_forest, x_5, y_5, cv=5))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')

#prediction probability plots
fig = pred_proba_plot(ml_10_rand_forest, x_10, y_10, no_iter=5, no_bins=36, x_min=0.3, classifier='Random Forest (ml_10)')
#fig.savefig('figures/random_forest_pred_proba_ml10_50iter.png')

#fig = pred_proba_plot(ml_5_rand_forest, x_5, y_5, no_iter=50, no_bins=35, x_min=0.3, classifier='Random Forest (ml_5)')
#fig.savefig('figures/random_forest_pred_proba_ml5_50iter.png')




# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')