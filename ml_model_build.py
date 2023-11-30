# -*- coding: utf-8 -*-


print('\n\n')
print(' ---------------- START ---------------- \n')

#-------------------------------- API-FOOTBALL --------------------------------

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
import lime
import lime.lime_tabular
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import StratifiedKFold


plt.close('all')

#------------------------------- ML MODEL BUILD -------------------------------


with open('2019_prem_generated_clean/2019_prem_df_for_ml_5.txt', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open('2019_prem_generated_clean/2019_prem_df_for_ml_10.txt', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)
    


#----------------------------- ML MODEL EVALUATION ----------------------------

#this section contains a series of functions which may be used for model evaluation

def pred_proba_plot(clf, x, y, cv=4, no_iter=5, no_bins=25, x_min=0.5, x_max=1, output_progress=True, classifier=''):
    y_pred_values = []

    for i in range(no_iter):
        if output_progress:
            if i % 2 == 0:
                print(f'completed {i} iterations')
        kf = KFold(n_splits=cv, shuffle=False)
        y_pred_iter = []  # Store predictions for each iteration separately
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf.fit(x_train, y_train)
            y_pred_cv = clf.predict(x_test)
            y_pred_iter.extend(y_pred_cv)  # Store predictions for each fold
        y_pred_values.append(y_pred_iter)  # Store predictions for each iteration

    # Flatten the list of lists
    y_pred_values_flat = [item for sublist in y_pred_values for item in sublist]

    bins = np.linspace(x_min, x_max, no_bins)
    fig, ax = plt.subplots()
    ax.hist(y_pred_values_flat, bins, alpha=0.5, edgecolor='#1E212A', color='blue', label='Predicted Values')
    ax.legend()
    fig.suptitle(f'{classifier} - Iterated {no_iter} Times', y=0.96, fontsize=16, fontweight='bold')
    ax.set(ylabel='Number of Occurrences',
           xlabel='Predicted Values')
    return fig

def lime_explanation(clf, x, y, index, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(x.values,
                                                      mode='regression',
                                                      feature_names=[str(name) for name in feature_names],
                                                      class_names=['Team AV Corneres'],
                                                      discretize_continuous=True)
    exp = explainer.explain_instance(x.iloc[index].values, clf.predict, num_features=len(feature_names))
    exp.show_in_notebook(show_table=True)


def plot_residuals(clf, x_test, y_test):
    residuals = y_test - clf.predict(x_test)
    plt.scatter(x_test.index, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Data Points')
    plt.ylabel('Residuals')
    plt.show()


def plot_feature_importance(clf, feature_names):
    feature_importance = clf.feature_importances_
    sorted_idx = feature_importance.argsort()[::-1]
    plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align="center")
    plt.xticks(range(len(feature_importance)), feature_names[sorted_idx], rotation='vertical')
    plt.title('Feature Importance')
    plt.show()


def plot_results(clf, x_train, y_train, x_test, y_test):
    # Plotting for training data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x_train.index, y_train, label='Actual')
    plt.scatter(x_train.index, clf.predict(x_train), label='Predicted')
    plt.title('Training Data: Actual vs Predicted')
    plt.xlabel('Data Points')
    plt.ylabel('Target Variable')
    plt.legend()

    # Plotting for test data
    plt.subplot(1, 2, 2)
    plt.scatter(x_test.index, y_test, label='Actual')
    plt.scatter(x_test.index, clf.predict(x_test), label='Predicted')
    plt.title('Test Data: Actual vs Predicted')
    plt.xlabel('Data Points')
    plt.ylabel('Target Variable')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return clf


# Define a custom scorer function
def custom_scorer(estimator, X, y):
    feature_names = X.columns if hasattr(X, 'columns') else None
    return estimator.score(X, y), feature_names


x_10 = df_ml_10.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
y_10 = df_ml_10['Team Av Corners'].astype(float)

x_5 = df_ml_5.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
y_5 = df_ml_5['Team Av Corners'].astype(float)





print('\nRANDOM FOREST\n')
#------------------------------- RANDOM FOREST --------------------------------

def rand_forest_train(df):
    # create features matrix
    x = df.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
    y = df['Team Av Corners'].astype(float)  # Use float for continuous target variable

    # instantiate the random forest regressor
    clf = RandomForestRegressor()  # Use RandomForestRegressor

    # split into training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    warnings.filterwarnings('ignore')
    # train the model
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred) ** 0.5
    print(mse)
    # training data
    train_data_score = round(clf.score(x_train, y_train) * 100, 1)
    print(f'Training data score = {train_data_score}%')

    # test data
    test_data_score = round(clf.score(x_test, y_test) * 100, 1)
    print(f'Test data score = {test_data_score}% \n')
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, feature_names=x_train.columns.values.tolist(),
                                                       class_names=['Team Av Corners'], verbose=True, mode='regression')
    j = 10
    exp = explainer.explain_instance(x_test.values[j], clf.predict, num_features=3)
    exp.show_in_notebook(show_table=True)
    exp.save_to_file("LIME.html")
    return clf
print("10 games avg")
ml_10_rand_forest = rand_forest_train(df_ml_10)
print("5 games avg")
ml_5_rand_forest = rand_forest_train(df_ml_5)
original_feature_names = x_10.columns.tolist()


with open('ml_model_build_random_forest/ml_models/random_forest_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_rand_forest, myFile)

with open('ml_model_build_random_forest/ml_models/random_forest_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_rand_forest, myFile)

ml_10_rand_forest = rand_forest_train(df_ml_10)
ml_5_rand_forest = rand_forest_train(df_ml_5)

# Extracting x_train, y_train, x_test, y_test from the dataframes
x_train_10, x_test_10, y_train_10, y_test_10 = train_test_split(df_ml_10.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1), df_ml_10['Team Av Corners'].astype(float), test_size=0.2)
x_train_5, x_test_5, y_train_5, y_test_5 = train_test_split(df_ml_5.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1), df_ml_5['Team Av Corners'].astype(float), test_size=0.2)

plot_results(ml_10_rand_forest, x_train_10, y_train_10, x_test_10, y_test_10)
plot_results(ml_5_rand_forest, x_train_5, y_train_5, x_test_5, y_test_5)

plot_residuals(ml_10_rand_forest, x_test_10, y_test_10)
plot_residuals(ml_5_rand_forest, x_test_5, y_test_5)

feature_names = x_10.columns  # Assuming x is your feature matrix
plot_feature_importance(ml_10_rand_forest, feature_names)
plot_feature_importance(ml_5_rand_forest, feature_names)

#---------- MODEL EVALUATION ----------

# Use the custom scorer in cross-validation
cv_score_av_10, feature_names_10 = custom_scorer(ml_10_rand_forest, x_10, y_10)
print('Cross-Validation Accuracy Score ML10: ', round(cv_score_av_10 * 100, 1), '%\n')

cv_score_av_5, feature_names_5 = custom_scorer(ml_5_rand_forest, x_5, y_5)
print('Cross-Validation Accuracy Score ML5: ', round(cv_score_av_5 * 100, 1), '%\n')

#prediction probability plots
#fig = pred_proba_plot(ml_10_rand_forest, x_10, y_10, no_iter=50, no_bins=36, x_min=0.3, classifier='Random Forest (ml_10)')
#fig.savefig('figures/random_forest_pred_values_ml10_50iter.png')

#fig = pred_proba_plot(ml_5_rand_forest, x_5, y_5, no_iter=50, no_bins=35, x_min=0.3, classifier='Random Forest (ml_5)')
#fig.savefig('figures/random_forest_pred_values_ml5_50iter.png')


# --------------- TESTING C PARAM ---------------

expo_iter = np.square(np.arange(0.1, 10, 0.1))


def testing_c_parms(df, iterable):
    training_score_li = []
    test_score_li = []
    for c in iterable:
        x = df.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
        y = df['Team Av Corners'].astype(int)
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
    x = df.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)

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
    clf = svm.SVR(C=optimal_c_value)

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





print('\nK NEAREST NEIGHBORS\n')
#----------------------------- K NEAREST NEIGHBORS ----------------------------


def k_nearest_neighbor_train(df):
    # create features matrix
    x = df.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
    y = df['Team Av Corners'].astype(int)

    # split into training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # instantiate the K Nearest Neighbor class
    clf = KNeighborsRegressor(n_neighbors=28)

    # train the model
    clf.fit(x_train, y_train)

    # training data
    train_data_score = round(clf.score(x_train, y_train) * 100, 1)
    print(f'Training data score = {train_data_score}%')

    # test data
    test_data_score = round(clf.score(x_test, y_test) * 100, 1)
    print(f'Test data score = {test_data_score}% \n')

    return clf


ml_10_knn = k_nearest_neighbor_train(df_ml_10)
ml_5_knn = k_nearest_neighbor_train(df_ml_5)

with open('ml_model_build_nearest_neighbor/ml_models/knn_model_5.pk1', 'wb') as myFile:
    pickle.dump(ml_5_knn, myFile)

with open('ml_model_build_nearest_neighbor/ml_models/knn_model_10.pk1', 'wb') as myFile:
    pickle.dump(ml_10_knn, myFile)


# --------------- TESTING N_NEIGHBORS PARAM ---------------

df = df_ml_10
x = df.drop(['Target Fixture ID', 'Team Av Corners', 'Opponent Av Corners'], axis=1)
y = df['Team Av Corners'].astype(int)
#split into training data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    

test_accuracy = []
for n in range(1, 50, 1):
    clf = KNeighborsRegressor(n_neighbors=n)
    clf.fit(x_train, y_train)
    test_accuracy.append(round(clf.score(x_test, y_test) * 100, 1))

fig, ax = plt.subplots()
ax.plot(range(1, 50, 1), test_accuracy)


#---------- MODEL EVALUATION ----------

#cross validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)

cv_score_av = round(np.mean(cross_val_score(ml_10_knn, x_10, y_10, cv=kf))*100,1)
print('Cross-Validation Accuracy Score ML10: ', cv_score_av, '%\n')

cv_score_av = round(np.mean(cross_val_score(ml_5_knn, x_5, y_5, cv=kf))*100,1)
print('Cross-Validation Accuracy Score ML5: ', cv_score_av, '%\n')


#prediction probability plots
#fig = pred_proba_plot(ml_10_knn, x_10, y_10, no_iter=50, no_bins=18, x_min=0.3, classifier='Nearest Neighbor (ml_10)')
#fig.savefig('figures/knn_pred_proba_ml10_50iter.png')

#fig = pred_proba_plot(ml_5_knn, x_5, y_5, no_iter=50, no_bins=18, x_min=0.3, classifier='Nearest Neighbor (ml_5)')
#fig.savefig('figures/knn_pred_proba_ml5_50iter.png')


    
# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')