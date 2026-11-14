import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def calculate_3_fairness_measures(process, x_test, y_test, y_pred):
    M = 0 # n males
    F = 0 # n females
    Y1_M = 0 # n males that got a 'yes'
    Y0_M = 0 # n males that got a 'no'
    Y1_F = 0 # n females that got a 'yes'
    Y0_F = 0 # n females that got a 'no'

    if process == 1:
        # read data
        df = pd.read_csv('data/compas-scores.csv')

        # counts the n of all of the variabels above
        for index, row in df.iterrows():
            if row['sex'] == "Male":
                M += 1
                if row['is_recid'] == 1:
                    Y1_M += 1
                elif row['is_recid'] == 0:
                    Y0_M += 1
            elif row['sex'] == "Female":
                F += 1
                if row['is_recid'] == 1:
                    Y1_F += 1
                elif row['is_recid'] == 0:
                    Y0_F += 1
        #print(M, F, Y1_M, Y0_M, Y1_F, Y0_F)

    elif process == 2:
        for index, row in y_test.iterrows():
            if row['sex'] == 1:
                M += 1
                if row['is_recid'] == 1:
                    Y1_M += 1
                elif row['is_recid'] == 0:
                    Y0_M += 1
            elif row['sex'] == 0:
                F += 1
                if row['is_recid'] == 1:
                    Y1_F += 1
                elif row['is_recid'] == 0:
                    Y0_F += 1

    elif process == 3:
        for gender, prediction in zip(y_test['sex'], y_pred):
            if gender == 1:
                M += 1
                if prediction == 1:
                    Y1_M += 1
                elif prediction == 0:
                    Y0_M += 1
            elif gender == 0:
                F += 1
                if prediction == 1:
                    Y1_F += 1
                elif prediction == 0:
                    Y0_F += 1

    # calculates the proportion of males and females that got a 'yes'
    P_Y1_M = Y1_M / M if M != 0 else np.nan
    P_Y1_F = Y1_F / F if F != 0 else np.nan

    # calculates the Demographic Parity Difference between males and females
    # that got a 'yes'
    DP_Diff = P_Y1_M - P_Y1_F
    print("Demographic Parity Difference:", DP_Diff)

    # calculates the Disparate Impact Ratio between males and females
    # that got a 'yes'
    DI_Ratio = P_Y1_M / P_Y1_F if P_Y1_F != 0 else np.nan
    print("Disparate Impact Ratio:", DI_Ratio)

    # for the Consistency, we need to calculate KNN, I choose KNN = 5
    # remove Y and S from data f
    # This needs to be done differently in all 3 processes (whole data
    # before mitigation: labels are taken from the actual classification
    # of the whole data-set; test data before mitigation: labels
    # are taken from the classification of the test-data
    # before it has gone through mitigation; and test data after
    # mitigation: the labels are taken from the predicted classifications.
    # That is why another if-function will be used for the labels.

    if process == 1:
        X = df.drop('is_recid', axis=1)
        X = X.drop('sex', axis=1)
        # convert categorical values into numbers
        X = pd.get_dummies(X)
        # scale features
        X = StandardScaler().fit_transform(X)
        labels = df['is_recid'].to_numpy()
    elif process == 2:
        X = x_test
        # using the test-set actual labels
        labels = y_test['is_recid'].to_numpy()
    elif process == 3:
        X = x_test
        # using the test-set predicted labels
        labels = np.asarray(y_pred)

    # get 5 nearest neighbours (it says 6 because it includes the selected
    # observation itself as well
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(X)

    distances, indices = knn.kneighbors(X)
    # 2D list of indexes of 5 NNs for each observation
    neighb = indices[:, 1:6]
    #print(len(neighb)) # n observations
    #print(neighb[0])

    # use the indices of the nearest neighbours of each row to calculate
    # the neighbour average
    neighbour_averages = []
    for indices in neighb:
        Kneighbour_classif_per_obs = []
        for index in indices: # loops 5 (k) times
            # puts the actual values representing 5 NNs of the observation
            Kneighbour_classif_per_obs.append(labels[index])
        # calculate the average neighbour and append it in neighbour_averages list
        average = sum(Kneighbour_classif_per_obs) / len(Kneighbour_classif_per_obs)
        neighbour_averages.append(average)
    # calculate the final differences between the predicted y of each
    # observation and their neighbour's average
    differences = []
    for i, neighbour_average in enumerate(neighbour_averages):
        differences.append(abs(labels[i] - neighbour_average))
    final_consistency = 1 - sum(differences) / len(differences)
    print("Consistency:", final_consistency)

    return DP_Diff, DI_Ratio, final_consistency
