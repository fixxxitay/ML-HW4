import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

import implements_the_modeling as imp

right_feature_set = ["Yearly_IncomeK", "Number_of_differnt_parties_voted_for", "Political_interest_Total_Score",
                     "Avg_Satisfaction_with_previous_vote", "Avg_monthly_income_all_years",
                     "Most_Important_Issue", "Overall_happiness_score", "Avg_size_per_room",
                     "Weighted_education_rank"]

from_num_to_label = {0: 'Blues', 1: 'Browns', 2: 'Greens', 3: 'Greys', 4: 'Khakis',
                     5: 'Oranges', 6: 'Pinks', 7: 'Purples', 8: 'Reds',
                     9: 'Turquoises', 10: 'Violets', 11: 'Whites', 12: 'Yellows'}

from_label_to_num = {'Blues': 0, 'Browns': 1, 'Greens': 2, 'Greys': 3, 'Khakis': 4,
                     'Oranges': 5, 'Pinks': 6, 'Purples': 7, 'Reds': 8,
                     'Turquoises': 9, 'Violets': 10, 'Whites': 11, 'Yellows': 12}

labels = ['Blues', 'Browns', 'Greens', 'Greys', 'Khakis',
          'Oranges', 'Pinks', 'Purples', 'Reds',
          'Turquoises', 'Violets', 'Whites', 'Yellows']


def load_prepared_data():
    # Load the prepared training set
    df_prepared_train = pd.read_csv("prepared_train.csv")
    # shuffle
    df_prepared_train = df_prepared_train.sample(frac=1).reset_index(drop=True)
    x_train = df_prepared_train.drop("Vote", 1)
    y_train = df_prepared_train["Vote"]
    # Load the prepared validation set
    df_prepared_validation = pd.read_csv("prepared_validation.csv")
    # shuffle
    df_prepared_validation = df_prepared_validation.sample(frac=1).reset_index(drop=True)
    x_validation = df_prepared_validation.drop("Vote", 1)
    y_validation = df_prepared_validation["Vote"]
    # Load prepared test set
    df_prepared_test = pd.read_csv("prepared_test.csv")
    # shuffle
    df_prepared_test = df_prepared_test.sample(frac=1).reset_index(drop=True)
    x_test = df_prepared_test.drop("Vote", 1)
    y_test = df_prepared_test["Vote"]
    return x_test, x_train, x_validation, y_test, y_train, y_validation


def draw_variances(features, variances, title):
    plt.title(title)
    plt.barh(features, variances)
    plt.show()


def calc_ratio_in_coalition(k_group_labels, y_train, k):
    dict_k = {}
    position = 1
    for group_index in range(k):
        index_coalition = which_group_is_bigger(k_group_labels, position, k)
        results = []
        for i in range(13):
            str_label = from_num_to_label[i]
            y_label_i = y_train.loc[y_train == i]
            y_label_i_clusters = k_group_labels[y_label_i.index]
            y_label_i_clusters_equal_to_index_coalition = y_label_i_clusters[y_label_i_clusters == index_coalition]
            res_label_i_equal_to_index_coalition = (len(y_label_i_clusters_equal_to_index_coalition) / len(y_label_i))
            results.append((str_label, res_label_i_equal_to_index_coalition))
        dict_k[position - 1] = results
        position += 1
    return dict_k


def get_groups_label_using_kmeans(x_train, kmeans):
    kmeans.fit(x_train)
    return kmeans.labels_


def print_variance_before_choose_coalition(x_train):
    x_train_var = x_train.var(axis=0)[right_feature_set]
    draw_variances(right_feature_set, x_train_var, "feature_variance")

def get_data_for_coalition(coalition, x, y):
    coalition_index = []
    for party in coalition:
        coalition_index.append(from_label_to_num[party])

    data = []
    i = 0
    for a in y:
        if a in coalition_index:
            data.append(x.iloc[i, :])
        i += 1

    data = np.array(data)

    return data.mean(axis=0)

def print_variance_after_choose_coalition(coalition, x_train, y_train):
    coalition_index = []
    for party in coalition:
        coalition_index.append(from_label_to_num[party])

    data = []
    i = 0
    for a in y_train:
        if a in coalition_index:
            data.append(x_train.iloc[i, :])
        i += 1
    
    data = np.array(data)
    x_train_coalition_var = data.var(axis=0)

    print("AVG Variance: ", np.mean(x_train_coalition_var))
    draw_variances(right_feature_set, x_train_coalition_var, "coalition_feature_variance")


def coalition_by_k_means_cluster(x_test, x_train, x_validation, y_test, y_train, y_validation):
    k = 3
    threshold = 0.45
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=600, n_init=10, random_state=0)
    coalition_by_k_means_clustering = get_coalition_cluster(kmeans, x_train, x_validation,
                                                                  y_train, y_validation, k, threshold
                                                                  , x_test, y_test)
    return coalition_by_k_means_clustering


def print_group_i(i, dict_k_train, y_train, threshold):
    group_i = dict_k_train[i]
    coalition, opposition = get_coalition_opposition(group_i, threshold)
    labels_ratio = y_train.value_counts(normalize=True)
    size_coalition = calc_size_coalition(coalition, labels_ratio)
    return size_coalition

def test_coalition(x, y, coalition):
    #test the coalition
    model = GaussianNB()
    model.fit(x, y)
    totalVote = 0
    for c in coalition:
        totalVote += model.class_prior_[from_label_to_num[c]]
    
    if totalVote >= 0.51:
        print("The chosen coalition is stable")
    else:
        print("The chosen coalition is NOT stable :(")

    print("Percentage of vote for selected coalition: ", totalVote)


def get_coalition_cluster(kmeans, x_train, x_validation, y_train, y_validation, k, threshold, x_test, y_test):
    x_train = x_train.append(x_validation).reset_index(drop=True)
    y_train = y_train.append(y_validation).reset_index(drop=True)
    
    print("Automatically choosing coalition by clustering")
    k_group_labels_train = get_groups_label_using_kmeans(x_train, kmeans)
    
    dict_k_train = calc_ratio_in_coalition(k_group_labels_train, y_train, k)
    res_size_coalition = []
    for i in range(k):
        size_coalition = print_group_i(i, dict_k_train, y_train, threshold)
        res_size_coalition.append((size_coalition, i))

    coalition_train = print_max_group(dict_k_train, res_size_coalition, y_train, threshold)
    print("The coalition chosen by clustering is ")
    print(coalition_train)

    test_coalition(x_test, y_test, coalition_train)

    return coalition_train


def print_max_group(dict_k_train, res_size_coalition, y_train, threshold):
    max_index_group = 0
    max_size_coalition = 0
    for size_coalition, i in res_size_coalition:
        if size_coalition > max_size_coalition:
            max_index_group = i
            max_size_coalition = size_coalition

    #print_group_i(max_index_group, dict_k_train, y_train, threshold)
    group_max = dict_k_train[max_index_group]

    coalition, opposition = get_coalition_opposition(group_max, threshold)
   
    labels_ratio = y_train.value_counts(normalize=True)
    coalition_size = calc_size_coalition(coalition, labels_ratio)
    
    return coalition

def calc_size_coalition(coalition, labels_ratio):
    sum_ratio = 0
    for label in coalition:
        index_label = from_label_to_num[label]
        sum_ratio += labels_ratio[index_label]

    return sum_ratio


def get_coalition_opposition(results_average, threshold):
    coalition, opposition = [], []
    for f in results_average:
        label, ratio_average = f
        if ratio_average >= threshold:
            coalition.append(label)
        else:
            opposition.append(label)

    return coalition, opposition


def which_group_is_bigger(two_group_labels_train, position, k):
    len_list = []
    for i in range(k):
        total = 0
        for element in two_group_labels_train:
            if element == i:
                total += 1
        len_list.append(total)
    len_list.sort()
    return len_list.index(len_list[-position])

def get_clustering_coalition(x_test, x_train, x_validation, y_test, y_train, y_validation):
    print_variance_before_choose_coalition(x_train)
    coalition_clustering = coalition_by_k_means_cluster(x_test, x_train, x_validation,
                                                                y_test, y_train, y_validation)

    print_variance_after_choose_coalition(coalition_clustering, x_train, y_train)


def main():
    x_test, x_train, x_validation, y_test, y_train, y_validation = load_prepared_data()

    get_clustering_coalition(x_test, x_train, x_validation, y_test, y_train, y_validation)
    get_generative_coalition(x_test, x_train, x_validation, y_test, y_train, y_validation)
    get_every_party_lead_feat(x_train, y_train)
    get_strong_coalition(x_train, y_train, x_test, y_test)
   
def get_strong_coalition(x_train, y_train, x_test, y_test):
    coalition = ['Greens', 'Greys', 'Khakis', 'Oranges', 'Pinks', 'Reds', 'Turquoises', 'Whites', 'Yellows']
    x_backup = x_train.copy()
    
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_test_pred_probability = np.mean(model.predict_proba(x_test), axis=0)
    totalVote = 0
    for c in coalition:
        totalVote += y_test_pred_probability[from_label_to_num[c]]
    print(totalVote)

    list_features = list()
    max = totalVote
    for f in right_feature_set:
        x_train[f] += 1.5
        totalVote = 0
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_test_pred_probability = np.mean(model.predict_proba(x_test), axis=0)

        for c in coalition:
            totalVote += y_test_pred_probability[from_label_to_num[c]]
        
        print(f)
        print(totalVote)
        if totalVote <= max:
            x_train = x_backup.copy()
        else:
            x_backup = x_train.copy()
            list_features.append(f)
            max = totalVote

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_test_pred_probability = np.mean(model.predict_proba(x_test), axis=0)
    totalVote = 0
    for c in coalition:
        totalVote += y_test_pred_probability[from_label_to_num[c]]
    print("The stronger coalition gets a probability of ", totalVote)
    print("The modified features are ", list_features)
    return


def get_every_party_lead_feat(x_train, y_train):
    for party in labels:
        print(party)
        
        x_train_coalition_var = x_train[y_train == from_label_to_num[party]].var(axis=0)

        print(x_train.columns.values[x_train_coalition_var <= 0.5])
        print()



def get_generative_coalition(x_test, x_train, x_validation, y_test, y_train, y_validation):
    print("\nManually chosing the coalition using a generative model")
    model = GaussianNB()
    model.fit(x_train, y_train)

    i = 0
    coal = list()
    coal.append(from_num_to_label[np.argmax(model.class_prior_)])
    
    print("For each party, press 1 to keep or 0 to discard")
    for prob in model.class_prior_:
        if from_num_to_label[i] not in coal:
            coal.append(from_num_to_label[i])

            #Evaluate euclidian dist between coalition and opposition
            opposition = [l for l in labels if l not in coal]
            x1 = get_data_for_coalition(coal, x_validation, y_validation)
            x2 = get_data_for_coalition(opposition, x_validation, y_validation)
            print(np.linalg.norm(x1-x2))

            print_variance_after_choose_coalition(coal, x_validation, y_validation)

            res = input()
            if res != "1":
                coal.remove(from_num_to_label[i])
        i += 1

    print("The manual selection using the generative model gave us the following coalition:")
    print(coal)
    
    test_coalition(x_test, y_test, coal)

    print_variance_after_choose_coalition(coal, x_validation, y_validation)
    

if __name__ == '__main__':
    main()