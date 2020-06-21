import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

import implements_the_modeling as imp

from features import right_feature_set


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


def print_variance_after_choose_coalition(coalition_by_k_means_clustering, x_train):
    coalition_index = []
    for party in coalition_by_k_means_clustering:
        coalition_index.append(from_label_to_num[party])
    x_train_coalition = x_train.loc[coalition_index]
    x_train_coalition_var = x_train_coalition.var(axis=0)[right_feature_set]
    draw_variances(right_feature_set, x_train_coalition_var, "coalition_feature_variance")


def coalition_by_k_means_cluster(x_test, x_train, x_validation, y_test, y_train, y_validation):
    k = 3
    threshold = 0.45
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=600, n_init=10, random_state=0)
    coalition_by_k_means_clustering = get_coalition_by_clustering(kmeans, x_train, x_validation,
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
        print("The generative coalition is stable")
    else:
        print("The generative coalition is NOT stable :(")

    print("Percentage of vote for selected coalition: ", totalVote)


def get_coalition_by_clustering(kmeans, x_train, x_validation, y_train, y_validation, k, threshold, x_test, y_test):
    x_train = x_train.append(x_validation).reset_index(drop=True)
    y_train = y_train.append(y_validation).reset_index(drop=True)

    imp.print_separation_lab("train")
    k_group_labels_train = get_groups_label_using_kmeans(x_train, kmeans)
    
    dict_k_train = calc_ratio_in_coalition(k_group_labels_train, y_train, k)
    res_size_coalition = []
    for i in range(k):
        size_coalition = print_group_i(i, dict_k_train, y_train, threshold)
        res_size_coalition.append((size_coalition, i))

    coalition_train = print_max_group(dict_k_train, res_size_coalition, y_train, threshold)

    test_coalition(x_test, y_test, coalition_train)

    return coalition_train


def print_max_group(dict_k_train, res_size_coalition, y_train, threshold):
    max_index_group = 0
    max_size_coalition = 0
    for size_coalition, i in res_size_coalition:
        if size_coalition > max_size_coalition:
            max_index_group = i
            max_size_coalition = size_coalition
    print()
    print("the max size group is: ")
    print_group_i(max_index_group, dict_k_train, y_train, threshold)
    group_max = dict_k_train[max_index_group]
    # print(group_max)
    coalition, opposition = get_coalition_opposition(group_max, threshold)
    print(coalition)
    labels_ratio = y_train.value_counts(normalize=True)
    coalition_size = calc_size_coalition(coalition, labels_ratio)
    print("this size of the coalition is: ")
    print(coalition_size)
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


def main():
    x_test, x_train, x_validation, y_test, y_train, y_validation = load_prepared_data()

    print_variance_before_choose_coalition(x_train)
    coalition_by_k_means_clustering = coalition_by_k_means_cluster(x_test, x_train, x_validation,
                                                                y_test, y_train, y_validation)

    print_variance_after_choose_coalition(coalition_by_k_means_clustering, x_train)


    get_generative_coalition(x_test, x_train, x_validation, y_test, y_train, y_validation)
   


def get_generative_coalition(x_test, x_train, x_validation, y_test, y_train, y_validation):
    model = GaussianNB()
    model.fit(x_train, y_train)
    print(model.class_prior_)
    i = 0
    coal = list()
    coal.append(from_num_to_label[np.argmax(model.class_prior_)])
    
    print("For each party, press 1 to keep or 0 to discard")
    for prob in model.class_prior_:
        if from_num_to_label[i] not in coal:
            coal.append(from_num_to_label[i])
            print_variance_after_choose_coalition(coal, x_validation)

            res = input()
            if res != "1":
                coal.remove(from_num_to_label[i])
        i += 1

    print("The manual selection using the generative model gave us the following coalition:")
    print(coal)
    
    test_coalition(x_test, y_test, coal)

    print_variance_after_choose_coalition(coal, x_test)
    

if __name__ == '__main__':
    main()