import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
# some helper functions for plotting
from matplotlib.patches import Ellipse
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier

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


def winner_party(clf, x_test):
    y_test_pred_probability = np.mean(clf.predict_proba(x_test), axis=0)
    winner_pred = np.argmax(y_test_pred_probability)
    print("The predicted winner of the elections is: " + from_num_to_label[winner_pred])
    plt.plot(y_test_pred_probability, "red")
    plt.title("Test predicted vote probabilities")
    plt.show()


def print_cross_val_accuracy(sgd_clf, x_train, y_train):
    k_folds = 10
    cross_val_scores = cross_val_score(sgd_clf, x_train, y_train, cv=k_folds, scoring='accuracy')
    print("accuracy in each fold:")
    print(cross_val_scores)
    print("mean training accuracy:")
    print(cross_val_scores.mean())
    print()
    return cross_val_scores.mean()


def vote_division(y_pred_test, y_train):
    pred_values = []
    for i, label in from_num_to_label.items():
        result_true = len(y_pred_test[y_pred_test == i])
        all_results = len(y_pred_test)
        ratio = (result_true / all_results) * 100
        pred_values.append(ratio)

    plt.figure(figsize=(5, 5))
    colors = ["blue", "brown", "green", "grey", "khaki", "orange",
              "pink", "purple", "red", "turquoise", "violet", "white", "yellow"]
    explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0]
    plt.pie(pred_values, labels=labels, autopct="%.1f%%", explode=explode, colors=colors)
    plt.title("Test prediction vote division cart")
    plt.show()

    real_values = []
    for i, label in from_num_to_label.items():
        num_res = len(y_train[y_train == i])
        all_results = len(y_train)
        ratio = (num_res / all_results) * 100
        real_values.append(ratio)

    plt.figure(figsize=(5, 5))
    plt.pie(real_values, labels=labels, autopct="%.1f%%", explode=explode, colors=colors)
    plt.title("Real vote division cart")
    plt.show()


def save_voting_predictions(y_test_pred):
    y_test_pred_labels = [from_num_to_label[x] for x in y_test_pred]
    df_y_test_pred_labels = pd.DataFrame(y_test_pred_labels)
    df_y_test_pred_labels.to_csv('test_voting_predictions.csv', index=False)


def train_some_models(x_train, y_train, x_validation, y_validation):
    ret = list()

    print("SGDClassifier")
    sgd_clf = SGDClassifier(random_state=92)
    sgd_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(sgd_clf, x_validation, y_validation)
    ret.append(("SGDClassifier", sgd_clf, acc))

    return ret


def calculate_overall_test_error(y_test, y_test_pred):
    overall_test_error = 1 - len(y_test[y_test_pred == y_test]) / len(y_test)
    print("overall test error is: ")
    print((overall_test_error * 100).__str__() + "%")


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


def plot_feature_variance(features, coalition_feature_variance, title):
    plt.barh(features, coalition_feature_variance)
    plt.title(title)
    plt.show()


def generate_kmeans_models():
    for num_of_clusters in [2, 3, 4]:
        model_name = f"KMean_with_{num_of_clusters}_clusters"
        cluster = KMeans(num_of_clusters, max_iter=2500, random_state=0)
        yield model_name, cluster


def choose_hyper_parameter(models, x, y):
    best_score = float('-inf')
    best_model = None
    for name, model in models:
        score = print_cross_val_accuracy(model, x, y)
        if score > best_score:
            best_score = score
            best_model = name, model, best_score
    return best_model


def plot_data(X):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], s=50, c='b')
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("some random data")


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


def predict_coalition_from_test(x_test, y_test, threshold):
    test_coalition = []
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=600, n_init=10, random_state=0)
    two_group_labels_test = get_groups_label_using_kmeans(x_test, kmeans)

    results_test = calc_ratio_in_coalition(two_group_labels_test, y_test)
    print(results_test)

    coalition, opposition = get_coalition_opposition(results_test, threshold)
    print("the coalition is: ")
    print(coalition)
    print("the opposition is: ")
    print(opposition)
    labels_ratio = y_test.value_counts(normalize=True)
    print("labels ratio is: ")
    print(labels_ratio)
    print_ratio_with_label(labels_ratio)
    size_coalition = calc_size_coalition(coalition, labels_ratio)

    print(size_coalition)
    return test_coalition


def main():
    x_test, x_train, x_validation, y_test, y_train, y_validation = load_prepared_data()

    print_variance_before_choose_coalition(x_train)
    coalition_by_k_means_clustering = coalition_by_k_means_cluster(x_test, x_train, x_validation,
                                                                   y_test, y_train, y_validation)
    print_variance_after_choose_coalition(coalition_by_k_means_clustering, x_train)

    # X = generate_blobs()
    # plot_blobs(X)
    # run_plot_gmm(X)

    # get_coalition_by_generative(df_train, df_val, df_test, y_pred_test)
    # get_strongest_features_by_dt(df_train)
    # build_stronger_coalition(df_train, df_val)
    # build_alternative_coalition(df_train, df_val, classifier)


def print_variance_before_choose_coalition(x_train):
    x_train_var = x_train.var(axis=0)[right_feature_set]
    plot_feature_variance(right_feature_set, x_train_var, "feature_variance")


def print_variance_after_choose_coalition(coalition_by_k_means_clustering, x_train):
    coalition_index = []
    for party in coalition_by_k_means_clustering:
        coalition_index.append(from_label_to_num[party])
    x_train_coalition = x_train.loc[coalition_index]
    x_train_coalition_var = x_train_coalition.var(axis=0)[right_feature_set]
    plot_feature_variance(right_feature_set, x_train_coalition_var, "coalition_feature_variance")


def generate_blobs():
    X, y_true = make_blobs(n_samples=400, centers=4,
                           cluster_std=0.60, random_state=0)
    X = X[:, ::-1]  # flip axes for better plotting
    return X


def plot_blobs(X):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], s=40, cmap='viridis')
    ax.grid()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def run_plot_gmm(X):
    gmm = GaussianMixture(n_components=4, random_state=42)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    plot_gmm(gmm, X, ax=ax)


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


def get_coalition_by_clustering(kmeans, x_train, x_validation, y_train, y_validation, k, threshold, x_test, y_test):
    # for train
    print("################################################################")
    print("################################################################")
    print("########################  train  ###############################")
    k_group_labels_train = get_groups_label_using_kmeans(x_train, kmeans)
    dict_k_train = calc_ratio_in_coalition(k_group_labels_train, y_train, k)
    res_size_coalition = []
    for i in range(k):
        size_coalition = print_group_i(i, dict_k_train, y_train, threshold)
        res_size_coalition.append((size_coalition, i))

    coalition_train = print_max_group(dict_k_train, res_size_coalition, y_train, threshold)

    # for validation
    print("################################################################")
    print("################################################################")
    print("#####################  validation  #############################")
    k_group_labels_validation = get_groups_label_using_kmeans(x_validation, kmeans)
    dict_k_validation = calc_ratio_in_coalition(k_group_labels_validation, y_validation, k)
    res_size_coalition = []
    for i in range(k):
        size_coalition = print_group_i(i, dict_k_validation, y_validation, threshold)
        res_size_coalition.append((size_coalition, i))

    coalition_validation = print_max_group(dict_k_validation, res_size_coalition, y_validation, threshold)

    # final coalition
    print("################################################################")
    print("################################################################")
    print("###################  final coalition  ##########################")
    print("final coalition is like train and validation coalition")
    final_coalition = coalition_train
    print(final_coalition)

    # for test
    print("################################################################")
    print("################################################################")
    print("#####################  test  #############################")
    k_group_labels_test = get_groups_label_using_kmeans(x_test, kmeans)
    dict_k_test = calc_ratio_in_coalition(k_group_labels_test, y_test, k)
    res_size_coalition = []
    for i in range(k):
        size_coalition = print_group_i(i, dict_k_test, y_test, threshold)
        res_size_coalition.append((size_coalition, i))

    coalition_test = print_max_group(dict_k_test, res_size_coalition, y_test, threshold)

    print()
    print("the final coalition is stable?")
    print("the answer is: ")
    print(final_coalition == coalition_test)

    return final_coalition


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


def print_ratio_with_label(labels_ratio):
    index_label_ration = []
    for index in range(13):
        index_label_ration.append((from_num_to_label[index], labels_ratio[index]))
    print(index_label_ration)


def calc_size_coalition(coalition, labels_ratio):
    sum_ratio = 0
    for label in coalition:
        index_label = from_label_to_num[label]
        sum_ratio += labels_ratio[index_label]

    return sum_ratio


def average_list(results_train, results_validation):
    results_average = []
    for f, b in zip(results_train, results_validation):
        label_train, ratio_train = f
        label_validation, ratio_validation = b
        average = (ratio_validation + ratio_train) / 2
        results_average.append((label_train, average))
    return results_average


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


def gaussian_nb_hyperparametrs_tuning(x_train, y_train, k_fold: int = 5):
    guassien_naive_base = (
        GaussianNB(var_smoothing=1e-7),
        GaussianNB(var_smoothing=1e-8),
        GaussianNB(var_smoothing=1e-9),
        GaussianNB(var_smoothing=1e-10)
    )
    best_score = float('-inf')
    best_clf = None
    for clf in guassien_naive_base:
        _score = score(x_train=x_train, y_train=y_train, clf=clf, k=k_fold)
        if _score > best_score:
            best_score = _score
            best_clf = clf

    return best_clf, best_score


def get_coalition_by_generative(x_train, x_validation, x_test, y_test):
    gaussian_nb_clf, gaussian_nb_score = gaussian_nb_hyperparametrs_tuning(df_train)

    labels_guassian_mean = labels_generative_mean(df_train, gaussian_nb_clf)

    x_val, y_val = divide_data(df_val)
    naive_base_coalitions = build_coalition_using_generative_data(y_val, labels_guassian_mean)
    qda_coalitions = build_coalition_using_generative_data(y_val, labels_qda_mean)

    coalition_nb, coalition_nb_feature_variance = get_most_homogeneous_coalition(df_val, naive_base_coalitions)
    coalition_nb_size = get_coalition_size(y_val, coalition_nb[1])
    print(f"coalition using Gaussian Naive Base model is {coalition_nb} with size of {coalition_nb_size}")
    plot_feature_variance(selected_numerical_features, coalition_nb_feature_variance)


    coalitions_generative = build_coalition_using_generative_data(y_test, labels_guassian_mean)
    coalitions_generative, coalition_feature_variance = get_most_homogeneous_coalition(df_test, coalitions_generative)
    coalitions_generative_size = get_coalition_size(y_test, coalitions_generative[1])
    print(f"TEST coalition using Gaussian Naive Base model is {coalitions_generative} with size of {coalitions_generative_size}")
    plot_feature_variance(selected_numerical_features, coalition_qda_feature_variance)

    return coalitions_generative


if __name__ == '__main__':
    main()
