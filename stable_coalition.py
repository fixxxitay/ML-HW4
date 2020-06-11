import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from features import right_feature_set

from_num_to_label = {0: 'Blues', 1: 'Browns', 2: 'Greens', 3: 'Greys', 4: 'Khakis',
                     5: 'Oranges', 6: 'Pinks', 7: 'Purples', 8: 'Reds',
                     9: 'Turquoises', 10: 'Violets', 11: 'Whites', 12: 'Yellows'}

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


def plot_feature_variance(features, coalition_feature_variance, title="Coalition Feature Variance"):
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


def calc_ratio_in_coalition(two_group_labels, y_train):
    results = []
    index_coalition = which_group_is_bigger(two_group_labels)
    for i in range(13):
        str_label = from_num_to_label[i]
        y_label_i = y_train.loc[y_train == i]
        y_label_i_clusters = two_group_labels[y_label_i.index]
        y_label_i_clusters_equal_to_index_coalition = y_label_i_clusters[y_label_i_clusters == index_coalition]
        res_label_i_equal_to_index_coalition = (len(y_label_i_clusters_equal_to_index_coalition) / len(y_label_i))
        results.append((str_label, res_label_i_equal_to_index_coalition))

    return results


def show_groups_using_kmeans(x_train, kmeans):
    kmeans.fit(x_train)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()
    return kmeans.labels_


def choose_k(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def main():
    x_test, x_train, x_validation, y_test, y_train, y_validation = load_prepared_data()
    # x_train_var = x_train.var(axis=0)[right_feature_set]
    # plot_feature_variance(right_feature_set, x_train_var, "Feature Variance")

    # rfc = RandomForestClassifier(n_jobs=-1, random_state=0, criterion='entropy',
    #                              min_samples_split=3, min_samples_leaf=1, n_estimators=400)
    # rfc.fit(x_train, y_train)
    # y_pred_test = rfc.predict(x_test)
    # calculate_overall_test_error(y_test, y_pred_test)

    # cluster = KMeans(2, max_iter=2500, random_state=0)

    # choose_k(x_train)
    get_coalition_by_clustering(x_train, x_validation, y_train, y_validation)

    # get_coalition_by_generative(df_train, df_val, df_test, y_pred_test)
    #
    # get_strongest_features_by_dt(df_train)
    #
    # build_stronger_coalition(df_train, df_val)
    #
    # build_alternative_coalition(df_train, df_val, classifier)
    # models = generate_kmeans_models()
    # _name, _model, best_score = choose_hyper_parameter(models, x_train, y_train)
    # print(_name)
    # print(_model)
    # print(best_score)





def get_coalition_by_clustering(x_train, x_validation, y_train, y_validation):
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=600, n_init=10, random_state=0)
    two_group_labels_train = show_groups_using_kmeans(x_train, kmeans)
    train_coalition_index = which_group_is_bigger(two_group_labels_train)

    results_train = calc_ratio_in_coalition(two_group_labels_train, y_train)
    print(results_train)
    two_group_labels_validation = show_groups_using_kmeans(x_validation, kmeans)

    validation_coalition_index = which_group_is_bigger(two_group_labels_validation)
    results_validation = calc_ratio_in_coalition(two_group_labels_validation, y_validation)
    print(results_validation)
    coalition, opposition = [], []
    if (train_coalition_index == 0) & (validation_coalition_index == 0):
        coalition, opposition = get_coalition_opposition(0, results_train, results_validation)
    elif (train_coalition_index == 1) & (validation_coalition_index == 1):
        coalition, opposition = get_coalition_opposition(1, results_train, results_validation)
    else:
        print("no have correlation between train and validation")


def which_group_is_bigger(two_group_labels_train):
    len_0 = len(two_group_labels_train[two_group_labels_train == 0])
    len_1 = len(two_group_labels_train[two_group_labels_train == 1])
    if len_0 > len_1:
        return 0
    else:
        return 1


def get_coalition_opposition(coalition_index, train_res, validation_res):
    coalition, opposition = [], []

    
    return coalition, opposition


if __name__ == '__main__':
    main()
