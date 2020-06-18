import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import automate_the_model_selection as auto
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

transportation_threshold = 0.8

from_num_to_label = {0: 'Blues', 1: 'Browns', 2: 'Greens', 3: 'Greys', 4: 'Khakis',
                     5: 'Oranges', 6: 'Pinks', 7: 'Purples', 8: 'Reds',
                     9: 'Turquoises', 10: 'Violets', 11: 'Whites', 12: 'Yellows'}

labels = ['Blues', 'Browns', 'Greens', 'Greys', 'Khakis',
          'Oranges', 'Pinks', 'Purples', 'Reds',
          'Turquoises', 'Violets', 'Whites', 'Yellows']

def print_separation_lab(text):
    print("################################################################")
    print("################################################################")
    print("########################  ", text, "  ###############################")


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

    print("KNeighborsClassifier")
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(knn_clf, x_validation, y_validation)
    ret.append(("KNeighborsClassifier", knn_clf, acc))

    print("DecisionTreeClassifier, min_samples_split=5 min_samples_leaf=2")
    dt_clf = DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_split=5,
                                    min_samples_leaf=2)
    dt_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(dt_clf, x_validation, y_validation)
    ret.append(("DecisionTreeClassifier, min_samples_split=5 min_samples_leaf=2", dt_clf, acc))

    print("DecisionTreeClassifier, min_samples_split=4")
    dt_clf = DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_split=4)
    dt_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(dt_clf, x_validation, y_validation)
    ret.append(("DecisionTreeClassifier, min_samples_split=4", dt_clf, acc))

    print("DecisionTreeClassifier2 - entropy, min_samples_split=3")
    dt_clf = DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_split=3)
    dt_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(dt_clf, x_validation, y_validation)
    ret.append(("DecisionTreeClassifier2 - entropy, min_samples_split=3", dt_clf, acc))

    print("DecisionTreeClassifier - gini")
    dt_clf = DecisionTreeClassifier(random_state=0, criterion='gini', min_samples_split=3)                                    
    dt_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(dt_clf, x_validation, y_validation)
    ret.append(("DecisionTreeClassifier - gini", dt_clf, acc))

    print("RandomForestClassifier - regular")
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=0)
    rf_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(rf_clf, x_validation, y_validation)
    ret.append(("RandomForestClassifier - regular", rf_clf, acc))

    print("RandomForestClassifier - gini")
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=0, criterion='gini')
    rf_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(rf_clf, x_validation, y_validation)
    ret.append(("RandomForestClassifier - gini", rf_clf, acc))

    print("RandomForestClassifier - entropy")
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=0, criterion='entropy')
    rf_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(rf_clf, x_validation, y_validation)
    ret.append(("RandomForestClassifier - entropy", rf_clf, acc))

    print("RandomForestClassifier - entropy, min_samples_split=3")
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=0, criterion='entropy', min_samples_split=3)
    rf_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(rf_clf, x_validation, y_validation)
    ret.append(("RandomForestClassifier - entropy, min_samples_split=3", rf_clf, acc))

    print("RandomForestClassifier - entropy, min_samples_split=5")
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=0, criterion='entropy', min_samples_split=5)
    rf_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(rf_clf, x_validation, y_validation)
    ret.append(("RandomForestClassifier - entropy, min_samples_split=5", rf_clf, acc))

    print("RandomForestClassifier - entropy, min_samples_split=5 min_samples_leaf=2")
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=0, criterion='entropy', min_samples_split=5,
                                    min_samples_leaf=2)
    rf_clf.fit(x_train, y_train)
    acc = print_cross_val_accuracy(rf_clf, x_validation, y_validation)
    ret.append(("RandomForestClassifier - entropy, min_samples_split=5 min_samples_leaf=2", rf_clf, acc))

    return ret

def calculate_overall_test_error(y_test, y_test_pred):
    overall_test_error = 1 - len(y_test[y_test_pred == y_test]) / len(y_test)
    print("overall test error is: ")
    print(overall_test_error)


def estimate_transportation(clf, x_test):
    y_test_pred_probability = clf.predict_proba(x_test)
    transport_dict = dict()
    for index in range(13):
        transport_dict[from_num_to_label[index]] = list()
    i_citizen = 0
    for citizen in y_test_pred_probability:
        i_label = 0
        for label_probability in citizen:
            if label_probability > transportation_threshold:
                transport_dict[from_num_to_label[i_label]].append(i_citizen)
            i_label += 1
        i_citizen += 1

    print(transport_dict)


def main():
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

    # Train and evaluate performances of multiple models
    models = train_some_models(x_train, y_train, x_validation, y_validation)

    # Select the best model for the prediction tasks
    best_model_clf = auto.find_best_model(models)

    # Load prepared test set
    df_prepared_test = pd.read_csv("prepared_test.csv")
    # shuffle
    df_prepared_test = df_prepared_test.sample(frac=1).reset_index(drop=True)
    x_test = df_prepared_test.drop("Vote", 1)
    y_test = df_prepared_test["Vote"]

    x_train_and_validation = x_train.append(x_validation).reset_index(drop=True)
    y_train_and_validation = y_train.append(y_validation).reset_index(drop=True)

    print("the best score from random forest on train + validation is:")
    print_cross_val_accuracy(best_model_clf, x_train_and_validation, y_train_and_validation)

    best_model_clf.fit(x_train_and_validation, y_train_and_validation)
    y_test_pred = best_model_clf.predict(x_test)

    # Use the selected model to provide the following:
    # vote division
    vote_division(y_test_pred, y_test)

    # the party that wins the elections is:
    print()
    winner_party(best_model_clf, x_test)
    print()

    # save
    save_voting_predictions(y_test_pred)

    # test confusion matrix
    plot_confusion_matrix(best_model_clf, x_test, y_test)
    plt.show()

    # overall test error
    calculate_overall_test_error(y_test, y_test_pred)

    # each party would like to suggest transportation services for its voters
    estimate_transportation(best_model_clf, x_test)


if __name__ == '__main__':
    main()
