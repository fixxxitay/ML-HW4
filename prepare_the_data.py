import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from features import *
from sklearn.model_selection import StratifiedShuffleSplit

right_feature_set = ["Vote", "Yearly_IncomeK", "Number_of_differnt_parties_voted_for", "Political_interest_Total_Score",
                     "Avg_Satisfaction_with_previous_vote", "Avg_monthly_income_all_years",
                     "Most_Important_Issue", "Overall_happiness_score", "Avg_size_per_room",
                     "Weighted_education_rank"]


def deterministic_split(df, train, test):
    df_train = df.iloc[0:round(len(df) * train), :]
    df_test = df.iloc[round(len(df) * train):round(len(df) * (train + test)), :]
    df_validation = df.iloc[round(len(df) * (train + test)):len(df), :]

    return df_train, df_test, df_validation


def save_files(df_train, df_test, df_validation):
    df_train.to_csv('prepared_train.csv', index=False)
    df_validation.to_csv('prepared_validation.csv', index=False)
    df_test.to_csv('prepared_test.csv', index=False)


def remove_na(df_train, df_test, df_validation):
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    df_validation = df_validation.dropna()

    return df_train, df_test, df_validation


def save_raw_data(df_test, df_train, df_validation):
    df_train.to_csv('raw_train.csv', index=False)
    df_test.to_csv('raw_test.csv', index=False)
    df_validation.to_csv('raw_validation.csv', index=False)


def complete_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, df_validation: pd.DataFrame) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_train = df_train[df_train >= 0]
    df_test = df_test[df_test >= 0]
    df_validation = df_validation[df_validation >= 0]

    for col in df_train.columns.values:
        if col == 'Vote':
            df_train[col].fillna(df_train[col].mode()[0], inplace=True)
            continue

        filler = None
        if col in nominal_features:
            filler = df_train[col].mode()[0]

        if col in integer_features:
            filler = round(df_train[col].mean())

        if col in float_features:
            filler = df_train[col].mean()

        df_train[col].fillna(filler, inplace=True)
        df_test[col].fillna(filler, inplace=True)
        df_validation[col].fillna(filler, inplace=True)

    return df_train, df_test, df_validation


def nominal_to_numerical_categories(df_test: pd.DataFrame, df_train: pd.DataFrame, df_validation: pd.DataFrame):
    # from nominal to Categorical
    df_train = df_train.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df_train = df_train.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)

    # from nominal to Categorical
    df_validation = df_validation.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df_validation = df_validation.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)

    # from nominal to Categorical
    df_test = df_test.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df_test = df_test.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)

    return df_test, df_train, df_validation


def apply_feature_selection(df_train, df_test, df_validation, feature_set):
    df_train = df_train[feature_set]
    df_test = df_test[feature_set]
    df_validation = df_validation[feature_set]

    return df_train, df_test, df_validation


def normalize(df_test: pd.DataFrame, df_train: pd.DataFrame, df_validation: pd.DataFrame):
    # min-max for uniform features
    #uniform_scaler = MinMaxScaler(feature_range=(-1, 1))
    #df_train[uniform_features_right_features] = uniform_scaler.fit_transform(df_train[uniform_features_right_features])
    #df_validation[uniform_features_right_features] = uniform_scaler.transform(df_validation[uniform_features_right_features])
    #df_test[uniform_features_right_features] = uniform_scaler.transform(df_test[uniform_features_right_features])

    # z-score for normal features
    normal_scaler = StandardScaler()
    df_train[normal_features_right_features] = normal_scaler.fit_transform(df_train[normal_features_right_features])
    df_validation[normal_features_right_features] = normal_scaler.transform(df_validation[normal_features_right_features])
    df_test[normal_features_right_features] = normal_scaler.transform(df_test[normal_features_right_features])

    #quick fix
    df_train[["Most_Important_Issue"]] = normal_scaler.fit_transform(df_train[["Most_Important_Issue"]])
    df_validation[["Most_Important_Issue"]] = normal_scaler.transform(df_validation[["Most_Important_Issue"]])
    df_test[["Most_Important_Issue"]] = normal_scaler.transform(df_test[["Most_Important_Issue"]])

    ##now everyone will be between -1 and 1
    #df_train[normal_features_right_features] = uniform_scaler.fit_transform(df_train[normal_features_right_features])
    #df_validation[normal_features_right_features] = uniform_scaler.transform(df_validation[normal_features_right_features])
    #df_test[normal_features_right_features] = uniform_scaler.transform(df_test[normal_features_right_features])

    return df_train, df_test, df_validation


def remove_outliers(threshold: float, df_train: pd.DataFrame, df_validation: pd.DataFrame, df_test: pd.DataFrame):
    mean = df_train[normal_features_right_features].mean()
    std = df_train[normal_features_right_features].std()

    z_train = (df_train[normal_features_right_features] - mean) / std
    z_val = (df_validation[normal_features_right_features] - mean) / std
    z_test = (df_test[normal_features_right_features] - mean) / std

    df_train[z_train.mask(abs(z_train) > threshold).isna()] = np.nan
    df_validation[z_val.mask(abs(z_val) > threshold).isna()] = np.nan
    df_test[z_test.mask(abs(z_test) > threshold).isna()] = np.nan

    return df_train, df_validation, df_test


def main():
    # first part - data preparation
    df = pd.read_csv("ElectionsData.csv")

    # split the data to train , test and validation
    df_train, df_test, df_validation = deterministic_split(df, 0.6, 0.2)

    # Save the raw data first
    save_raw_data(df_test, df_train, df_validation)

    # apply feature selection
    df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, right_feature_set)

    # Convert nominal types to numerical categories
    df_test, df_train, df_validation = nominal_to_numerical_categories(df_test, df_train, df_validation)

    # 1 - Imputation - Complete missing values
    df_train, df_test, df_validation = complete_missing_values(df_train, df_test, df_validation)

    # 2 - Data Cleansing
    # Outlier detection using z score

    threshold = 3  # .3
    df_train, df_validation, df_test = remove_outliers(threshold, df_train, df_validation, df_test)

    # Remove lines containing na values
    df_train, df_test, df_validation = remove_na(df_train, df_test, df_validation)

    # 3 - Normalization (scaling)
    df_train, df_test, df_validation = normalize(df_test, df_train, df_validation)

    # step number 3
    # Save the 3x2 data sets in CSV files
    # CSV files of the prepared train, validation and test data sets
    save_files(df_train, df_test, df_validation)


if __name__ == '__main__':
    main()
