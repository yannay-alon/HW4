import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import cross_val_score


def inverse_propensity_weighting(data: pd.DataFrame):
    features = data.drop(columns=["T", "Y"])
    treatments = data["T"]
    results = data["Y"]

    treatment_mask = treatments == 1

    model = RandomForestClassifier(max_depth=10, class_weight="balanced")

    model.fit(features, treatments)
    propensity = model.predict_proba(features)

    weighting = propensity[:, 1] / propensity[:, 0]

    treated = np.sum(results[treatment_mask]) / np.sum(treatments)
    untreated = np.sum(results[~treatment_mask] * weighting[~treatment_mask]) / np.sum(weighting[~treatment_mask])

    ATT = treated - untreated
    return ATT


def s_learner(data: pd.DataFrame):
    features = data.drop(columns=["Y"])
    features[[f"T_{column}" for column in features.columns]] = features.apply(lambda row: row * row["T"], axis=1)
    treatments = data["T"]
    results = data["Y"]

    model = RandomForestRegressor(max_depth=15, oob_score=True)
    # model = KernelRidge(kernel="rbf", gamma=0.1)

    model.fit(features, results)

    features["T"] = 1
    predictions_1 = model.predict(features)

    features["T"] = 0
    predictions_0 = model.predict(features)

    treatment_mask = treatments == 1
    ATT = np.mean(predictions_1[treatment_mask] - predictions_0[treatment_mask])

    print(f"MSE: {np.mean(np.square(model.predict(features) - results))}")

    return ATT


def t_learner(data: pd.DataFrame):
    features = data.drop(columns=["Y", "T"])
    treatments = data["T"]
    results = data["Y"]

    treatment_mask = treatments == 1

    model_0 = RandomForestRegressor(max_depth=5, oob_score=True)
    model_1 = RandomForestRegressor(max_depth=5, oob_score=True)
    # model_0 = KernelRidge(kernel="rbf", gamma=0.1)
    # model_1 = KernelRidge(kernel="rbf", gamma=0.1)

    model_0.fit(features[treatment_mask], results[treatment_mask])
    model_1.fit(features[~treatment_mask], results[~treatment_mask])

    predictions_0 = model_0.predict(features)
    predictions_1 = model_1.predict(features)

    ATT = np.mean(predictions_1[treatment_mask] - predictions_0[treatment_mask])

    return ATT


def matching(data: pd.DataFrame):
    features = data.drop(columns=["Y", "T"])
    treatments = data["T"]
    results = data["Y"]

    treatment_mask = treatments == 1

    distances = distance_matrix(features[treatment_mask], features[~treatment_mask])

    treated_neighbors = np.argmin(distances, axis=1)
    untreated_neighbors = np.argmin(distances, axis=0)

    ite_treated = results[treatment_mask].to_numpy() - results[treated_neighbors].to_numpy()
    ite_untreated = results[~treatment_mask].to_numpy() - results[untreated_neighbors].to_numpy()

    ATT = np.mean(np.concatenate([ite_treated, ite_untreated]))

    return ATT


def main():
    path_data_1 = r"data/data1.csv"
    path_data_2 = r"data/data2.csv"

    df_1 = pd.read_csv(path_data_1)
    df_2 = pd.read_csv(path_data_2)

    df_1 = pd.get_dummies(df_1)
    df_2 = pd.get_dummies(df_2)

    # print(f"IPW ATT: {inverse_propensity_weighting(df_1)}")
    # print(f"IPW ATT: {inverse_propensity_weighting(df_2)}")

    # print(f"S Learner ATT: {s_learner(df_1)}")
    # print(f"S Learner ATT: {s_learner(df_2)}")

    # print(f"T Learner ATT: {t_learner(df_1)}")
    # print(f"T Learner ATT: {t_learner(df_2)}")

    print(f"Matching ATT: {matching(df_1)}")
    print(f"Matching ATT: {matching(df_2)}")


if __name__ == '__main__':
    main()
