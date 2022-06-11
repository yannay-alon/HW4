import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge

from causalinference import CausalModel
from causallib.estimation import IPW, Matching


def inverse_propensity_weighting(data: pd.DataFrame):
    """IPW using random forest classifier"""
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
    """S learner - using random forest regressor"""
    features = data.drop(columns=["Y"])
    # features[[f"T_{column}" for column in features.columns]] = features.apply(lambda row: row * row["T"], axis=1)
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

    return ATT


def t_learner(data: pd.DataFrame):
    """T learner using random forest regression"""
    features = data.drop(columns=["Y", "T"])
    treatments = data["T"]
    results = data["Y"]

    treatment_mask = treatments == 1

    model_0 = RandomForestRegressor(max_depth=15, oob_score=True)
    model_1 = RandomForestRegressor(max_depth=15, oob_score=True)
    # model_0 = KernelRidge(kernel="rbf", gamma=0.1)
    # model_1 = KernelRidge(kernel="rbf", gamma=0.1)

    model_0.fit(features[treatment_mask], results[treatment_mask])
    model_1.fit(features[~treatment_mask], results[~treatment_mask])

    predictions_0 = model_0.predict(features)
    predictions_1 = model_1.predict(features)

    ATT = np.mean(predictions_1[treatment_mask] - predictions_0[treatment_mask])

    return ATT


def matching(data: pd.DataFrame):
    """K-NN matching"""
    features = data.drop(columns=["Y", "T"])
    treatments = data["T"]
    results = data["Y"]

    treatment_mask = treatments == 1

    num_neighbors = 1

    data_covariance = np.cov(features.T)

    # knn = NearestNeighbors(n_neighbors=num_neighbors, metric="mahalanobis", metric_params={"V": data_covariance})
    # knn.fit(features[treatment_mask], results[treatment_mask])
    # _, untreated_neighbors = knn.kneighbors(features[~treatment_mask])

    knn = NearestNeighbors(n_neighbors=num_neighbors, metric="mahalanobis", metric_params={"V": data_covariance})
    knn.fit(features[~treatment_mask], results[~treatment_mask])
    _, treated_neighbors = knn.kneighbors(features[treatment_mask])

    ite_treated = results[treatment_mask].to_numpy() - np.take_along_axis(
        results[~treatment_mask].to_numpy().reshape(-1, 1), treated_neighbors, axis=0).mean(axis=1)
    # ite_untreated = results[~treatment_mask].to_numpy() - np.take_along_axis(
    #     results[treatment_mask].to_numpy().reshape(-1, 1), untreated_neighbors, axis=0).mean(axis=1)

    # ATT = np.mean(np.concatenate([ite_treated, ite_untreated]))
    ATT = np.mean(ite_treated)

    return ATT


def comp_att(data: pd.DataFrame):
    features = data.drop(columns=["Y", "T"])
    treatments = data["T"]
    results = data["Y"]

    treatment_mask = treatments == 1

    # ipw = IPW(RandomForestClassifier(max_depth=10, class_weight="balanced"))
    # ipw.fit(features, treatments)
    # print(
    #     f"""\tIPW (Forest) COMP ATT: {ipw.estimate_population_outcome(features[treatment_mask],
    #                                                                   treatments[treatment_mask], results[treatment_mask])}""")
    #
    # ipw = IPW(LogisticRegression(max_iter=1e6))
    # ipw.fit(features, treatments)
    # print(
    #     f"""\tIPW (LR) COMP ATT: {ipw.estimate_population_outcome(features[treatment_mask],
    #                                                               treatments[treatment_mask], results[treatment_mask])}""")
    #
    # model = Matching()
    # model.fit(features, treatments, results)
    # print(f"""Matching COMP ATT: {model.estimate_individual_outcome(features[treatment_mask],
    #                                                                 treatments[treatment_mask]).mean()}""")

    model = CausalModel(results.values, treatments.values, features.values)
    model.est_via_matching()
    print(model.estimates)
    model.reset()
    model.est_via_ols()
    print(model.estimates)
    model.reset()
    model.est_propensity_s()
    model.est_via_weighting()
    print(model.estimates)
    model.reset()



def main():
    path_data_1 = r"data/data1.csv"
    path_data_2 = r"data/data2.csv"

    df_1 = pd.read_csv(path_data_1)
    df_2 = pd.read_csv(path_data_2)

    df_1 = pd.get_dummies(df_1)
    df_2 = pd.get_dummies(df_2)

    print("Data 1:")
    print(f"\tIPW: {inverse_propensity_weighting(df_1)}")
    print(f"\tS Learner: {s_learner(df_1)}")
    print(f"\tT Learner: {t_learner(df_1)}")
    print(f"\tMatching: {matching(df_1)}")
    # comp_att(df_1)

    print("Data 2:")
    print(f"\tIPW: {inverse_propensity_weighting(df_2)}")
    print(f"\tS Learner: {s_learner(df_2)}")
    print(f"\tT Learner: {t_learner(df_2)}")
    print(f"\tMatching: {matching(df_2)}")
    # comp_att(df_2)


if __name__ == '__main__':
    main()
