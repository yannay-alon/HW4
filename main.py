import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from causalinference import CausalModel
from sklearn import metrics, calibration
from sklearn.linear_model import LogisticRegression, LinearRegression


def inverse_propensity_weighting(data: pd.DataFrame, show: bool = False):
    """IPW using random logistic regression"""
    features = data.drop(columns=["T", "Y"])
    treatments = data["T"]
    results = data["Y"]

    treatment_mask = treatments == 1

    model = LogisticRegression(max_iter=1e6)

    model.fit(features, treatments)
    propensity = model.predict_proba(features)

    if show:
        sns.histplot(x=propensity[:, 1], hue=treatment_mask)
        plt.show()

        fpr, tpr, thresholds = metrics.roc_curve(y_true=treatments, y_score=propensity[:, 1])
        sns.lineplot(x=fpr, y=tpr)
        plt.show()
        print(metrics.auc(fpr, tpr))

        prob_true, prob_pred = calibration.calibration_curve(treatments, propensity[:, 1], n_bins=10)
        sns.scatterplot(x=prob_true, y=prob_pred)
        sns.lineplot(x=np.arange(0, 1.1, 0.1), y=np.arange(0, 1.1, 0.1))
        plt.show()

    weighting = propensity[:, 1] / propensity[:, 0]

    treated = np.sum(results[treatment_mask]) / np.sum(treatments)
    untreated = np.sum(results[~treatment_mask] * weighting[~treatment_mask]) / np.sum(weighting[~treatment_mask])

    ATT = treated - untreated

    return ATT


def s_learner(data: pd.DataFrame, interacted: bool = True):
    """S learner - using KernelRidge"""
    original_features = data.drop(columns=["Y"])
    treatments = data["T"]
    results = data["Y"]

    def re_calc(T: int = None):
        features = original_features.copy()
        if T is not None:
            features["T"] = T
        if interacted:
            features[[f"T_{column}" for column in features.columns]] = features.apply(lambda row: row * row['T'],
                                                                                      axis=1)
        return features

    model = KernelRidge(kernel="poly", degree=3)
    model.fit(re_calc(), results)
    predictions = model.predict(re_calc())
    rmse = np.sqrt(np.mean(np.power(predictions - results, 2)))
    print(f"\t{rmse=}")

    predictions_1 = model.predict(re_calc(1))
    predictions_0 = model.predict(re_calc(0))

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

    model_0.fit(features[~treatment_mask], results[~treatment_mask])
    model_1.fit(features[treatment_mask], results[treatment_mask])

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

    knn = NearestNeighbors(n_neighbors=num_neighbors, metric="mahalanobis", metric_params={"VI": data_covariance})
    knn.fit(features[~treatment_mask], results[~treatment_mask])
    _, treated_neighbors = knn.kneighbors(features[treatment_mask])

    ite_treated = results[treatment_mask].to_numpy() - np.take_along_axis(
        results[~treatment_mask].to_numpy().reshape(-1, 1), treated_neighbors, axis=0).mean(axis=1)
    ATT = np.mean(ite_treated)

    return ATT


def comp_att(data: pd.DataFrame):
    def doubly_robust(df, X, T, Y):
        ps = LogisticRegression(max_iter=1e6).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
        mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
        mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
        treatment_mask = df[T] == 1
        return (
                np.mean(
                    df[treatment_mask][T] * (df[treatment_mask][Y] - mu1[treatment_mask]) / ps[treatment_mask] + mu1[
                        treatment_mask])
                - np.mean(
            (1 - df[treatment_mask][T]) * (df[treatment_mask][Y] - mu0[treatment_mask]) / (1 - ps[treatment_mask]) +
            mu0[treatment_mask])
        )

    features = data.drop(columns=["Y", "T"])
    treatments = data["T"]
    results = data["Y"]

    model = CausalModel(results.values, treatments.values, features.values)
    model.est_via_matching()
    print(model.estimates)
    model.reset()
    model.est_via_ols()
    print(model.estimates)

    T = 'T'
    Y = 'Y'
    X = data.columns.drop([T, Y])
    print(doubly_robust(data, X, T, Y))


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
    comp_att(df_1)

    print("Data 2:")
    print(f"\tIPW: {inverse_propensity_weighting(df_2)}")
    print(f"\tS Learner: {s_learner(df_2)}")
    print(f"\tT Learner: {t_learner(df_2)}")
    print(f"\tMatching: {matching(df_2)}")
    comp_att(df_2)


def propensity_saver():
    """save propensity score"""

    path_data_1 = r"data/data1.csv"
    path_data_2 = r"data/data2.csv"

    df_1 = pd.read_csv(path_data_1)
    df_2 = pd.read_csv(path_data_2)

    df_1 = pd.get_dummies(df_1)
    df_2 = pd.get_dummies(df_2)

    features = df_1.drop(columns=["T", "Y"])
    treatments = df_1["T"]
    model = LogisticRegression(max_iter=1e6)
    model.fit(features, treatments)
    propensity_1 = model.predict_proba(features)[:, 1]

    features = df_2.drop(columns=["T", "Y"])
    treatments = df_2["T"]
    model = LogisticRegression(max_iter=1e6)
    model.fit(features, treatments)
    propensity_2 = model.predict_proba(features)[:, 1]

    propensities = np.vstack([propensity_1, propensity_2])
    propensities = pd.DataFrame(propensities).T
    propensities.columns = ['data1', 'data2']
    propensities = propensities.T
    propensities.to_csv("models_propensity.csv")


if __name__ == '__main__':
    propensity_saver()
