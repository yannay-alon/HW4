import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay


def plot_decision_boundary(data, estimator, label: str):
    # Plotting decision regions
    DecisionBoundaryDisplay.from_estimator(
        estimator, data.drop(columns=["T", "Y"]), alpha=0.4, response_method="predict"
    )
    plt.scatter(data[:, 0], data[:, 1], c=data[label], s=20, edgecolor="k")

    plt.show()


def inverse_propensity_weighting(data: pd.DataFrame):
    model = RandomForestClassifier(max_depth=10, random_state=0)

    features = data.drop(columns=["T", "Y"])
    labels = data["T"]

    model.fit(features, labels)
    probabilities = model.predict_proba(features)

    plot_decision_boundary(data, model, "T")

    print(probabilities)


def main():
    path_data_1 = r"data/data1.csv"
    path_data_2 = r"data/data2.csv"

    df_1 = pd.read_csv(path_data_1)
    df_2 = pd.read_csv(path_data_2)

    df_1 = pd.get_dummies(df_1)
    df_2 = pd.get_dummies(df_2)

    inverse_propensity_weighting(df_1)
    inverse_propensity_weighting(df_2)


if __name__ == '__main__':
    main()
