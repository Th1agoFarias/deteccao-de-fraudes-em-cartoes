import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summary_statistics(df, features):

    results = []

    for feature in features:

        results.append(
            {
                "Feature": feature,
                "Mean": df[feature].mean(),
                "Median": df[feature].median(),
                "Std": df[feature].std(),
                "Min": df[feature].min(),
                "Max": df[feature].max(),
                "Missing Values": df[feature].isnull().sum(),
            }
        )

    return pd.DataFrame(results)


def bar_plot(df, features, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    counts = df[features].value_counts().sort_index()
    proportions = df[features].value_counts(normalize=True).sort_index()

    bars = ax.bar(
        proportions.index.astype(str),
        proportions.values,
    )

    for bar, count in zip(bars, counts.values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    ax.set_title(f"Distribuição de {features}")
    ax.set_xlabel(features)
    ax.set_ylabel("Proporção")

    return ax


def histogram_plot(df, features, ax_box=None, ax_hist=None):

    if ax_box is None or ax_hist is None:

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            gridspec_kw={"height_ratios": [1, 4]},
        )

        ax_box = axes[0]
        ax_hist = axes[1]

    sns.boxplot(
        x=df[features],
        ax=ax_box,
    )

    sns.histplot(
        data=df,
        x=features,
        kde=True,
        ax=ax_hist,
    )

    ax_box.set_title(f"Distribuição de {features}")
    ax_hist.set_xlabel(features)

    return ax_box, ax_hist


def boxplot_by_class(df, features, target, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=target,
        y=features,
        ax=ax,
    )

    ax.set_title(f"{features} por {target}")

    return ax


def kde_by_class(df, features, target, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    sns.kdeplot(
        data=df,
        x=features,
        hue=target,
        common_norm=False,
        ax=ax,
    )

    ax.set_title(f"Distribuição de {features} por {target}")

    return ax


def correlation_plot(df, target, top_n=15, ax=None):

    correlations = df.corr(numeric_only=True)[target].drop(target)

    correlations = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )

    plot_data = correlations.head(top_n).sort_values()

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        plot_data.index,
        plot_data.values,
    )

    ax.axvline(0)

    ax.set_title(f"Correlação das features com {target}")
    ax.set_xlabel("Correlação")

    return ax


def get_outlier_summary(df, features):

    results = []

    for feature in features:

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[feature] < lower) | (df[feature] > upper)

        results.append(
            {
                "Feature": feature,
                "Outliers": mask.sum(),
                "Percentage": mask.mean() * 100,
            }
        )

    return pd.DataFrame(results).sort_values(
        "Outliers",
        ascending=False,
    )


def plot_outlier_summary(outlier_df, top_n=15, ax=None):

    plot_data = outlier_df.head(top_n).sort_values("Outliers")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        plot_data["Feature"],
        plot_data["Outliers"],
    )

    ax.set_title("Features com maior quantidade de outliers")
    ax.set_xlabel("Quantidade de outliers")

    return ax
