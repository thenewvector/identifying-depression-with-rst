#distribution.py

from __future__ import annotations

from collections.abc import Hashable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _get_group_values(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    group_value: Hashable,
    *,
    positive_only: bool = False,
) -> pd.Series:
    """Return numeric values for one group after basic validation."""
    missing = {feature, group_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    values = pd.to_numeric(
        df.loc[df[group_col] == group_value, feature],
        errors="coerce",
    ).dropna()

    if positive_only:
        values = values[values > 0]

    return values


def _group_label(
    group_value: Hashable,
    group_labels: dict[Hashable, str] | None,
) -> str:
    """Resolve a display label for a group."""
    if group_labels is None:
        return f"Group {group_value}"
    return group_labels.get(group_value, f"Group {group_value}")


def plot_distribution_histogram(
    df: pd.DataFrame,
    feature: str,
    *,
    group_col: str = "ds_num",
    group_0: Hashable = 0,
    group_1: Hashable = 1,
    group_labels: dict[Hashable, str] | None = None,
    bins: int = 20,
    positive_only: bool = False,
    density: bool = True,
    alpha: float = 0.6,
    figsize: tuple[float, float] = (9, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot overlaid histograms of one numeric feature for two groups.

    The groups use identical bin boundaries. Density normalization is enabled
    by default because unequal group sizes otherwise distort bar heights.

    Parameters
    ----------
    df:
        DataFrame containing the feature and grouping columns.
    feature:
        Numeric column to plot, e.g. ``"causal"`` or ``"rel_entropy"``.
    group_col:
        Column containing group membership.
    group_0, group_1:
        Values identifying the two groups.
    group_labels:
        Optional mapping such as ``{0: "Control", 1: "Depression"}``.
    bins:
        Number of shared histogram bins.
    positive_only:
        Exclude zero and negative values.
    density:
        Normalize each histogram to probability density.
    alpha:
        Histogram transparency.
    figsize:
        Figure size when no existing axes are supplied.
    title, xlabel:
        Optional custom labels.
    ax:
        Existing Matplotlib axes.
    show:
        Call ``plt.show()`` before returning.

    Returns
    -------
    tuple[Figure, Axes]
        The Matplotlib figure and axes.
    """
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")

    values_0 = _get_group_values(
        df, feature, group_col, group_0, positive_only=positive_only
    )
    values_1 = _get_group_values(
        df, feature, group_col, group_1, positive_only=positive_only
    )

    combined = pd.concat([values_0, values_1], ignore_index=True)
    if combined.empty:
        suffix = " after excluding non-positive values" if positive_only else ""
        raise ValueError(f"No usable values found for {feature!r}{suffix}")

    value_min = float(combined.min())
    value_max = float(combined.max())

    if np.isclose(value_min, value_max):
        padding = max(abs(value_min) * 0.05, 0.01)
        value_min -= padding
        value_max += padding

    bin_edges = np.linspace(value_min, value_max, bins + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    label_0 = _group_label(group_0, group_labels)
    label_1 = _group_label(group_1, group_labels)

    ax.hist(
        values_0,
        bins=bin_edges,
        alpha=alpha,
        density=density,
        label=f"{label_0} (n={len(values_0)})",
    )
    ax.hist(
        values_1,
        bins=bin_edges,
        alpha=alpha,
        density=density,
        label=f"{label_1} (n={len(values_1)})",
    )

    default_title = f"Distribution of {feature} by group"
    if positive_only:
        default_title += " — positive values only"

    ax.set_title(title or default_title)
    ax.set_xlabel(xlabel or feature)
    ax.set_ylabel("Density" if density else "Number of documents")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_distribution_ecdf(
    df: pd.DataFrame,
    feature: str,
    *,
    group_col: str = "ds_num",
    group_0: Hashable = 0,
    group_1: Hashable = 1,
    group_labels: dict[Hashable, str] | None = None,
    positive_only: bool = False,
    figsize: tuple[float, float] = (9, 5),
    title: str | None = None,
    xlabel: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot empirical cumulative distribution functions for two groups.

    An ECDF shows the proportion of documents whose feature value is less than
    or equal to each x-value. Unlike a histogram, it does not depend on binning.
    """
    values_0 = _get_group_values(
        df, feature, group_col, group_0, positive_only=positive_only
    )
    values_1 = _get_group_values(
        df, feature, group_col, group_1, positive_only=positive_only
    )

    if values_0.empty and values_1.empty:
        suffix = " after excluding non-positive values" if positive_only else ""
        raise ValueError(f"No usable values found for {feature!r}{suffix}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for group_value, values in ((group_0, values_0), (group_1, values_1)):
        if values.empty:
            continue

        x = np.sort(values.to_numpy(dtype=float))
        y = np.arange(1, len(x) + 1, dtype=float) / len(x)
        label = _group_label(group_value, group_labels)

        ax.step(
            x,
            y,
            where="post",
            label=f"{label} (n={len(x)})",
        )

    default_title = f"ECDF of {feature} by group"
    if positive_only:
        default_title += " — positive values only"

    ax.set_title(title or default_title)
    ax.set_xlabel(xlabel or feature)
    ax.set_ylabel("Cumulative proportion of documents")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


__all__ = [
    "plot_distribution_histogram",
    "plot_distribution_ecdf",
]
