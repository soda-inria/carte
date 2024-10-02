"""
Functions that can be utilized for visualization. 
For Critical difference diagram, it modifies some of the codes from scikit-posthocs.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict, Set
from matplotlib import colors
from matplotlib.axes import SubplotBase
from matplotlib.colorbar import ColorbarBase, Colorbar
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from pandas import DataFrame, Series
from seaborn import heatmap
from carte_ai.configs.carte_configs import carte_singletable_baseline_mapping
from carte_ai.configs.directory import config_directory


# Normalization function of the results
def _normalize(group):
    min_score = group["score"].min()
    max_score = group["score"].max()
    group["normalized_score"] = (group["score"] - min_score) / (max_score - min_score)
    return group


# Prepare dataframe suitable for the learning curves
def prepare_result(task, models="all", rank_at=2048):

    # load result
    result_dir = f"{config_directory['results']}/compiled_results/results_carte_baseline_singletable.csv"
    df_score = pd.read_csv(result_dir)

    # control for not important values
    mask_cls = df_score["task"] == "classification"
    temp = df_score["score"].copy()
    temp[np.logical_and(mask_cls, temp < 0.5)] = 0.5
    temp[np.logical_and(~mask_cls, temp < 0)] = 0
    df_score["score"] = temp

    # select results based on task
    mask = df_score["task"] == task
    df_score = df_score[mask].reset_index(drop=True)

    # select result with model of interest
    if models == "all":
        pass
    else:
        mask = df_score["model"].isin(models)
        df_score = df_score[mask]
        df_score.reset_index(drop=True, inplace=True)

    # Change the names of models for clarity
    temp = df_score["model"].copy()
    for key in carte_singletable_baseline_mapping:
        temp = temp.str.replace(key, carte_singletable_baseline_mapping[key])
    df_score["model"] = temp.copy()

    # Apply normalization on scores
    df_normalized = df_score.groupby(["data_name"], group_keys=True).apply(_normalize)
    df_normalized.reset_index(drop=True, inplace=True)

    # Ranking
    if rank_at == "all":
        temp = df_normalized["num_train"].astype(float)
        mask = temp <= max(temp)
        df_normalized_ = df_normalized[mask].copy()
        avg_rank = (
            df_normalized_.groupby("model")
            .normalized_score.mean()
            .rank(ascending=False)
        )
        avg_rank = avg_rank.sort_values()
        rank_order = avg_rank.index.tolist()
    else:
        mask = df_normalized["num_train"] == rank_at
        df_normalized_ = df_normalized[mask].copy()
        avg_rank = (
            df_normalized_.groupby("model")
            .normalized_score.mean()
            .rank(ascending=False)
        )
        avg_rank = avg_rank.sort_values()
        rank_order = avg_rank.index.tolist()

    df_normalized = df_normalized.sort_values(by="num_train", ascending=True)
    df_normalized["num_train"] = df_normalized["num_train"].astype(str)
    df_normalized.reset_index(drop=True, inplace=True)

    return df_normalized, rank_order


# Generate dataframe suitable for creating critical difference diagram
def generate_df_cdd(df_normalized, train_size="all"):

    # Set the base df
    df_cdd = df_normalized.copy()
    df_cdd["case"] = (
        df_normalized["data_name"]
        + "_"
        + df_normalized["num_train"].astype(str)
        + "_"
        + df_normalized["random_state"].astype(str)
    )

    # select the train_size for comparison
    if train_size == "all":
        return df_cdd
    else:
        mask = df_cdd["num_train"].str.contains(f"{train_size}")
        df_cdd = df_cdd[mask].copy()
        df_cdd.reset_index(drop=True, inplace=True)
        return df_cdd


# Sign array for scikit-posthoc
def sign_array(p_values: Union[List, np.ndarray], alpha: float = 0.05) -> np.ndarray:

    p_values = np.array(p_values)
    p_values[p_values > alpha] = 0
    p_values[(p_values < alpha) & (p_values > 0)] = 1
    np.fill_diagonal(p_values, 1)

    return p_values


# Sign table for scikit-posthoc
def sign_table(
    p_values: Union[List, np.ndarray, DataFrame], lower: bool = True, upper: bool = True
) -> Union[DataFrame, np.ndarray]:

    if not any([lower, upper]):
        raise ValueError("Either lower or upper triangle must be returned")

    pv = (
        DataFrame(p_values, copy=True)
        if not isinstance(p_values, DataFrame)
        else p_values.copy()
    )

    ns = pv > 0.05
    three = (pv < 0.001) & (pv >= 0)
    two = (pv < 0.01) & (pv >= 0.001)
    one = (pv < 0.05) & (pv >= 0.01)

    pv = pv.astype(str)
    pv[ns] = "NS"
    pv[three] = "***"
    pv[two] = "**"
    pv[one] = "*"

    np.fill_diagonal(pv.values, "-")
    if not lower:
        pv.values[np.tril_indices(pv.shape[0], -1)] = ""
    elif not upper:
        pv.values[np.triu_indices(pv.shape[0], 1)] = ""

    return pv


# Sign plot for scikit-posthoc
def sign_plot(
    x: Union[List, np.ndarray, DataFrame],
    g: Union[List, np.ndarray] = None,
    flat: bool = False,
    labels: bool = True,
    cmap: List = None,
    cbar_ax_bbox: List = None,
    ax: SubplotBase = None,
    **kwargs,
) -> Union[SubplotBase, Tuple[SubplotBase, Colorbar]]:

    for key in ["cbar", "vmin", "vmax", "center"]:
        if key in kwargs:
            del kwargs[key]

    if isinstance(x, DataFrame):
        df = x.copy()
    else:
        x = np.array(x)
        g = g or np.arange(x.shape[0])
        df = DataFrame(np.copy(x), index=g, columns=g)

    dtype = df.values.dtype

    if not np.issubdtype(dtype, np.integer) and flat:
        raise ValueError("X should be a sign_array or DataFrame of integers")
    elif not np.issubdtype(dtype, np.floating) and not flat:
        raise ValueError("X should be an array or DataFrame of float p values")

    if not cmap and flat:
        # format: diagonal, non-significant, significant
        cmap = ["1", "#fbd7d4", "#1a9641"]
    elif not cmap and not flat:
        # format: diagonal, non-significant, p<0.001, p<0.01, p<0.05
        cmap = ["1", "#fbd7d4", "#005a32", "#238b45", "#a1d99b"]

    if flat:
        np.fill_diagonal(df.values, -1)
        hax = heatmap(
            df, vmin=-1, vmax=1, cmap=ListedColormap(cmap), cbar=False, ax=ax, **kwargs
        )
        if not labels:
            hax.set_xlabel("")
            hax.set_ylabel("")
        return hax

    else:
        df[(x < 0.001) & (x >= 0)] = 1
        df[(x < 0.01) & (x >= 0.001)] = 2
        df[(x < 0.05) & (x >= 0.01)] = 3
        df[(x >= 0.05)] = 0

        np.fill_diagonal(df.values, -1)

        if len(cmap) != 5:
            raise ValueError("Cmap list must contain 5 items")

        hax = heatmap(
            df,
            vmin=-1,
            vmax=3,
            cmap=ListedColormap(cmap),
            center=1,
            cbar=False,
            ax=ax,
            **kwargs,
        )
        if not labels:
            hax.set_xlabel("")
            hax.set_ylabel("")

        cbar_ax = hax.figure.add_axes(cbar_ax_bbox or [0.95, 0.35, 0.04, 0.3])
        cbar = ColorbarBase(
            cbar_ax,
            cmap=(ListedColormap(cmap[2:] + [cmap[1]])),
            norm=colors.NoNorm(),
            boundaries=[0, 1, 2, 3, 4],
        )
        cbar.set_ticks(
            list(np.linspace(0, 3, 4)),
            labels=["p < 0.001", "p < 0.01", "p < 0.05", "NS"],
        )

        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor("0.5")
        cbar.ax.tick_params(size=0)

        return hax, cbar


def _find_maximal_cliques(adj_matrix: DataFrame) -> List[Set]:

    if (adj_matrix.index != adj_matrix.columns).any():
        raise ValueError("adj_matrix must be symmetric, indices do not match")
    if not adj_matrix.isin((0, 1)).values.all():
        raise ValueError("Input matrix must be binary")
    if adj_matrix.empty or not (adj_matrix.T == adj_matrix).values.all():
        raise ValueError("Input matrix must be non-empty and symmetric")

    result = []
    _bron_kerbosch(
        current_clique=set(),
        candidates=set(adj_matrix.index),
        visited=set(),
        adj_matrix=adj_matrix,
        result=result,
    )
    return result


def _bron_kerbosch(
    current_clique: Set,
    candidates: Set,
    visited: Set,
    adj_matrix: DataFrame,
    result: List[Set],
) -> None:

    while candidates:
        v = candidates.pop()
        _bron_kerbosch(
            current_clique | {v},
            # Restrict candidate vertices to the neighbors of v
            {n for n in candidates if adj_matrix.loc[v, n]},
            # Restrict visited vertices to the neighbors of v
            {n for n in visited if adj_matrix.loc[v, n]},
            adj_matrix,
            result,
        )
        visited.add(v)

    # We do not need to report a clique if a children call aready did it.
    if not visited:
        # If this is not a terminal call, i.e. if any clique was reported.
        result.append(current_clique)


def critical_difference_diagram(
    ranks: Union[dict, Series],
    sig_matrix: DataFrame,
    *,
    ax: SubplotBase = None,
    label_fmt_left: str = "{label} ({rank:.2g})",
    label_fmt_right: str = "({rank:.2g}) {label}",
    label_props: dict = None,
    marker_props: dict = None,
    elbow_props: dict = None,
    crossbar_props: dict = None,
    color_palette: Union[Dict[str, str], List] = {},
    line_style: Union[Dict[str, str], List] = {},
    text_h_margin: float = 0.01,
) -> Dict[str, list]:

    ## check color_palette consistency
    if len(color_palette) == 0:
        pass
    elif isinstance(color_palette, Dict) and (
        (len(set(ranks.keys()) & set(color_palette.keys()))) == len(ranks)
    ):
        pass
    elif isinstance(color_palette, List) and (len(ranks) <= len(color_palette)):
        pass
    else:
        raise ValueError(
            "color_palette keys are not consistent, or list size too small"
        )

    elbow_props = elbow_props or {}
    marker_props = {"zorder": 3, **(marker_props or {})}
    label_props = {"va": "center", **(label_props or {})}
    crossbar_props = {
        "color": "k",
        "zorder": 3,
        "linewidth": 2,
        **(crossbar_props or {}),
    }

    ax = ax or pyplot.gca()
    ax.yaxis.set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.spines["top"].set_position("zero")

    # lists of artists to be returned
    markers = []
    elbows = []
    labels = []
    crossbars = []

    # True if pairwise comparison is NOT significant
    adj_matrix = DataFrame(
        1 - sign_array(sig_matrix),
        index=sig_matrix.index,
        columns=sig_matrix.columns,
        dtype=bool,
    )

    ranks = Series(ranks)  # Standardize if ranks is dict
    points_left, points_right = np.array_split(ranks.sort_values(), 2)

    # Sets of points under the same crossbar
    crossbar_sets = _find_maximal_cliques(adj_matrix)

    # Sort by lowest rank and filter single-valued sets
    crossbar_sets = sorted(
        (x for x in crossbar_sets if len(x) > 1), key=lambda x: ranks[list(x)].min()
    )

    # Create stacking of crossbars: for each level, try to fit the crossbar,
    # so that it does not intersect with any other in the level. If it does not
    # fit in any level, create a new level for it.
    crossbar_levels: list[list[set]] = []
    for bar in crossbar_sets:
        for level, bars_in_level in enumerate(crossbar_levels):
            if not any(bool(bar & bar_in_lvl) for bar_in_lvl in bars_in_level):
                ypos = -level - 1
                bars_in_level.append(bar)
                break
        else:
            ypos = -len(crossbar_levels) - 1
            crossbar_levels.append([bar])

        crossbars.append(
            ax.plot(
                # Adding a separate line between each pair enables showing a
                # marker over each elbow with crossbar_props={'marker': 'o'}.
                [ranks[i] for i in bar],
                [ypos] * len(bar),
                **crossbar_props,
            )
        )

    lowest_crossbar_ypos = -len(crossbar_levels)

    # def _change_label(label):
    #     label_ = label.split("-")
    #     label_ = [rf"$\bf{x}$" for x in label_]
    #     label_ = ("-").join(label_)
    #     return label_

    def _change_label(label):
        label_temp = label.split("-")
        label_ = []
        for x in label_temp:
            if len(x.split(" ")) != 1:
                temp = x.split(" ")
                temp = (" ").join([r"$\bf\{" + f"{x}" + r"}$" for x in temp])
                label_.append(temp)
            else:
                label_.append(r"$\bf\{" + f"{x}" + r"}$")
        label_ = ("-").join(label_)
        label_ = label_.replace("\\{", "{")
        return label_

    def plot_items(points, xpos, label_fmt, color_palette, line_style, label_props):
        """Plot each marker + elbow + label."""
        ypos = lowest_crossbar_ypos - 1
        for idx, (label, rank) in enumerate(points.items()):
            if len(color_palette) == 0:
                elbow, *_ = ax.plot(
                    [xpos, rank, rank],
                    [ypos, ypos, 0],
                    **elbow_props,
                )
                label_ = label
            else:
                elbow, *_ = ax.plot(
                    [xpos, rank, rank],
                    [ypos, ypos, 0],
                    c=(
                        color_palette[label]
                        if isinstance(color_palette, Dict)
                        else color_palette[idx]
                    ),
                    ls=(
                        line_style[label]
                        if isinstance(line_style, Dict)
                        else line_style[idx]
                    ),
                    **elbow_props,
                )
                if color_palette[label] != "black":  # darkgrey black
                    label_ = _change_label(label)
                else:
                    label_ = label
            elbows.append(elbow)
            curr_color = elbow.get_color()
            markers.append(ax.scatter(rank, 0, **{"color": curr_color, **marker_props}))
            labels.append(
                ax.text(
                    xpos,
                    ypos,
                    label_fmt.format(label=label_, rank=-1 * rank),
                    **{"color": curr_color, **label_props},
                )
            )
            ypos -= 1.5

    plot_items(
        points_left,
        xpos=points_left.iloc[0] - text_h_margin,
        label_fmt=label_fmt_left,
        color_palette=color_palette,
        line_style=line_style,
        label_props={
            "ha": "right",
            **label_props,
        },
    )
    plot_items(
        points_right[::-1],
        xpos=points_right.iloc[-1] + text_h_margin,
        label_fmt=label_fmt_right,
        color_palette=color_palette,
        line_style=line_style,
        label_props={"ha": "left", **label_props},
    )

    return {
        "markers": markers,
        "elbows": elbows,
        "labels": labels,
        "crossbars": crossbars,
    }
