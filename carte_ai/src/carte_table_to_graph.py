import torch
import numpy as np
import pandas as pd
import fasttext
import fasttext.util
import gc  # Import the garbage collector module
from typing import Union
from torch_geometric.data import Data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import (
    FeatureHasher,
)  # Import FeatureHasher from scikit-learn
from carte_ai.configs.directory import config_directory


def _create_edge_index(
    num_nodes: int,
    edge_attr: torch.Tensor,
    undirected: bool = True,
    self_loop: bool = True,
):
    """
    Sets the edge_index and edge_attr for graphs.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edge_attr : torch.Tensor
        Edge attributes tensor.
    undirected : bool, optional
        Whether the graph is undirected, by default True.
    self_loop : bool, optional
        Whether to add self-loops, by default True.

    Returns
    -------
    edge_index : torch.Tensor
        Edge indices tensor.
    edge_attr : torch.Tensor
        Edge attributes tensor.
    """
    edge_index_ = torch.triu_indices(num_nodes, num_nodes, offset=1)
    edge_index_ = edge_index_[:, (edge_index_[0] == 0)]
    edge_index = edge_index_.clone()
    edge_attr_ = edge_attr.clone()

    if undirected:
        edge_index = torch.cat((edge_index, torch.flip(edge_index, [0])))
        edge_attr_ = torch.cat((edge_attr_, edge_attr_))

    if self_loop:
        unique_nodes = edge_index_[1].unique()
        edge_index_self_loop = torch.stack((unique_nodes, unique_nodes))
        edge_index = torch.cat((edge_index, edge_index_self_loop), dim=1)
        edge_attr_ = torch.cat(
            (
                edge_attr_,
                torch.ones(
                    unique_nodes.size(0), edge_attr_.size(1), dtype=edge_attr_.dtype
                ),
            )
        )

    return edge_index, edge_attr_


class Table2GraphTransformer(TransformerMixin, BaseEstimator):
    """
    Transformer from tables to a list of graphs.

    Parameters
    ----------
    include_edge_attr : bool, optional
        Whether to include edge attributes, by default True.
    lm_model : str, optional
        Language model to use, by default "fasttext".
    n_components : int, optional
        Number of components for the encoder, by default 300.
    n_jobs : int, optional
        Number of jobs for parallel processing, by default 1.
    fasttext_model_path : str, optional
        Path to the FastText model file, required if lm_model is 'fasttext'.
    """

    def __init__(
        self,
        *,
        include_edge_attr: bool = True,
        lm_model: str = "fasttext",
        n_components: int = 300,
        n_jobs: int = 1,
        fasttext_model_path: str = None,
    ):
        super().__init__()
        self.include_edge_attr = include_edge_attr
        self.lm_model = lm_model
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.fasttext_model_path = fasttext_model_path
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit function used for the Table2GraphTransformer.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to fit.
        y : array-like, optional
            Target values, by default None.

        Returns
        -------
        self : Table2GraphTransformer
            Fitted transformer.
        """
        self.y_ = y

        if not hasattr(self, "lm_model_"):
            self._load_lm_model()

        cat_col_names = (
            X.select_dtypes(include="object")
            .columns.str.replace("\n", " ", regex=True)
            .str.lower()
        )
        self.cat_col_names = list(cat_col_names)
        num_col_names = (
            X.select_dtypes(exclude="object")
            .columns.str.replace("\n", " ", regex=True)
            .str.lower()
        )
        self.num_col_names = list(num_col_names)
        self.col_names = self.cat_col_names + self.num_col_names

        self.num_transformer_ = PowerTransformer().set_output(transform="pandas")


        # Ensure numerical columns exist before fitting the transformer
        if self.num_col_names:
            num_cols_exist = [col for col in self.num_col_names if col in X.columns]
            if num_cols_exist:
                self.num_transformer_.fit(X[num_cols_exist])

        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Apply Table2GraphTransformer to each row of the data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to transform.
        y : array-like, optional
            Target values, by default None.

        Returns
        -------
        data_graph : list
            List of transformed graph objects.
        """
        X_ = X.replace("\n", " ", regex=True)
        num_data = X_.shape[0]

        y_ = (
            torch.tensor(self.y_, dtype=torch.float32).reshape((num_data, 1))
            if self.y_ is not None
            else None
        )

        X_categorical = X_.select_dtypes(include="object").copy()
        X_categorical.columns = self.cat_col_names
        X_numerical = X_.select_dtypes(exclude="object").copy()
        X_numerical.columns = self.num_col_names

        cat_names = (
            pd.melt(X_categorical)["value"].dropna().astype(str).str.lower().unique()
        )
        names_total = np.unique(np.hstack([self.col_names, cat_names]))
        name_dict = {name: idx for idx, name in enumerate(names_total)}

        name_attr_total = self._transform_names(names_total)
        if self.num_col_names:
            num_cols_exist = [col for col in self.num_col_names if col in X.columns]
            if num_cols_exist:
                X_numerical = self._transform_numerical(X_numerical[num_cols_exist])

        data_graph = [
            self._graph_construct(
                X_categorical.iloc[idx],
                X_numerical.iloc[idx],
                name_attr_total,
                name_dict,
                y_,
                idx,
            )
            for idx in range(num_data)
        ]

        self.y_ = None

        # Manually trigger garbage collection after transforming data
        gc.collect()

        return data_graph

    def _load_lm_model(self):
        """
        Load the language model for features of nodes and edges.
        """
        if self.lm_model == "fasttext":
            if self.fasttext_model_path is None:
                raise ValueError(
                    "The 'fasttext_model_path' must be provided when using 'fasttext' as lm_model."
                )
            self.lm_model_ = fasttext.load_model(self.fasttext_model_path)
            if self.n_components != 300:
                fasttext.util.reduce_model(self.lm_model_, self.n_components)


    def _transform_numerical(self, X):
        """
        Transform numerical columns using power transformer.

        Parameters
        ----------
        X : pandas.DataFrame
            Input numerical data.

        Returns
        -------
        transformed_X : pandas.DataFrame
            Transformed numerical data.
        """
        return self.num_transformer_.transform(X)

    def _transform_names(self, names_total):
        """
        Obtain the feature for a given list of string values.

        Parameters
        ----------
        names_total : array-like
            List of string values.

        Returns
        -------
        name_features : np.ndarray
            Transformed features for names.
        """
        if self.lm_model == "fasttext":
            return np.array(
                [self.lm_model_.get_sentence_vector(name) for name in names_total],
                dtype=np.float32,
            )

    def _graph_construct(self, data_cat, data_num, name_attr_total, name_dict, y, idx):
        """
        Transform to graph objects.

        Parameters
        ----------
        data_cat : pandas.Series
            Categorical data for a single instance.
        data_num : pandas.Series
            Numerical data for a single instance.
        name_attr_total : np.ndarray
            Transformed features for names.
        name_dict : dict
            Dictionary mapping names to indices.
        y : torch.Tensor or None
            Target values.
        idx : int
            Index of the instance.

        Returns
        -------
        data : torch_geometric.data.Data
            Graph data object.
        """
        data_cat = data_cat.dropna().str.lower()
        data_num = data_num.dropna()
        num_cat = len(data_cat)
        num_num = len(data_num)

        edge_attr_cat = np.array(
            [name_attr_total[name_dict[col]] for col in data_cat.index],
            dtype=np.float32,
        )
        edge_attr_num = np.array(
            [name_attr_total[name_dict[col]] for col in data_num.index],
            dtype=np.float32,
        )

        x_cat = torch.tensor(
            np.array([name_attr_total[name_dict[val]] for val in data_cat]),
            dtype=torch.float32,
        )
        x_num = torch.tensor(
            data_num.values[:, None] * edge_attr_num, dtype=torch.float32
        )

        if x_cat.size(0) == 0:
            x_cat = torch.empty((0, self.n_components), dtype=torch.float32)
            edge_attr_cat = torch.empty((0, self.n_components), dtype=torch.float32)
        if x_num.size(0) == 0:
            x_num = torch.empty((0, self.n_components), dtype=torch.float32)
            edge_attr_num = torch.empty((0, self.n_components), dtype=torch.float32)

        x = torch.cat((x_cat, x_num))
        x = torch.cat((torch.ones((1, x.size(1))), x))
        edge_attr = torch.tensor(
            np.vstack((edge_attr_cat, edge_attr_num)), dtype=torch.float32
        )

        num_nodes = num_cat + num_num + 1
        edge_index, edge_attr = _create_edge_index(num_nodes, edge_attr, False, True)

        Z = torch.mul(edge_attr, x[edge_index[1]])
        x[0, :] = Z[edge_index[0] == 0].mean(dim=0)

        y_ = y[idx].clone() if y is not None else torch.tensor([])

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_,
            g_idx=idx,
        )
