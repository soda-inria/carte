"""The Table2GraphTransformer Class"""

import torch
import numpy as np
import pandas as pd
import fasttext
from typing import Union
from torch_geometric.data import Data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline
from configs.directory import config_directory
from dirty_cat import MinHashEncoder  # change to skrubs


def _create_edge_index(
    num_nodes: int,
    edge_attr: torch.tensor,
    undirected: bool = True,
    self_loop: bool = True,
):
    # the list of possible edge_index (directed with the numbering)
    edge_index_ = torch.combinations(torch.arange(num_nodes), 2).transpose(0, 1)
    edge_index_ = edge_index_[:, (edge_index_[0] == 0)]
    edge_index = edge_index_.clone()
    edge_attr_ = edge_attr.clone()
    # undirected
    if undirected:
        edge_index = torch.hstack((edge_index, torch.flipud(edge_index)))
        edge_attr_ = torch.vstack((edge_attr_, edge_attr_))
    # self-loop
    if self_loop:
        edge_index_self_loop = torch.vstack(
            (edge_index_[1].unique(), edge_index_[1].unique())
        )
        edge_index = torch.hstack((edge_index, edge_index_self_loop))
        edge_attr_ = torch.vstack(
            (edge_attr_, torch.ones(num_nodes - 1, edge_attr_.size(1)))
        )
    return edge_index, edge_attr_


class Table2GraphTransformer(TransformerMixin, BaseEstimator):
    """Transformer from tables to a list of graphs.

    The list of graphs are generated in a row-wise fashion.
    """

    def __init__(
        self,
        *,
        include_edge_attr: bool = True,
        lm_model: str = "fasttext",
        n_components: float = 300,
        n_jobs: int = 1,
    ):
        super(Table2GraphTransformer, self).__init__()

        self.include_edge_attr = include_edge_attr
        self.lm_model = lm_model
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit function used for the Table2GraphTransformer

        Parameters
        ----------
        X : pandas DataFrame (n_samples, n_features)
            The input data used to transform to graphs.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """

        self.y_ = y

        self.is_fitted_ = False

        # Load language_model
        if hasattr(self, "lm_model_") == False:
            self._load_lm_model()

        # Relations
        cat_col_names = X.select_dtypes(include="object").columns
        cat_col_names = cat_col_names.str.replace("\n", " ", regex=True).str.lower()
        self.cat_col_names = list(cat_col_names)
        num_col_names = X.select_dtypes(exclude="object").columns
        num_col_names = num_col_names.str.replace("\n", " ", regex=True).str.lower()
        self.num_col_names = list(num_col_names)
        self.col_names = self.cat_col_names + self.num_col_names

        # Numerical transformer - Powertransformer
        self.num_transformer_ = PowerTransformer().set_output(transform="pandas")
        if self.lm_model == "minhash":
            self.name_transformer = make_pipeline(
                MinHashEncoder(n_components=self.n_components, n_jobs=self.n_jobs),
                PowerTransformer(),
            )

        return self

    def transform(self, X, y=None):
        """Apply Table2GraphTransformer to each row of the data

        Parameters
        ----------
        X : Pandas DataFrame. (n_samples, n_features)
            The input data used to transform to graphs.

        y : None
            Ignored.

        Returns
        -------
        Graph Data : list of size (n_samples).
            The list of transformed graph data.
        """

        # Preprocess the features
        X_ = X.copy()
        X_ = X_.replace("\n", " ", regex=True)
        num_data = X_.shape[0]

        # Preprocess the target
        y_ = None
        if self.y_ is not None:
            y_ = np.array(self.y_)
            y_ = torch.tensor(y_).reshape((num_data, 1))

        # Separate categorical and numerical columns
        X_categorical = X_.select_dtypes(include="object").copy()
        X_categorical.columns = self.cat_col_names
        X_numerical = X_.select_dtypes(exclude="object").copy()
        X_numerical.columns = self.num_col_names

        # Features for names
        cat_names = pd.melt(X_categorical)["value"]
        cat_names = cat_names.dropna()
        cat_names = cat_names.astype(str)
        cat_names = cat_names.str.replace("\n", " ", regex=True).str.lower()
        cat_names = cat_names.unique()
        names_total = np.hstack([self.col_names, cat_names])
        names_total = np.unique(names_total)
        name_dict = {names_total[i]: i for i in range(names_total.shape[0])}

        # preprocess values
        name_attr_total = self._transform_names(names_total)
        if len(self.num_col_names) != 0:
            X_numerical = self._transform_numerical(X_numerical)
        if self.is_fitted_ == False:
            self.is_fitted_ = True

        data_graph = [
            self._graph_construct(
                X_categorical,
                X_numerical,
                name_attr_total,
                name_dict,
                y_,
                idx=i,
            )
            for i in range(num_data)
        ]

        if self.y_ is not None:
            self.y_ = None

        return data_graph

    def _load_lm_model(self):
        """Load the language model for features of nodes and edges."""

        if self.lm_model == "fasttext":
            # Loading fasttext
            self.lm_model_ = fasttext.load_model(config_directory["fasttext"])
            if self.n_components != 300:
                fasttext.util.reduce_model(self.lm_model_, self.n_components)
        elif self.lm_model == "minhash":
            self.lm_model_ = MinHashEncoder(
                n_components=self.n_components,
                n_jobs=self.n_jobs,
            )

    def _transform_numerical(self, X):
        X_num = X.copy()
        if self.is_fitted_ == False:
            X_num = self.num_transformer_.fit_transform(X_num)
        else:
            X_num = self.num_transformer_.transform(X_num)
        return X_num

    def _transform_names(self, names_total):
        if self.lm_model == "fasttext":
            name_attr_total = [
                self.lm_model_.get_sentence_vector(i) for i in names_total
            ]
            name_attr_total = np.array(name_attr_total).astype(np.float32)
            pass
        elif self.lm_model == "minhash":
            name_attr_total = self.name_transformer.fit_transform(
                names_total.reshape(-1, 1)
            )
            name_attr_total = name_attr_total.astype(np.float32)
        return name_attr_total

    def _graph_construct(
        self,
        X_categorical,
        X_numerical,
        name_attr_total,
        name_dict,
        y,
        idx,
    ):

        # Obtain the data for a 'idx'-th row
        data_cat = X_categorical.iloc[idx]
        data_cat = data_cat.dropna()
        num_cat = len(data_cat)
        if num_cat != 0:
            data_cat = data_cat.str.replace("\n", " ", regex=True).str.lower()
        data_num = X_numerical.iloc[idx]
        data_num = data_num.dropna()
        num_num = len(data_num)

        # edge_attributes
        if self.include_edge_attr:
            edge_attr_cat = [name_attr_total[name_dict[x]] for x in data_cat.index]
            edge_attr_cat = np.array(edge_attr_cat).astype(np.float32)
            edge_attr_num = [name_attr_total[name_dict[x]] for x in data_num.index]
            edge_attr_num = np.array(edge_attr_num).astype(np.float32)
        else:
            edge_attr_cat = np.ones((num_cat, self.n_components)).astype(np.float32)
            edge_attr_num = np.ones((num_num, self.n_components)).astype(np.float32)

        # node_attributes
        x_cat = [name_attr_total[name_dict[x]] for x in data_cat]
        x_cat = np.array(x_cat).astype(np.float32)
        x_cat = torch.tensor(x_cat)
        if x_cat.size(0) == 0:
            x_cat = x_cat.reshape(0, self.n_components)
            edge_attr_cat = edge_attr_cat.reshape(0, self.n_components)

        x_num_ = np.array(data_num).astype("float32")
        x_num = x_num_.reshape(-1, 1) * edge_attr_num
        x_num = torch.tensor(x_num)
        if x_num.size(0) == 0:
            x_num = x_num.reshape(0, self.n_components)
            edge_attr_num = edge_attr_num.reshape(0, self.n_components)

        # combined node/edge attributes
        x = torch.vstack((x_cat, x_num))
        x = torch.vstack((torch.ones((1, x.size(1))), x))
        edge_attr = np.vstack((edge_attr_cat, edge_attr_num))
        edge_attr = torch.tensor(edge_attr)

        # edge_index
        num_nodes = num_cat + num_num + 1
        edge_index, edge_attr = _create_edge_index(num_nodes, edge_attr, False, True)

        # Set the center node
        Z = torch.mul(edge_attr, x[edge_index[1]])
        x[0, :] = Z[(edge_index[0] == 0), :].mean(dim=0)

        # Target
        if y is not None:
            y_ = y[idx].clone()
        else:
            y_ = torch.tensor([])

        # graph index (g_idx)
        g_idx = idx

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_,
            g_idx=g_idx,
        )

        return data
