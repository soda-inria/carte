""" Functions used for preprocessing the data. """

import numpy as np
import pandas as pd
from carte_ai.configs.directory import config_directory


def _clean_entity_names(data_entity_name):
    data_entity_name = (
        data_entity_name.str.replace("<", "")
        .str.replace(">", "")
        .str.replace("\n", "")
        .str.replace("_", " ")
        .str.lower()
    )
    return data_entity_name


def _serialize_instance(data):
    data_temp = data.copy()
    data_temp = data_temp.dropna()  # Exclude cells with Null values
    data_temp = _clean_entity_names(data_temp)
    serialization = np.array(data_temp.index) + " " + np.array(data_temp) + ". "
    sentence = ""
    for i in range(len(data_temp)):
        sentence += serialization[i]
    sentence = sentence[:-1]
    return sentence


def extract_fasttext_features(data: pd.DataFrame, extract_col_name: str):
    import fasttext

    # Preliminary Settings
    lm_model = fasttext.load_model(config_directory["fasttext"])

    # Original data
    data_ = data.copy()
    data_.replace("\n", " ", regex=True, inplace=True)
    data_ = data.copy()

    # Entity Names
    ent_names = _clean_entity_names(data[extract_col_name])
    ent_names = list(ent_names)

    # Data Fasttext for entity names
    data_fasttext = [lm_model.get_sentence_vector(str(x)) for x in ent_names]
    data_fasttext = np.array(data_fasttext)
    data_fasttext = pd.DataFrame(data_fasttext)
    col_names = [f"X{i}" for i in range(data_fasttext.shape[1])]
    data_fasttext = data_fasttext.set_axis(col_names, axis="columns")
    data_fasttext = pd.concat([data_fasttext, data[extract_col_name]], axis=1)
    # data_fasttext.drop_duplicates(inplace=True)
    data_fasttext = data_fasttext.reset_index(drop=True)

    return data_fasttext


def extract_llm_features(
    data: pd.DataFrame,
    extract_col_name: str,
    device: str = "cuda:0",
):
    # Load LLM Model
    from sentence_transformers import SentenceTransformer

    lm_model = SentenceTransformer("intfloat/e5-large-v2", device=device)

    # Original data
    data_ = data.copy()
    data_.replace("\n", " ", regex=True, inplace=True)

    # Entity Names
    ent_names = _clean_entity_names(data_[extract_col_name].copy())
    ent_names = ent_names.astype(str)
    ent_names = (
        "query: " + ent_names
    )  # following the outlined procedure using "query: "
    ent_names = list(ent_names)

    # Data for entity names
    embedding = lm_model.encode(ent_names, convert_to_numpy=True)
    embedding = pd.DataFrame(embedding)
    col_names = [f"X{i}" for i in range(embedding.shape[1])]
    embedding = embedding.set_axis(col_names, axis="columns")
    embedding = pd.concat([embedding, data[extract_col_name]], axis=1)
    # data_fasttext.drop_duplicates(inplace=True)
    embedding = embedding.reset_index(drop=True)

    return embedding


def extract_ken_features(
    data: pd.DataFrame,
    extract_col_name: str,
):
    # KEN embeddings
    ken_emb = pd.read_parquet(config_directory["ken_embed"])
    ken_ent = ken_emb["Entity"].str.lower()
    ken_embed_ent2idx = {ken_ent[i]: i for i in range(len(ken_emb))}

    # Original data
    data_ = data.copy()
    data_.replace("\n", " ", regex=True, inplace=True)
    data_ = data.copy()
    data_[extract_col_name] = data_[extract_col_name].str.lower()

    # Mapping
    mapping = data_[extract_col_name].map(ken_embed_ent2idx)
    mapping = mapping.dropna()
    mapping = mapping.astype(np.int64)
    mapping = np.array(mapping)

    # KEN data
    data_ken = ken_emb.iloc[mapping]
    data_ken.rename(columns={"Entity": "name"}, inplace=True)
    data_ken.drop_duplicates(inplace=True)
    data_ken = data_ken.reset_index(drop=True)

    return data_ken


def table2llmfeatures(
    data: pd.DataFrame,
    embed_numeric: bool,
    device: str = "cuda:0",
):
    # Load LLM Model
    from sentence_transformers import SentenceTransformer

    lm_model = SentenceTransformer("intfloat/e5-large-v2", device=device)

    # Preprocessing for the strings (subject to specifics of the data)
    data = data.replace("\n", " ", regex=True)
    num_data = len(data)
    data_x = data.copy()

    if embed_numeric:
        num_cols = data_x.select_dtypes(exclude="object").columns.tolist()
        data_x[num_cols] = data_x[num_cols].astype("str")

    data_x_cat = data_x.select_dtypes(include="object")
    data_x_num = data_x.select_dtypes(exclude="object")

    sentences = []
    for idx in range(num_data):
        data_ = data_x_cat.iloc[idx]
        sentence = _serialize_instance(data_)
        sentence = (
            "query: " + sentence
        )  # following the outlined procedure using "query: "
        sentences.append(sentence)

    X_categorical = lm_model.encode(sentences, convert_to_numpy=True)
    X_categorical = pd.DataFrame(X_categorical)

    col_names = [f"X{i}" for i in range(X_categorical.shape[1])]
    X_categorical = X_categorical.set_axis(col_names, axis="columns")

    data_total = pd.concat([X_categorical, data_x_num], axis=1)

    return data_total
