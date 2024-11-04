"""Specific configurations for the CARTE paper."""

# Dataset names
carte_datalist = [
    "anime_planet",
    "babies_r_us",
    "beer_ratings",
    "bikedekho",
    "bikewale",
    "buy_buy_baby",
    "cardekho",
    "chocolate_bar_ratings",
    "clear_corpus",
    "coffee_ratings",
    "company_employees",
    "employee_remuneration",
    "employee_salaries",
    "fifa22_players",
    "filmtv_movies",
    "journal_jcr",
    "journal_sjr",
    "jp_anime",
    "k_drama",
    "michelin",
    "mlds_salaries",
    "movies",
    "museums",
    "mydramalist",
    "nba_draft",
    "prescription_drugs",
    "ramen_ratings",
    "roger_ebert",
    "rotten_tomatoes",
    "spotify",
    "us_accidents_counts",
    "us_accidents_severity",
    "us_presidential",
    "used_cars_24",
    "used_cars_benz_italy",
    "used_cars_dot_com",
    "used_cars_pakistan",
    "used_cars_saudi_arabia",
    "videogame_sales",
    "whisky",
    "wikiliq_beer",
    "wikiliq_spirit",
    "wina_pl",
    "wine_dot_com_prices",
    "wine_dot_com_ratings",
    "wine_enthusiasts_prices",
    "wine_enthusiasts_ratings",
    "wine_vivino_price",
    "wine_vivino_rating",
    "yelp",
    "zomato",
]

# Dictionary of baseline methods
carte_singletable_baselines = dict()
carte_singletable_baselines["full"] = [
    "carte-gnn",
    "catboost",
    "sentence-llm-concat-num_histgb",
    "sentence-llm-concat-num_xgb",
    "sentence-llm-embed-num_histgb",
    "sentence-llm-embed-num_xgb",
    "tablevectorizer-fasttext_histgb",
    "tablevectorizer-fasttext_xgb",
    "tablevectorizer-llm_histgb",
    "tablevectorizer-llm_xgb",
    "tablevectorizer_histgb",
    "tablevectorizer_logistic",
    "tablevectorizer_mlp",
    "tablevectorizer_randomforest",
    "tablevectorizer_resnet",
    "tablevectorizer_ridge",
    "tablevectorizer_xgb",
    "tablevectorizer_tabpfn",
    "target-encoder_histgb",
    "target-encoder_logistic",
    "target-encoder_mlp",
    "target-encoder_randomforest",
    "target-encoder_resnet",
    "target-encoder_ridge",
    "target-encoder_xgb",
    "target-encoder_tabpfn",
]

carte_singletable_baselines["reduced"] = [
    "carte-gnn",
    "catboost",
    "sentence-llm-concat-num_xgb",
    "sentence-llm-embed-num_xgb",
    "tablevectorizer_logistic",
    "tablevectorizer_mlp",
    "tablevectorizer_randomforest",
    "tablevectorizer_resnet",
    "tablevectorizer_ridge",
    "tablevectorizer_xgb",
    "target-encoder_tabpfn",
]

carte_multitable_baselines = [
    "original_carte-multitable",
    "matched_carte-multitable",
    "original_catboost-multitable",
    "matched_catboost-multitable",
    "original-sentence-llm_histgb-multitable",
    "matched-sentence-llm_histgb-multitable",
]


# Dictionary of method mapping
carte_singletable_baseline_mapping = dict()
carte_singletable_baseline_mapping["carte-gnn"] = "CARTE"

# Preprocessings
carte_singletable_baseline_mapping["tablevectorizer_"] = "TabVec-"
carte_singletable_baseline_mapping["tablevectorizer-"] = "TabVec-"
carte_singletable_baseline_mapping["target-encoder_"] = "TarEnc-"
carte_singletable_baseline_mapping["fasttext_"] = "FT-"
carte_singletable_baseline_mapping["llm_"] = "LLM-"
carte_singletable_baseline_mapping["sentence-llm-concat-num_"] = "S-LLM-CN-"
carte_singletable_baseline_mapping["sentence-llm-embed-num_"] = "S-LLM-EN-"

# Estimators
carte_singletable_baseline_mapping["catboost"] = "CatBoost"
carte_singletable_baseline_mapping["xgb"] = "XGB"
carte_singletable_baseline_mapping["histgb"] = "HGB"
carte_singletable_baseline_mapping["randomforest"] = "RF"
carte_singletable_baseline_mapping["ridge"] = "Ridge"
carte_singletable_baseline_mapping["logistic"] = "Logistic"
carte_singletable_baseline_mapping["mlp"] = "MLP"
carte_singletable_baseline_mapping["resnet"] = "ResNet"
carte_singletable_baseline_mapping["tabpfn"] = "TabPFN"

# Bagging
carte_singletable_baseline_mapping["bagging"] = "Bagging"

# Colors for visualization
carte_singletable_color_palette = dict()
carte_singletable_color_palette["CARTE"] = "C3"
carte_singletable_color_palette["CatBoost"] = "C0"
carte_singletable_color_palette["TabVec-XGB"] = "C1"
carte_singletable_color_palette["TabVec-RF"] = "C2"
carte_singletable_color_palette["TabVec-Ridge"] = "C4"
carte_singletable_color_palette["TabVec-Logistic"] = "C5"
carte_singletable_color_palette["S-LLM-CN-XGB"] = "C6"
carte_singletable_color_palette["S-LLM-EN-XGB"] = "C7"
carte_singletable_color_palette["TabVec-ResNet"] = "C8"
carte_singletable_color_palette["TabVec-MLP"] = "C9"
carte_singletable_color_palette["TarEnc-TabPFN"] = "#A9561E"

# Markers for visualization
carte_singletable_markers = dict()
carte_singletable_markers["CARTE"] = "o"
carte_singletable_markers["TabVec-XGB"] = (4, 0, 45)
carte_singletable_markers["TabVec-RF"] = "P"
carte_singletable_markers["CatBoost"] = "X"
carte_singletable_markers["S-LLM-CN-XGB"] = (4, 0, 0)
carte_singletable_markers["S-LLM-EN-XGB"] = "d"
carte_singletable_markers["TabVec-Ridge"] = "v"
carte_singletable_markers["TabVec-Logistic"] = "v"
carte_singletable_markers["TabVec-ResNet"] = "^"
carte_singletable_markers["TabVec-MLP"] = "p"
carte_singletable_markers["TarEnc-TabPFN"] = (5, 1, 0)
