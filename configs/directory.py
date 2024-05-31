"""
Configurations for directory
"""

from pathlib import Path


# Need change the paths
base_path = Path().cwd()
config_directory = dict()
config_directory["base_path"] = base_path
config_directory["fasttext"] = str(base_path / "data/etc/cc.en.300.bin")
config_directory["pretrained_model"] = str(base_path / "data/etc/kg_pretrained_temp.pt")
# config_directory["pretrained_model"] = str(base_path / "data/etc/kg_pretrained.pt")
config_directory["data_ds"] = str(base_path / "data/data_ds/")
config_directory["data_ds_oov"] = str(base_path / "data/data_ds_oov/")
config_directory["data_ds_mt"] = str(base_path / "data/data_ds_mt/")
config_directory["data_ds_tabllm"] = str(base_path / "data/data_ds_tabllm/")
config_directory["data_kg"] = str(base_path / "data/data_kg/")
config_directory["data_raw"] = str(base_path / "data/raw/")
config_directory["results"] = str(base_path / "results/")
config_directory["visualization"] = str(base_path / "visualization/")


# Optional
config_directory["tmp"] = "/data/parietal/store3/work/mkim/tmp/"
config_directory["ken_embed"] = (
    "/data/parietal/store3/work/jstojano/gitlab/alexis_cvetkov/KEN/experiments/embedding_visualization/emb_mure_yago3_2022_full.parquet"
)
config_directory["infloat-v2"] = "/data/parietal/store3/work/mkim/intfloat_e5-large-v2/"
