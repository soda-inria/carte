"""
Configurations for directory
"""

from pathlib import Path

base_path = Path().cwd()
config_directory = dict()
config_directory["base_path"] = base_path

config_directory["checkpoints"] = str(base_path / "data/ckpt/")
config_directory["pretrained_model"] = str(base_path / "data/etc/kg_pretrained.pt")
config_directory["data_raw"] = str(base_path / "data/data_raw/")
config_directory["data_singletable"] = str(base_path / "data/data_singletable/")
config_directory["data_ds_oov"] = str(base_path / "data/data_ds_oov/")
config_directory["data_multitable"] = str(base_path / "data/data_multitable/")
config_directory["data_ds_tabllm"] = str(base_path / "data/data_ds_tabllm/")
config_directory["data_yago"] = str(base_path / "data/data_yago/")

config_directory["results"] = str(base_path / "results/")
config_directory["visualization"] = str(base_path / "visualization/")

# Specify the directory in which you have downloaded each
config_directory["fasttext"] = str(base_path / "data/etc/cc.en.300.bin")
config_directory["infloat-v2"] = str(base_path / "data/etc/intfloat_e5-large-v2")
# config_directory[ken_embedding] = 


