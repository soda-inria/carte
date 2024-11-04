from pathlib import Path

# Get the base path relative to this file's location
base_path = (
    Path(__file__).resolve().parent.parent
)  # This gives '/home/infres/gbrison/carte/carte_ai'

config_directory = dict()
config_directory["base_path"] = base_path

config_directory["data"] = str(base_path / "data/")
config_directory["pretrained_model"] = str(
    base_path / "data/etc/kg_pretrained.pt"
)  # Correct path
config_directory["data_raw"] = str(base_path / "data/data_raw/")
config_directory["data_singletable"] = str(base_path / "data/data_singletable/")
config_directory["data_yago"] = str(base_path / "data/data_yago/")
config_directory["etc"] = str(base_path / "data/etc/")

config_directory["results"] = str(base_path / "results/")
config_directory["compiled_results"] = str(base_path / "results/compiled_results/")
config_directory["visualization"] = str(base_path / "visualization/")

# Specify the directory in which you have downloaded each
config_directory["fasttext"] = str(base_path / "data/etc/cc.en.300.bin")
config_directory["ken_embedding"] = str(base_path / "data/etc/ken_embedding.parquet")
