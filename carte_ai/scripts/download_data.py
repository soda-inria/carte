"""Script for downloading required data."""

# >>>
if __name__ == "__main__":
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ["PROJECT_DIR"] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import shutil
import os
import requests
from zipfile import ZipFile
from carte_ai.configs.directory import config_directory


def _download_with_request(url, download_path):
    req = requests.get(url, stream=True)
    with open(download_path, "wb") as f:
        for chunk in req.iter_content(chunk_size=8192):
            f.write(chunk)


def _download_fasttext():
    import fasttext.util

    fasttext.util.download_model("en", if_exists="ignore")
    ft_path = str(config_directory["base_path"] / "cc.en.300.bin")
    shutil.move(ft_path, config_directory["fasttext"])
    os.remove(str(config_directory["base_path"] / "cc.en.300.bin.gz"))


def _download_ken():
    ken_url = "https://figshare.com/ndownloader/files/39142985"
    ken_path = config_directory["ken_embedding"]
    _download_with_request(ken_url, ken_path)


def _download_raw(option="carte"):
    url = "https://huggingface.co/datasets/inria-soda/carte-benchmark/resolve/main/data_raw.zip"
    download_path = str(config_directory["base_path"] / "data_raw.zip")
    _download_with_request(url, download_path)
    if option == "carte":
        carte_example_data = [
            "wina_pl",
            "spotify",
            "wine_dot_com_prices",
            "wine_vivino_price",
        ]
        with ZipFile(download_path, "r") as zObject:
            for name in carte_example_data:
                raw_data_path = f"data_raw/{name}.csv"
                zObject.extract(raw_data_path, path=f"{config_directory['data']}")
    elif option == "full":
        with ZipFile(download_path, "r") as zObject:
            zObject.extractall(path=config_directory["data"])
        zObject.close()
    os.remove(download_path)


def _download_preprocessed(option="carte", include_llm=False):
    if include_llm:
        url = "https://huggingface.co/datasets/inria-soda/carte-benchmark/resolve/main/data_singletable.zip"
    else:
        url = "https://huggingface.co/datasets/inria-soda/carte-benchmark/resolve/main/data_singletable_light.zip"
    download_path = str(config_directory["base_path"] / "data_singletable.zip")
    _download_with_request(url, download_path)
    if option == "carte":
        carte_example_data = [
            "wina_pl",
            "spotify",
            "wine_dot_com_prices",
            "wine_vivino_price",
        ]
        with ZipFile(download_path, "r") as zObject:
            for name in carte_example_data:
                raw_data_path = f"data_singletable/{name}/raw.parquet"
                config_path = f"data_singletable/{name}/config_data.json"
                zObject.extract(raw_data_path, path=f"{config_directory['data']}")
                zObject.extract(config_path, path=f"{config_directory['data']}")
                if include_llm:
                    external_path = f"data_singletable/{name}/external.pickle"
                    zObject.extract(external_path, path=f"{config_directory['data']}")
    elif option == "full":
        with ZipFile(download_path, "r") as zObject:
            zObject.extractall(path=config_directory["data"])
        zObject.close()
    os.remove(download_path)


# Main
def main(option="carte", include_raw=False, include_ken=False):

    if os.path.exists(config_directory["fasttext"]):
        pass
    else:
        _download_fasttext()

    if option == "carte":
        option_ = "full"
    else:
        if option == "basic_examples":
            option_, include_llm = "carte", False
        elif option == "full_examples":
            option_, include_llm = "full", False
        elif option == "full_benchmark":
            option_, include_llm = "full", True
        _download_preprocessed(option_, include_llm)

    if include_raw:
        _download_raw(option=option_)

    if include_ken:
        _download_ken()

    return None


if __name__ == "__main__":

    # Set parser
    import argparse

    parser = argparse.ArgumentParser(description="Download data.")
    parser.add_argument(
        "-op",
        "--option",
        type=str,
        help="option for downloading",
    )
    parser.add_argument(
        "-ir",
        "--include_raw",
        type=str,
        help="include raw data for downloading",
    )
    parser.add_argument(
        "-ik",
        "--include_ken",
        type=str,
        help="include ken data for downloading",
    )
    args = parser.parse_args()

    if args.include_raw == "True":
        include_raw = True
    else:
        include_raw = False

    if args.include_ken == "True":
        include_ken = True
    else:
        include_ken = False

    main(args.option, include_raw, include_ken)
