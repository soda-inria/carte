"""Script for preprocessing raw data."""

# >>>
if __name__ == "__main__":
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ["PROJECT_DIR"] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import os
import json
import ast
import pandas as pd
import numpy as np
from carte_ai.configs.directory import config_directory
from carte_ai.configs.carte_configs import carte_datalist


def _drop_high_null(data, proportion=0.5):
    """Drop columns with high fraction of missing values"""
    null_num = np.array([data[col].isnull().sum() for col in data.columns])
    null_crit = int(len(data) * proportion)
    null_col = list(data.columns[null_num > null_crit])
    return data.drop(columns=null_col)


def _drop_single_unique(data):
    """Drop columns with single unique values."""
    num_unique_cols = [col for col in data.columns if data[col].nunique() == 1]
    return data.drop(columns=num_unique_cols)


def _load_raw_data(data_name, file_type="csv", sep=","):
    """Load the raw data for preprocessing."""
    data_dir = f"{config_directory['data_raw']}/{data_name}.{file_type}"
    if file_type == "csv":
        data = pd.read_csv(data_dir, sep=sep)
    elif file_type == "json":
        data_file = open(data_dir)
        data = []
        for line in data_file:
            data.append(json.loads(line))
        data = pd.DataFrame(data)
        data_file.close()
    data.columns = data.columns.str.replace(" ", "_")
    data.columns = data.columns.str.replace("\n", "_")
    data.columns = data.columns.str.replace("%", "Percentage")
    data.replace("\n", " ", regex=True, inplace=True)
    return data


def _save_processed_data(data_name, data, target_name, entity_name, task, repeated):
    """Save the preprocessed data and configs."""
    # save the data
    save_dir = f"{config_directory['data_singletable']}/{data_name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    data.to_parquet(save_dir + "raw.parquet")
    # save the config file
    config = dict()
    config["entity_name"] = entity_name
    config["target_name"] = target_name
    config["task"] = task
    config["repeated"] = repeated
    with open(save_dir + "config_data.json", "w") as outfile:
        json.dump(config, outfile)
    return None


def preprocess_data(data_name):
    """Preprocess the raw data with the given name of the dataset."""

    # Load data
    data = _load_raw_data(data_name)

    # Preoprocess depending on each data
    if data_name == "anime_planet":
        # basic info
        target_name = "Rating_Score"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        data.replace("Unknown", np.nan, inplace=True)
        target_name = "Rating_Score"
        data.dropna(subset=target_name, inplace=True)
        data[target_name] = data[target_name].astype("float")
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("Anime-PlanetID")
        drop_col.append("Number_Votes")
        drop_col.append("Url")
        data.drop(columns=drop_col, inplace=True)
        data["Finished"] = data["Finished"].astype("str")
        data["Episodes"] = data["Episodes"].astype("float")
        data["Duration"] = data["Duration"].astype("float")
    elif data_name == "babies_r_us":
        # basic info
        target_name = "price"
        entity_name = "title"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("int_id")
        drop_col.append("ext_id")
        drop_col.append("SKU")
        data.drop(columns=drop_col, inplace=True)
        temp = data["is_discounted"].copy()
        temp = temp.astype("str")
        data["is_discounted"] = temp
    elif data_name == "beer_ratings":
        # basic info
        target_name = "review_overall"
        entity_name = "Beer_Name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = [col for col in data.columns if "review" in col]
        drop_col.remove(target_name)
        data.drop(columns=drop_col, inplace=True)
        numeric_cols = data.select_dtypes(exclude="object").columns.to_list()
        data[numeric_cols] = data[numeric_cols].astype("float")
        data.rename(columns={"Beer_Name_(Full)": "Beer_Name"}, inplace=True)
    elif data_name == "bikedekho":
        # basic info
        target_name = "price"
        entity_name = "bike_name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(10, data[target_name])
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("id")
        data.drop(columns=drop_col, inplace=True)
        data["model_year"] = data["model_year"].astype("str")
        data["km_driven"] = data["km_driven"].astype("float")
    elif data_name == "bikewale":
        # basic info
        target_name = "price"
        entity_name = "bike_name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        mask = data[target_name] >= 500
        data = data[mask]
        data[target_name] = np.emath.logn(10, data[target_name])
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("id")
        data.drop(columns=drop_col, inplace=True)
        data["model_year"] = data["model_year"].astype("str")
        data["km_driven"] = data["km_driven"].astype("float")
    elif data_name == "buy_buy_baby":
        # basic info
        target_name = "price"
        entity_name = "title"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name] + 1)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("int_id")
        drop_col.append("ext_id")
        drop_col.append("SKU")
        drop_col.append("company_free")
        data.drop(columns=drop_col, inplace=True)
        temp = data["is_discounted"].copy()
        temp = temp.astype("str")
        temp[temp == "True"] = "1"
        temp[temp == "False"] = "0"
        data["is_discounted"] = temp
    elif data_name == "cardekho":
        # basic info
        target_name = "price"
        entity_name = "model"
        task = "regression"
        repeated = False
        # preprocess
        data.rename(columns={"km": "mileage"}, inplace=True)
        data["model_year"] = data["model_year"].astype(str)
        target_name = "price"
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(100, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "chocolate_bar_ratings":
        # basic info
        target_name = "Rating"
        entity_name = "Specific_Bean_Origin_or_Bar_Name"
        task = "classification"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        temp = data["Rating"].copy()
        temp[temp < 3.25] = 0
        temp[temp != 0] = 1
        data["Rating"] = temp
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data.drop(columns="REF", inplace=True)
        data.columns = data.columns.str.replace(" ", "_")
        data["Review_Date"] = data["Review_Date"].astype("str")
        data["Cocoa_Percent"] = data["Cocoa_Percent"].str.replace("%", "")
        data["Cocoa_Percent"] = data["Cocoa_Percent"].astype("float")
    elif data_name == "clear_corpus":
        # basic info
        target_name = "BT_Easiness"
        entity_name = "Title"
        task = "regression"
        repeated = False
        # preprocess
        data = data.replace("?", np.nan)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("ID")
        drop_col.append("BT_s.e.")
        drop_col.append("MPAA_#Avg")
        drop_col.append("MPAA__#Max")
        data.drop(columns=drop_col, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data["Pub_Year"] = data["Pub_Year"].astype("str")
        data["Pub_Year"] = data["Pub_Year"].str.split(".").str[0]
        numeric_cols = data.select_dtypes(exclude="object").columns.to_list()
        data[numeric_cols] = data[numeric_cols].astype("float")
    elif data_name == "coffee_ratings":
        # basic info
        target_name = "rating"
        entity_name = "name"
        task = "classification"
        repeated = False
        # preprocess
        temp = data[target_name].copy()
        temp[temp <= 93] = 0
        temp[temp != 0] = 1
        data[target_name] = temp
        data.reset_index(drop=True, inplace=True)
        data[target_name] = data[target_name].astype("float")
        data.dropna(subset=target_name, inplace=True)
        data.drop_duplicates(subset=["name"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("slug")
        drop_col.append("all_text")
        drop_col.append("review_date")
        drop_col.append("est_price")
        drop_col.append("aroma")
        drop_col.append("acid")
        drop_col.append("body")
        drop_col.append("flavor")
        drop_col.append("aftertaste")
        drop_col.append("agtron")
        data.drop(columns=drop_col, inplace=True)
    elif data_name == "company_employees":
        # basic info
        target_name = "current_employee_estimate"
        entity_name = "name"
        task = "regression"
        repeated = False
        # preprocess
        data.drop(columns=["Unnamed:_0"], inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.drop_duplicates(subset="name", keep=False, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = data[target_name].astype("float")
        data[target_name] = np.emath.logn(10, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_cols = []
        drop_cols.append("country")
        drop_cols.append("total_employee_estimate")
        data.drop(columns=drop_cols, inplace=True)
        data["year_founded"] = data["year_founded"].astype("str")
        data["year_founded"] = data["year_founded"].str.split(".").str[0]
        temp = data["year_founded"].copy()
        temp[temp == "nan"] = np.nan
        data["year_founded"] = temp
        num_cols = data.select_dtypes(exclude="object").columns
        data[num_cols] = data[num_cols].astype("float")
    elif data_name == "employee_remuneration":
        # Exception with different sep
        data = _load_raw_data(data_name, sep=";")
        # basic info
        target_name = "Remuneration"
        entity_name = "Title"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(10, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data.drop(columns=["Name"], inplace=True)
        data["Year"] = data["Year"].astype("str")
    elif data_name == "employee_salaries":
        # basic info
        target_name = "current_annual_salary"
        entity_name = "employee_position_title"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(10, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data["year_first_hired"] = data["year_first_hired"].astype("str")
    elif data_name == "fifa22_players":
        # basic info
        target_name = "wage_eur"
        entity_name = "name"
        task = "regression"
        repeated = False
        # preprocess
        drop_col_url = [col for col in data.columns if "_url" in col]
        drop_col_id = [col for col in data.columns if "_id" in col]
        data.drop(columns=drop_col_url + drop_col_id, inplace=True)
        data = data[data.columns[:-68]]
        data.rename(columns={"short_name": "name"}, inplace=True)
        data.drop_duplicates(subset=["name"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(10, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("long_name")
        drop_col.append("overall")
        drop_col.append("potential")
        drop_col.append("league_name")
        drop_col.append("league_level")
        drop_col.append("weak_foot")
        drop_col.append("skill_moves")
        drop_col.append("real_face")
        data.drop(columns=drop_col, inplace=True)
        data["club_jersey_number"] = data["club_jersey_number"].astype("str")
        data["club_jersey_number"] = data["club_jersey_number"].str.split(".").str[0]
        data["club_contract_valid_until"] = data["club_contract_valid_until"].astype(
            "str"
        )
        data["club_contract_valid_until"] = (
            data["club_contract_valid_until"].str.split(".").str[0]
        )
        num_cols = data.select_dtypes(exclude="object").columns
        data[num_cols] = data[num_cols].astype("float")
    elif data_name == "filmtv_movies":
        # basic info
        target_name = "public_vote"
        entity_name = "title"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data["year"] = data["year"].astype(str)
        data["duration"] = data["duration"].astype(float)
        drop_col = []
        drop_col.append("filmtv_id")
        drop_col.append("avg_vote")
        drop_col.append("total_votes")
        drop_col.append("humor")
        drop_col.append("rhythm")
        drop_col.append("effort")
        drop_col.append("tension")
        drop_col.append("erotism")
        data.drop(columns=drop_col, inplace=True)
    elif data_name == "journal_jcr":
        # basic info
        target_name = "2021_JIF"
        entity_name = "Journal_name"
        task = "regression"
        repeated = False
        # preprocess
        data.replace("N/A", np.nan, regex=True, inplace=True)
        num_cols = data.columns[4:8]
        num_cols = num_cols.append(data.columns[10:])
        for col in num_cols:
            data[col] = data[col].str.replace(",", "")
            data[col] = data[col].astype("float")
            data.dropna(subset=target_name, inplace=True)
            data.reset_index(drop=True, inplace=True)
            data[target_name] = np.log(data[target_name] + 1)
            data = _drop_high_null(data)
            data = _drop_single_unique(data)
            remove_cols = []
            remove_cols.append("Total_Citations")
            remove_cols.append("2021_JCI")
            remove_cols.append("JIF_Without_Self_Cites")
            remove_cols.append("5_Year_JIF")
            remove_cols.append("Immediacy_Index")
            remove_cols.append("Normalized_Eigenfactor")
            remove_cols.append("Eigenfactor")
            remove_cols.append("Article_Influence_Score")
            remove_cols.append("Total_Articles")
            data.drop(columns=remove_cols, inplace=True)
    elif data_name == "journal_sjr":
        # Exception with different sep
        data = _load_raw_data(data_name, sep=";")
        # basic info
        target_name = "H_index"
        entity_name = "Title"
        task = "regression"
        repeated = False
        # preprocess
        col_keep = list(data.columns[[2, 3, 4, 7]]) + list(data.columns)[-6:]
        data = data[col_keep]
        data.columns = data.columns.str.replace(" ", "_")
        temp1 = data["Issn"].str.split(",").str[0]
        temp1 = temp1.rename("ISSN")
        data["Issn"] = temp1
        temp2 = data["Issn"].str.split(",").str[1]
        temp2 = temp2.rename("e-ISSN")
        data["e-ISSN"] = temp2
        data.drop_duplicates(subset="Title", inplace=True)
        target_name = "H_index"
        data.dropna(subset=target_name, inplace=True)
        data[target_name] = np.log10(data[target_name] + 1)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "jp_anime":
        # basic info
        target_name = "Score"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        data.replace("UNKNOWN", np.nan, inplace=True)
        data.replace("Unknown", np.nan, inplace=True)
        mask = data["English_name"].isnull()
        temp = data["English_name"].copy()
        temp[mask] = data["Name"][mask]
        data["English_name"] = temp
        data.reset_index(drop=True, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data[target_name] = data[target_name].astype("float")
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        temp = data["Aired"].copy()
        data["Start_Date"] = temp.str.split(" to ").str[0]
        data["End_Date"] = temp.str.split(" to ").str[1]
        keep_col = list(data.columns)
        keep_col.remove("anime_id")
        keep_col.remove("Name")
        keep_col.remove("Other_name")
        keep_col.remove("Scored_By")
        keep_col.remove("Image_URL")
        keep_col.remove("Rank")
        keep_col.remove("Aired")
        data = data[keep_col]
        data.rename(columns={"English_name": "Name"}, inplace=True)
        num_cols = data.select_dtypes(exclude="object").columns
        data[num_cols] = data[num_cols].astype("float")
        data["Rating"] = data["Rating"].str.split(" - ").str[0]
        data.drop_duplicates(subset="Name", inplace=True)
        data.reset_index(drop=True, inplace=True)
        data["Episodes"] = data["Episodes"].astype(float)
        temp = data["Duration"].copy()
        temp = temp.astype(str)
        temp = temp.str.replace(" per ep", "", regex=False)
        temp1 = temp.str.split(" hr").str[0]
        temp1[~temp1.str.isnumeric()] = "0"
        temp1 = temp1.astype(float) * 60
        temp2 = temp.str.split(" hr").str[1]
        temp2 = temp2.astype(str)
        temp2 = temp2.str.replace(" min", "", regex=False)
        temp2 = temp2.str.replace(" ", "", regex=False)
        temp2[~temp2.str.isnumeric()] = "0"
        temp2 = temp2.astype(float)
        temp3 = temp.copy()
        temp3[temp.str.contains("hr")] = "nan"
        temp3 = temp3.str.replace(" min", "", regex=False)
        temp3 = temp3.str.replace(" ", "", regex=False)
        temp3[~temp3.str.isnumeric()] = "0"
        temp3 = temp3.astype(float)
        temp = temp1 + temp2 + temp3
        temp[temp == 0] = np.nan
        data["Duration"] = temp
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "k_drama":
        # basic info
        target_name = "score"
        entity_name = "Kdrama_name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data[target_name] = data[target_name].astype("float")
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_cols = []
        drop_cols.append("scored_by")
        drop_cols.append("Ranked")
        data.drop(columns=drop_cols, inplace=True)
        data["Content_Rating"] = data["Content_Rating"].str.split(" - ").str[0]
    elif data_name == "michelin":
        # basic info
        target_name = "Award"
        entity_name = "Kdrama_name"
        task = "classification"
        repeated = False
        # preprocess
        temp = data["Award"].copy()
        temp[temp.str.contains("MICHELIN")] = "1"
        temp[temp.str.contains("Bib Gourmand")] = "0"
        temp = temp.astype("float")
        data["Award"] = temp
        data.rename(columns={"WebsiteUrl": "Website_Url"}, inplace=True)
        data.rename(
            columns={"FacilitiesAndServices": "Facilities_And_Services"}, inplace=True
        )
        data.rename(columns={"PhoneNumber": "Phone_Number"}, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data["Facilities_And_Services"] = data["Facilities_And_Services"].str.replace(
            ",", ", "
        )
        drop_col = []
        drop_col.append("Phone_Number")
        drop_col.append("Url")
        drop_col.append("Price")
        drop_col.append("Facilities_And_Services")
        data.drop(columns=drop_col, inplace=True)
    elif data_name == "mlds_salaries":
        # basic info
        target_name = "salary_in_usd"
        entity_name = "job_title"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data[target_name] = data[target_name].astype("float")
        data[target_name] = np.log10(data[target_name])
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data["work_year"] = data["work_year"].astype("str")
        data["remote_ratio"] = data["remote_ratio"].astype("str")
        mapping = dict()
        mapping["experience_level"] = dict()
        mapping["experience_level"]["SE"] = "Senior-level / Expert"
        mapping["experience_level"]["EN"] = "Entry-level / Junior"
        mapping["experience_level"]["MI"] = "Mid-level / Intermediate"
        mapping["experience_level"]["EX"] = "Executive-level / Director"
        mapping["employment_type"] = dict()
        mapping["employment_type"]["FT"] = "Full-time"
        mapping["employment_type"]["PT"] = "Part-time"
        mapping["employment_type"]["CT"] = "Contract"
        mapping["employment_type"]["FL"] = "Freelance"
        mapping["remote_ratio"] = dict()
        mapping["remote_ratio"]["0"] = "No remote work"
        mapping["remote_ratio"]["50"] = "Partially remote"
        mapping["remote_ratio"]["100"] = "Fully remote"
        mapping["company_size"] = dict()
        mapping["company_size"]["M"] = "Medium"
        mapping["company_size"]["L"] = "Large"
        mapping["company_size"]["S"] = "Small"
        for name in mapping.keys():
            temp = data[name].copy()
            temp = temp.map(mapping[name])
            data[name] = temp
        drop_col = []
        drop_col.append("salary")
        drop_col.append("salary_currency")
        data.drop(columns=drop_col, inplace=True)
    elif data_name == "movies":
        # basic info
        target_name = "revenue"
        entity_name = "title"
        task = "regression"
        repeated = False
        # preprocess
        mask = data["revenue"] >= 1000  # >= 10000000
        data = data[mask]
        data.dropna(subset="revenue", inplace=True)
        data.reset_index(drop=True, inplace=True)
        temp = data["budget"].copy()
        mask = temp.str.contains(".jpg")
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        temp = data["runtime"].copy()
        mask = temp == 0
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        data["budget"] = data["budget"].astype("float")
        temp = data["budget"].copy()
        temp[temp == 0] = np.nan
        data["budget"] = temp
        data["popularity"] = data["popularity"].astype("float")
        data.fillna(value=np.nan, inplace=True)
        data.columns = data.columns.str.replace(" ", "_")
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(10, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        adjust_cols = [
            "belongs_to_collection",
            "genres",
            "production_companies",
            "production_countries",
            "spoken_languages",
        ]
        extract_name = ["name", "name", "name", "name", "iso_639_1"]
        for i in range(len(adjust_cols)):
            if adjust_cols[i] in data.columns:
                col = []
                for idx in range(len(data)):
                    temp = data[adjust_cols[i]][idx]
                    if str(temp) == "nan":
                        col.append(np.nan)
                    else:
                        temp = ast.literal_eval(temp)
                        if isinstance(temp, list) is False:
                            temp = [temp]
                        if len(temp) == 0:
                            col.append(np.nan)
                        else:
                            temp = pd.DataFrame(temp, index=None)
                            temp[extract_name[i]] = temp[extract_name[i]] + ", "
                            col.append(temp[extract_name[i]].sum()[:-2])
                col = pd.Series(col)
                col = col.rename(adjust_cols[i])
                data[adjust_cols[i]] = col
            else:
                pass
        drop_col = []
        drop_col.append("id")
        drop_col.append("imdb_id")
        drop_col.append("overview")
        drop_col.append("poster_path")
        drop_col.append("original_title")
        drop_col.append("original_language")
        data.drop(columns=drop_col, inplace=True)
        data.drop_duplicates(subset=["title", target_name], inplace=True)
        data.reset_index(drop=True, inplace=True)
    elif data_name == "museums":
        # basic info
        target_name = "Revenue"
        entity_name = "Museum_Name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        mask = data[target_name] > 0
        data = data[mask].copy()
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(100, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("Museum_ID")
        drop_col.append("Income")
        data.drop(columns=drop_col, inplace=True)
        num_cols = data.select_dtypes(exclude="object").columns.tolist()
        num_cols.remove(target_name)
        data[num_cols] = data[num_cols].astype("str")
        for col in num_cols:
            data[col] = data[col].str.strip(".0")
        data.reset_index(drop=True, inplace=True)
    elif data_name == "mydramalist":
        # basic info
        target_name = "rating"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        temp = data["category"].copy()
        mask = temp == "Drama"
        data = data[mask]
        data.reset_index(drop=True, inplace=True)
        temp = data["country"].copy()
        mask = temp == "South Korea"
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for col in data.select_dtypes(include="object").columns:
            temp = data[col].copy()
            temp = temp.astype(str)
            temp[temp.str.isspace()] = np.nan
            temp[temp == "nan"] = np.nan
            data[col] = temp
        data.replace("", "", regex=True, inplace=True)
        data.replace("", "", regex=True, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("url")
        data.drop(columns=drop_col, inplace=True)
        temp = data["duration"].copy()
        temp = temp.astype(str)
        mask = temp.str.contains("hr")
        temp[~mask] = "0"
        temp = temp.str.split("hr").str[0]
        temp1 = temp.astype(float) * 60
        temp = data["duration"].copy()
        temp = temp.astype(str)
        temp[mask] = "0"
        temp = temp.str.split("min").str[0]
        temp2 = temp.astype(float)
        data["duration"] = temp1 + temp2
    elif data_name == "nba_draft":
        # basic info
        target_name = "value_over_replacement"
        entity_name = "player"
        task = "classification"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        temp = data[target_name].copy()
        temp[temp <= 0] = 0
        temp[temp != 0] = 1
        data[target_name] = temp
        keep_col = []
        keep_col.append(target_name)
        keep_col.append("year")
        keep_col.append("overall_pick")
        keep_col.append("team")
        keep_col.append("player")
        keep_col.append("college")
        keep_col.append("years_active")
        data = data[keep_col]
        data["year"] = data["year"].astype("str")
        data["overall_pick"] = data["overall_pick"].astype("str")
        data.reset_index(drop=True, inplace=True)
    elif data_name == "prescription_drugs":
        # basic info
        target_name = "WAC_at_Introduction"
        entity_name = "Drug_Product_Description"
        task = "regression"
        repeated = False
        # preprocess
        unnamed_col = [col for col in data.columns if "Unnamed:" in col]
        data.drop(columns=unnamed_col, inplace=True)
        temp = data["Estimated_Number_of_Patients"].copy()
        temp[temp == 0] = np.nan
        data["Estimated_Number_of_Patients"] = temp
        temp = data["Date_Introduced_to_Market"].copy()
        temp = temp.str.split("-").str[0]
        data["Date_Introduced_to_Market"] = temp
        data.dropna(
            subset=["Drug_Product_Description", "WAC_at_Introduction"], inplace=True
        )
        data.reset_index(drop=True, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(10, data[target_name])
        data = _drop_high_null(data, 0.9)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("NDC_Number")
        data.drop(columns=drop_col, inplace=True)
    elif data_name == "ramen_ratings":
        # basic info
        target_name = "Stars"
        entity_name = "Variety"
        task = "classification"
        repeated = False
        # preprocess
        data["Stars"] = data["Stars"].str.replace("NS", "-1")
        data["Stars"] = data["Stars"].str.replace("NR", "-1")
        data["Stars"] = data["Stars"].str.replace("Unrated", "-1")
        data["Stars"] = data["Stars"].str.split("/").str[0]
        data["Stars"] = data["Stars"].astype("float")
        temp = data["Stars"].copy()
        temp[temp == -1] = np.nan
        data["Stars"] = temp
        data.dropna(subset="Stars", inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("Review_#")
        data.drop(columns=drop_col, inplace=True)
        temp = data["Stars"].copy()
        temp[temp < 4] = 0
        temp[temp != 0] = 1
        data["Stars"] = temp
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
    elif data_name == "roger_ebert":
        # basic info
        target_name = "critic_rating"
        entity_name = "movie_name"
        task = "classification"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        mask = data[target_name] > 2
        data = data[mask]
        data.reset_index(drop=True, inplace=True)
        temp = data[target_name].copy()
        temp[temp < 3.5] = 0
        temp[temp != 0] = 1
        data[target_name] = temp
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data.drop(columns="id", inplace=True)
        data["year"] = data["year"].astype("str")
        data["year"] = data["year"].str[:4]
        temp = data["duration"].str.extract(r"([0-9]+)")[0]
        temp = temp.astype("float")
        data["duration"] = temp
    elif data_name == "rotten_tomatoes":
        # basic info
        target_name = "Rating_Value"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        data.drop(columns="Id", inplace=True)
        data.drop(columns="ReviewCount", inplace=True)
        data.drop(columns="Actors", inplace=True)
        data.rename(columns={"RatingValue": "Rating_Value"}, inplace=True)
        data.rename(columns={"RatingCount": "Rating_Count"}, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data["Year"] = data["Year"].astype("str")
        data["Creator"] = data["Creator"].str.replace(",", ", ", regex=False)
        data["Cast"] = data["Cast"].str.replace(",", ", ", regex=False)
        data["Genre"] = data["Genre"].str.replace(",", ", ", regex=False)
        data["Country"] = data["Country"].str.replace(",", ", ", regex=False)
        data["Language"] = data["Language"].str.replace(",", ", ", regex=False)
        data["Release_Date"] = data["Release_Date"].str.split("(").str[0]
        data["Duration"] = data["Duration"].str.replace("min", "")
        data["Duration"] = data["Duration"].astype("float")
        data["Rating_Count"] = data["Rating_Count"].str.replace(",", "").astype("float")
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "spotify":
        # basic info
        target_name = "popularity"
        entity_name = "track"
        task = "classification"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_cols = []
        drop_cols.append("uri")
        data.drop(columns=drop_cols, inplace=True)
        data["time_signature"] = data["time_signature"].astype("str")
        data["sections"] = data["sections"].astype("str")
        data["key"] = data["key"].astype("str")
        data["duration_ms"] = data["duration_ms"].astype("float")
        temp = data["mode"].copy()
        mapping = {1: "Major", 0: "Minor"}
        temp = temp.map(mapping)
        data["mode"] = temp
    elif data_name == "us_accidents_counts":
        # basic info
        target_name = "Counts"
        entity_name = "City"
        task = "regression"
        repeated = False
        # preprocess
    elif data_name == "us_accidents_severity":
        # basic info
        target_name = "Severity"
        entity_name = "Location"
        task = "classification"
        repeated = False
        # preprocess
    elif data_name == "us_presidential":
        # basic info
        target_name = "target"
        entity_name = "region"
        task = "regression"
        repeated = False
        # preprocess
    elif data_name == "used_cars_24":
        # basic info
        target_name = "Price"
        entity_name = "Model"
        task = "regression"
        repeated = False
        # preprocess
        drop_col = []
        drop_col.append("Unnamed:_0")
        drop_col.append("EMI_(monthly)")
        data.drop(columns=drop_col, inplace=True)
        data.rename(columns={"Driven_(Kms)": "Mileage"}, inplace=True)
        data["Model"] = data["Car_Brand"] + " " + data["Model"]
        temp = data["Ownership"].copy()
        temp = temp.astype(str)
        temp[temp == "1"] = "First"
        temp[temp == "2"] = "Second"
        temp[temp == "3"] = "Third"
        temp[temp == "4"] = "Fourth"
        data["Ownership"] = temp
        data["Model_Year"] = data["Model_Year"].astype(str)
        data["Mileage"] = data["Mileage"].astype(float)
        data["Price"] = data["Price"].astype(float)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(100, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        for col in data.select_dtypes(include="object").columns:
            temp = data[col].copy()
            temp = temp.astype(str)
            temp[temp == "nan"] = np.nan
            data[col] = temp
    elif data_name == "used_cars_benz_italy":
        # Exception with different sep
        data = _load_raw_data(data_name, sep=";")
        # basic info
        target_name = "price"
        entity_name = "model"
        task = "regression"
        repeated = False
        # preprocess
        data.replace("unknown", np.nan, inplace=True)
        drop_col = []
        drop_col.append("Unnamed:_0")
        data.drop(columns=drop_col, inplace=True)
        data["model"] = data["brand"] + " " + data["model"]
        mapping = dict()
        mapping["fuel"] = dict()
        mapping["fuel"]["d"] = "diesel"
        mapping["fuel"]["g"] = "petrol"
        mapping["fuel"]["e"] = "electric"
        mapping["fuel"]["l"] = "lpg"
        mapping["seller_type"] = dict()
        mapping["seller_type"]["d"] = "dealer"
        mapping["seller_type"]["p"] = "private"
        for col in mapping.keys():
            temp = data[col].copy()
            temp = temp.map(mapping[col])
            data[col] = temp
        rename_map = dict()
        rename_map["first_reg"] = "first_registration_date"
        rename_map["mileage_km"] = "mileage"
        rename_map["power_hp"] = "power"
        data.rename(columns=rename_map, inplace=True)
        data["mileage"] = data["mileage"].astype(float)
        data["price"] = data["price"].astype(float)
        temp = data["power"].copy()
        temp = temp.astype(str)
        temp[~temp.str.isnumeric()] = np.nan
        temp = temp.astype(float)
        data["power"] = temp
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        mask = data[target_name] > 100
        data = data[mask]
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(10, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "used_cars_dot_com":
        # basic info
        target_name = "price"
        entity_name = "model"
        task = "regression"
        repeated = False
        # preprocess
        data.rename(columns={"milage": "mileage"}, inplace=True)
        data["model_year"] = data["model_year"].astype(str)
        temp = data["mileage"].copy()
        temp = temp.str.replace(" mi.", "", regex=False).str.replace(
            ",", "", regex=False
        )
        temp = temp.astype(float)
        data["mileage"] = temp
        temp = data["price"].copy()
        temp = temp.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        temp = temp.astype(float)
        data["price"] = temp
        data["model"] = data["brand"] + " " + data["model"]
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(100, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "used_cars_pakistan":
        # basic info
        target_name = "Price"
        entity_name = "Model"
        task = "regression"
        repeated = False
        # preprocess
        data.rename(columns={"Make": "Brand"}, inplace=True)
        data.rename(columns={"Make_Year": "Year"}, inplace=True)
        data.rename(columns={"CC": "Engine_Capacity"}, inplace=True)
        data["Year"] = data["Year"].astype(str)
        data["Engine_Capacity"] = data["Engine_Capacity"].astype(float)
        data["Mileage"] = data["Mileage"].astype(float)
        data["Model"] = data["Brand"] + " " + data["Model"] + ", " + data["Version"]
        drop_col = []
        drop_col.append("Brand")
        drop_col.append("Version")
        data.drop(columns=drop_col, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(100, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "used_cars_saudi_arabia":
        # basic info
        target_name = "Price"
        entity_name = "Model"
        task = "regression"
        repeated = False
        # preprocess
        data["Year"] = data["Year"].astype(str)
        data["Mileage"] = data["Mileage"].astype(float)
        data["Negotiable"] = data["Negotiable"].astype(str)
        data["Model"] = data["Make"] + " " + data["Type"]
        drop_col = []
        drop_col.append("Make")
        drop_col.append("Type")
        data.drop(columns=drop_col, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        mask = data[target_name] < 10
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.emath.logn(100, data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "videogame_sales":
        # basic info
        target_name = "Global_Sales"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log10(data[target_name] * 1e6)
        drop_col = [col for col in data.columns if "Sales" in col]
        drop_col.remove(target_name)
        drop_col.append("Rank")
        data.drop(columns=drop_col, inplace=True)
        data["Year"] = data["Year"].astype("str")
        data["Year"] = data["Year"].str.split(".").str[0]
        temp = data["Year"].copy()
        temp[temp == "nan"] = np.nan
        data["Year"] = temp
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data.drop_duplicates(subset=["Name", "Year", "Global_Sales"], inplace=True)
        data.reset_index(drop=True, inplace=True)
    elif data_name == "whisky":
        # basic info
        target_name = "Meta_Critic"
        entity_name = "Whisky"
        task = "classification"
        repeated = False
        # preprocess
        temp = data["Cost"]
        map = dict()
        map["$$$$$+"] = "over 300 CAD"
        map["$$$$$"] = "between 125 and 300 CAD"
        map["$$$$"] = "between 70 and 125 CAD"
        map["$$$"] = "between 50 and 70 CAD"
        map["$$"] = "between 30 and 50 CAD"
        map["$"] = "less than 30 CAD"
        data["Cost"] = temp.map(map)
        temp = data["Cluster"]
        map = dict()
        map["A"] = "Full-bodied, sweet, pronounced sherry, fruity, honey, spicy"
        map["B"] = "Full-bodied, sweet, pronounced sherry, fruity, floral, malty"
        map["C"] = "Full-bodied, sweet, pronounced sherry, fruity, floral, nutty, spicy"
        map["E"] = "Medium-bodied, medium-sweet, fruity, honey, malty, winey"
        map["F"] = "Full-bodied, sweet, malty, fruity, spicy, smoky"
        map["G"] = "Light-bodied, sweet, apéritif-style, honey, floral, fruity, spicy"
        map["H"] = "Very light-bodied, sweet, apéritif-style, malty, fruity, floral"
        map["I"] = "Medium-bodied, medium-sweet, smoky, medicinal, spicy, fruity, nutty"
        map["J"] = "Full-bodied, dry, very smoky, pungent"
        map["R0"] = "No Rye whisky"
        map["R1"] = "Low Rye whisky"
        map["R2"] = "Standard Rye whisky"
        map["R3"] = "High Rye whisky"
        map["R4"] = "Strong Rye whisky"
        data["Cluster"] = temp.map(map)
        data.drop(columns=["STDEV", "#", "Super_Cluster"], inplace=True)
        data.fillna(value=np.nan, inplace=True)
        data.reset_index(drop=True, inplace=True)
        temp = data["Meta_Critic"].copy()
        temp[temp <= 8.6] = 0
        temp[temp != 0] = 1
        data["Meta_Critic"] = temp
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "wikiliq_beer":
        # basic info
        target_name = "Price"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        data.replace("None", np.nan, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = data[target_name].str.replace("$", "", regex=False)
        data[target_name] = data[target_name].astype(float)
        mask = data[target_name].copy() == 0
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("Unnamed:_0")
        drop_col.append("Rating")
        data.drop(columns=drop_col, inplace=True)
        data["ABV"] = data["ABV"].str[:-1]
        data["ABV"] = data["ABV"].astype(float)
        data["Rate_Count"] = data["Rate_Count"].astype(float)
    elif data_name == "wikiliq_spirit":
        # basic info
        target_name = "Price"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = data[target_name].str.replace("$", "", regex=False)
        data[target_name] = data[target_name].astype(float)
        mask = data[target_name].copy() == 0
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        drop_col = []
        drop_col.append("Unnamed:_0")
        drop_col.append("Rating")
        data.drop(columns=drop_col, inplace=True)
        data["ABV"] = data["ABV"].str[:-1]
        data["ABV"] = data["ABV"].astype(float)
        data["Rate_Count"] = data["Rate_Count"].astype(float)
        data.replace("®", "", regex=True, inplace=True)
        data.replace("™", "", regex=True, inplace=True)
    elif data_name == "wina_pl":
        # basic info
        target_name = "price"
        entity_name = "name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log10(data[target_name])
        data["vegan"] = data["vegan"].astype(str)
        data["natural"] = data["natural"].astype(str)
        data["vintage"] = data["vintage"].astype(str)
        data["vintage"] = data["vintage"].str[:4]
        temp = data["vintage"].copy()
        temp[temp == "nan"] = np.nan
        data["vintage"] = temp
        data["volume"] = data["volume"] * 1000
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "wine_dot_com_prices":
        # basic info
        target_name = "Prices"
        entity_name = "Names"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        mask = data[target_name] == 0
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        temp = data["Names"].copy()
        data["Year"] = temp.str[-4:]
        temp = data["Countrys"].copy()
        data["Grapes"] = temp.str.split("from").str[0]
        data["Region"] = temp.str.split("from").str[-1]
        temp = data["Capacity"].copy()
        temp = temp.str.replace("ml", "", regex=False)
        temp = temp.astype("float")
        data["Capacity"] = temp
        drop_col = []
        drop_col.append("Countrys")
        data.drop(columns=drop_col, inplace=True)
    elif data_name == "wine_dot_com_ratings":
        # basic info
        target_name = "Ratings"
        entity_name = "Names"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        mask = data[target_name] == 0
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        temp = data["Names"].copy()
        data["Year"] = temp.str[-4:]
        temp = data["Countrys"].copy()
        data["Grapes"] = temp.str.split("from").str[0]
        data["Region"] = temp.str.split("from").str[-1]
        temp = data["Capacity"].copy()
        temp = temp.str.replace("ml", "", regex=False)
        temp = temp.astype("float")
        data["Capacity"] = temp
        drop_col = []
        drop_col.append("Countrys")
        data.drop(columns=drop_col, inplace=True)
    elif data_name == "wine_enthusiasts_prices":
        # basic info
        target_name = "price"
        entity_name = "title"
        task = "regression"
        repeated = False
        # preprocess
        drop_col = []
        drop_col.append("Unnamed:_0")
        drop_col.append("region_1")
        drop_col.append("region_2")
        drop_col.append("taster_twitter_handle")
        data.drop(columns=drop_col, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "wine_enthusiasts_ratings":
        # basic info
        target_name = "points"
        entity_name = "title"
        task = "regression"
        repeated = False
        # preprocess
        drop_col = []
        drop_col.append("Unnamed:_0")
        drop_col.append("region_1")
        drop_col.append("region_2")
        drop_col.append("taster_twitter_handle")
        data.drop(columns=drop_col, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "wine_vivino_price":
        # basic info
        target_name = "Price"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        mask = data[target_name] == 0
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        data[target_name] = np.log(data[target_name])
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data["Number_Of_Ratings"] = data["Number_Of_Ratings"].astype(float)
        data["Region"] = data["Region"] + ", " + data["Country"]
        drop_col = []
        drop_col.append("Country")
        data.drop(columns=drop_col, inplace=True)
    elif data_name == "wine_vivino_rating":
        # basic info
        target_name = "Rating"
        entity_name = "Name"
        task = "regression"
        repeated = False
        # preprocess
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data["Number_Of_Ratings"] = data["Number_Of_Ratings"].astype(float)
    elif data_name == "yelp":
        # Exception with different file_type
        data = _load_raw_data(data_name, file_type="json")
        # basic info
        target_name = "stars"
        entity_name = "name"
        task = "classification"
        repeated = False
        # preprocess
        temp = data["categories"].copy()
        mask = temp.str.contains("Restaurants") | temp.str.contains("Food")
        data = data[mask].copy()
        data.reset_index(drop=True, inplace=True)
        data.dropna(subset=[target_name], inplace=True)
        data.reset_index(drop=True, inplace=True)
        temp = data["stars"].copy()
        temp[temp <= 3.5] = 0
        temp[temp != 0] = 1
        data["stars"] = temp
        temp = data["attributes"].copy()
        temp = temp.to_list()
        temp1 = [{} if x is None else x for x in temp]
        attributes_df = pd.DataFrame(temp1)
        attribute_extract_cols = []
        attribute_extract_cols.append(("RestaurantsPriceRange2", "price_range"))
        for col in attribute_extract_cols:
            data[col[1]] = attributes_df[col[0]]
            temp = data[col[1]].copy()
            temp[temp.isnull()] = np.nan
            temp[temp == "None"] = np.nan
            data[col[1]] = temp
        temp = data["hours"].copy()
        temp = temp.astype("str")
        temp = temp.str.extractall(r"([A-Z]+)")
        temp = temp.groupby(level=0).sum()[0]
        temp = temp.str.replace("N", "")
        temp = temp.str.len()
        temp = temp.astype("float")
        temp[temp == 0] = np.nan
        data["number_of_days_open"] = temp
        temp = data["is_open"].copy()
        temp[temp == 1] = "open"
        temp[temp == 0] = "closed"
        temp = temp.astype("str")
        data["is_open"] = temp
        data["review_count"] = data["review_count"].astype("float")
        data.drop(columns="hours", inplace=True)
        data.drop(columns="attributes", inplace=True)
        data.drop(columns="business_id", inplace=True)
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
    elif data_name == "zomato":
        # basic info
        target_name = "rating"
        entity_name = "name"
        task = "classification"
        repeated = False
        # preprocess
        data[target_name].replace("--", np.nan, inplace=True)
        data.dropna(subset=target_name, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data[target_name] = data[target_name].astype("float")
        temp = data[target_name].copy()
        temp[temp < 4] = 0
        temp[temp != 0] = 1
        data[target_name] = temp
        data = _drop_high_null(data)
        data = _drop_single_unique(data)
        data["cost"] = data["cost"].str[1:]
        data["cost"] = data["cost"].astype("float")
        drop_col = []
        drop_col.append("Unnamed:_0")
        drop_col.append("id")
        drop_col.append("menu")
        data.drop(columns=drop_col, inplace=True)

    # Save data
    _save_processed_data(data_name, data, target_name, entity_name, task, repeated)

    return None


# Main
def main(data_name_list):

    if "all" in data_name_list:
        data_name_list = carte_datalist
    else:
        if isinstance(data_name_list, list) == False:
            data_name_list = [data_name_list]

    for data_name in data_name_list:
        preprocess_data(data_name)
        print(f"{data_name} complete!")

    return None


if __name__ == "__main__":

    # Set parser
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw data.")
    parser.add_argument(
        "-dt",
        "--data_name_list",
        nargs="+",
        type=str,
        help="data_name to preprocess",
    )
    args = parser.parse_args()

    main(args.data_name_list)
