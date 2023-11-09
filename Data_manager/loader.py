import pandas as pd


def load_URM(file_path):
    data = pd.read_csv(filepath_or_buffer=file_path,
                       sep=",",
                       dtype={0: int, 1: int, 2: int},
                       engine='python')
    data.columns = ["UserID", "ItemID", "Interaction"]

    import scipy.sparse as sps

    mapped_id, original_id = pd.factorize(data["UserID"].unique())
    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    mapped_id, original_id = pd.factorize(data["ItemID"].unique())
    item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    data["UserID"] = data["UserID"].map(user_original_ID_to_index)
    data["ItemID"] = data["ItemID"].map(item_original_ID_to_index)
    n_users = len(data["UserID"].unique())
    n_items = len(data["ItemID"].unique())

    URM_all = sps.csr_matrix((data["Interaction"].values,
                              (data["UserID"].values, data["ItemID"].values)),
                             shape=(n_users, n_items))
    return URM_all
