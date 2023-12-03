def getURM_all(remove_users_with_more_than_iterations=None):
    import pandas as pd
    import scipy.sparse as sps

    URM_path = "data_train.csv"
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                    sep=",",
                                    dtype={0: int, 1: int, 2: int},
                                    engine='python')

    URM_all_dataframe.columns = ["UserID", "ItemID", "Interaction"]

    # Mapping user and item IDs to indices
    mapped_id, original_id = pd.factorize(URM_all_dataframe["UserID"].unique())
    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    mapped_id, original_id = pd.factorize(URM_all_dataframe["ItemID"].unique())
    item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    # Mapping original IDs to indices in the DataFrame
    URM_all_dataframe["UserID"] = URM_all_dataframe["UserID"].map(user_original_ID_to_index)
    URM_all_dataframe["ItemID"] = URM_all_dataframe["ItemID"].map(item_original_ID_to_index)

    # Creating the URM matrix
    n_users = len(user_original_ID_to_index)
    n_items = len(item_original_ID_to_index)
    URM_all = sps.csr_matrix((URM_all_dataframe["Interaction"].values,
                              (URM_all_dataframe["UserID"].values, URM_all_dataframe["ItemID"].values)),
                             shape=(n_users, n_items))

    # Remove users with more than a specified number of iterations
    if remove_users_with_more_than_iterations is not None:
        user_interactions = URM_all.sum(axis=1).A1  # Get sum of interactions per user
        users_to_keep = user_interactions <= remove_users_with_more_than_iterations
        URM_all = URM_all[users_to_keep, :]  # Filter out users

    return URM_all

def generateSubmission(recommender, top_recommender):
    import pandas as pd

    URM_path = "data_train.csv"
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                    sep=",",
                                    # header=None,
                                    dtype={0: int, 1: int, 2: int},
                                    engine='python')

    URM_all_dataframe.columns = ["UserID", "ItemID", "Interaction"]
    import scipy.sparse as sps

    mapped_id, original_id = pd.factorize(URM_all_dataframe["UserID"].unique())
    user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    mapped_id, original_id = pd.factorize(URM_all_dataframe["ItemID"].unique())
    item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    URM_path = "data_target_users_test.csv"
    URM_all_submission_dataframe = pd.read_csv(filepath_or_buffer=URM_path,
                                               sep=",",
                                               # header=None,
                                               dtype={0: int},
                                               engine='python')

    URM_all_submission_dataframe.columns = ["UserID"]
    URM_all_submission_dataframe["UserIDMapped"] = URM_all_submission_dataframe["UserID"].map(user_original_ID_to_index)
    # First, create a mask for non-null 'UserIDMapped' values
    mask = ~URM_all_submission_dataframe['UserIDMapped'].isna()
    URM_all_submission_dataframe['item_list'] = top_recommender.recommend(URM_all_submission_dataframe['UserIDMapped'],
                                                                          cutoff=10, remove_seen_flag=False)
    URM_all_submission_dataframe.loc[mask, 'item_list'] = pd.Series(
        recommender.recommend(URM_all_submission_dataframe.loc[mask, 'UserIDMapped'].to_numpy().astype(int), cutoff=10,
                              remove_seen_flag=True), index=URM_all_submission_dataframe.index[mask])

    item_index_to_original_id = item_original_ID_to_index.reset_index().set_index(0).to_numpy().reshape(-1)
    URM_all_submission_dataframe['item_list'] = URM_all_submission_dataframe['item_list'].apply(
        lambda item_indices: [item_index_to_original_id[index] for index in item_indices])

    submission = URM_all_submission_dataframe[["UserID", "item_list"]]
    submission.columns = ["user_id", "item_list"]
    submission['item_list'] = submission['item_list'].apply(lambda x: ' '.join(map(str, x)))  # remove brakets

    # Convert item_list to string format
    submission['item_list'] = submission['item_list'].astype(str)

    # Save to CSV without quotes and with the desired format
    # submission.to_csv("submission.csv", index=False, header=["user_id", "item_list"], quoting=csv.QUOTE_NONE, sep=',')
    def write_csv_without_quotes(df, file_path):
        with open(file_path, 'w', newline='') as f:
            f.write("user_id,item_list\n")
            for index, row in df.iterrows():
                f.write(f"{row['user_id']},{row['item_list']}\n")

    # Save the DataFrame to CSV without quotes and with the desired format
    write_csv_without_quotes(submission, "submission.csv")