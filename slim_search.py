import pandas as pd

from Data_manager.UserUtils import *
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.SLIMElasticNetRecommender import *


def getTarget(recommender, top_recommender):
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
    return submission


df_target = pd.read_csv("submission_top_slim.csv")

URM_all = getURM_all()
top_pop = TopPop(URM_all)
top_pop.fit()
i = 0
while 1:
    print("current iteration is:" + str(i))
    # slim = SLIMElasticNetRecommender(URM_all)
    slim = MultiThreadSLIM_SLIMElasticNetRecommender(URM_all)
    slim.fit(topK=8894, l1_ratio=0.05565733019999427, alpha=0.0012979360257937668, workers=7)
    URM_try = getTarget(slim, top_pop)

    df = URM_try.compare(df_target)

    val = df['item_list'].count()
    print("current diffs:" + val)
    i += 1
    if val == 0:
        np.save(slim.W_sparse, "W_sparse.npy")
        print(slim.W_sparse)
        break
