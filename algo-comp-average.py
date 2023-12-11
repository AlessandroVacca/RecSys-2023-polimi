from Data_manager.UserUtils import *
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import *
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender

import numpy as np
import scipy.sparse as sps
from Evaluation.Evaluator import EvaluatorHoldout
import pandas as pd
for index in range(1, 100):
    URM_all = getURM_all()
    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    # SETUP EVALUATORS

    # evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    profile_length = np.ediff1d(sps.csr_matrix(URM_train_validation).indptr)


    MAP_recommender_per_group = {}

    collaborative_recommender_class = {"TopPop": TopPop,
                                       "UserKNNCF": UserKNNCFRecommender,
                                       "ItemKNNCF": ItemKNNCFRecommender,
                                       "P3alpha": P3alphaRecommender,
                                       "RP3beta": RP3betaRecommender,
                                       "PureSVD": PureSVDRecommender,
                                       "SLIM_ELASTIC": SLIMElasticNetRecommender,
                                       }
    collaborative_recommender_class.items()

    recommender_object_dict = {}

    for label, recommender_class in collaborative_recommender_class.items():
        recommender_object = recommender_class(URM_train_validation)
        recommender_object_dict[label] = recommender_object

    recommender_object_dict["TopPop"].fit()
    recommender_object_dict["SLIM_ELASTIC"].fit(topK=8894, l1_ratio=0.05565733019999427, alpha=0.0012979360257937668)
                                                #workers=7)
    recommender_object_dict["PureSVD"].fit(num_factors=70, random_seed=42)
    recommender_object_dict["P3alpha"].fit(topK=76, alpha=0.377201600381895, normalize_similarity=True)
    recommender_object_dict["RP3beta"].fit(topK=101, alpha=0.3026342852596128, beta=0.058468783118329024)
    recommender_object_dict["UserKNNCF"].fit(topK=469, shrink=38, similarity='asymmetric', normalize=True,
                                             feature_weighting='TF-IDF', asymmetric_alpha=0.40077406933762383)
    recommender_object_dict["ItemKNNCF"].fit(topK=31, shrink=435, similarity='tversky', normalize=True,
                                             feature_weighting='BM25', tversky_alpha=0.17113169506422393,
                                             tversky_beta=0.5684024974085575)


    MAP_recommender_per_group = {}
    cutoff = 10

    # Define the cutoffs for profile length for each group
    profile_length_cutoffs = [2, 3, 4, 6, 8, 25, 80, 100, 250, 1500]

    for group_id in range(len(profile_length_cutoffs)):

        # Select users whose profile length is less than the cutoff for the group and greater than or equal to the cutoff for the previous group
        if group_id == 0:
            users_in_group_flag = profile_length < profile_length_cutoffs[group_id]
        else:
            users_in_group_flag = np.logical_and(profile_length >= profile_length_cutoffs[group_id - 1],
                                                 profile_length < profile_length_cutoffs[group_id])
        users_in_group = np.arange(len(profile_length))[users_in_group_flag]

        if len(users_in_group) > 0:
            users_in_group_p_len = profile_length[users_in_group]

            print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
                group_id,
                users_in_group.shape[0],
                users_in_group_p_len.mean(),
                np.median(users_in_group_p_len),
                users_in_group_p_len.min(),
                users_in_group_p_len.max()))

            # Select users not in the current group
            users_not_in_group_flag = np.isin(np.arange(len(profile_length)), users_in_group, invert=True)
            users_not_in_group = np.arange(len(profile_length))[users_not_in_group_flag]

            evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

            for label, recommender in recommender_object_dict.items():
                result_df, _ = evaluator_test.evaluateRecommender(recommender)
                if label in MAP_recommender_per_group:
                    MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
                else:
                    MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]
        else:
            print(f"Group {group_id} has no users.")

    df = pd.DataFrame(MAP_recommender_per_group)
    df.to_csv(f'result_{index}.csv', header=True)
    print(f'{index} done')