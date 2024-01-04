import numpy as np

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import *

from numpy import linalg as LA
import scipy.sparse as sps


# Custom Hybrid class

class HybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid"

    def __init__(self, URM_train):
        super(HybridRecommender, self).__init__(URM_train)
        self.slim_recommender = SLIMElasticNetRecommender(self.URM_train)
        self.RP3_recommender = RP3betaRecommender(self.URM_train)
        self.userknn_recommender = UserKNNCFRecommender(self.URM_train)
        self.P3_recommender = P3alphaRecommender(self.URM_train)
        self.itemknn_recommender = ItemKNNCFRecommender(self.URM_train)
        self.MV_recommender = MultVAERecommender(self.URM_train)
        self.ials_recommender = IALSRecommender(self.URM_train)
        self.ease_recommender = EASE_R_Recommender(self.URM_train)

    def fit(self):
        if self.URM_train.shape[0] == 22222:
            self.slim_recommender.load_model(folder_path="slim_models", file_name="slim_24.zip")
        else:
            self.slim_recommender.fit(topK=8894, l1_ratio=0.05565733019999427, alpha=0.0012979360257937668)
        self.RP3_recommender.fit(alpha=0.20026352123406477, beta=0.15999879728761354, topK=32)
        self.userknn_recommender.fit(topK=469, shrink=38, similarity='asymmetric', normalize=True,
                                       feature_weighting='TF-IDF', asymmetric_alpha=0.40077406933762383)
        self.itemknn_recommender.fit(topK=31, shrink=435, similarity='tversky', normalize=True,
                                       feature_weighting='BM25', tversky_alpha=0.17113169506422393, tversky_beta=0.5684024974085575)
        self.knn_recommender = TwoScoresHybridRecommender(self.URM_train, self.userknn_recommender, self.itemknn_recommender)
        self.knn_recommender.fit(alpha=0.022195783788315104)
        if self.URM_train.shape[0] == 22222:
            self.ease_recommender.load_model(folder_path="slim_models", file_name="ease_all.zip")
        else:
            self.ease_recommender.fit(topK=24, l2_norm=37.54323189430143)
        self.hybrid_one = ScoresHybridRecommender(self.URM_train, self.knn_recommender, self.MV_recommender, self.IALS_recommender)
        self.hybrid_one.fit(alpha=0.7509279131476281,beta=0.028318614437090585)
        self.hybrid_li = LinearHybridRecommender(self.URM_train, [self.slim_recommender, self.RP3_recommender, self.knn_recommender, self.ease_recommender])
        self.hybrid_li.fit(alphas=[0.8857667100747117, 0.6379807942443021, 0.0690929184825888, 0.11953478623354052])
        self.hybrid_hi = TwoScoresHybridRecommender(self.URM_train, self.slim_recommender, self.RP3_recommender)
        self.hybrid_hi.fit(alpha=0.6201320790279279)


    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = np.empty([len(user_id_array), 22222])
        for i in range(len(user_id_array)):

            interactions = len(self.URM_train[user_id_array[i], :].indices)

            if interactions == 1:
                w = self.hybrid_one._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w
            elif interactions >= 2 & interactions <= 14:
                w = self.hybrid_li._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w
            else:
                w = self.hybrid_hi._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w

        return item_weights


class LinearHybridRecommender(BaseRecommender):
    """
    This recommender merges N recommenders by weighting their ratings
    """

    RECOMMENDER_NAME = "LinearHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'LinearHybridRecommender'

        super(LinearHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alphas=None):
        self.alphas = alphas

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        result = self.alphas[0] * self.recommenders[0]._compute_item_score(user_id_array, items_to_compute)
        for index in range(1, len(self.alphas)):
            result = result + self.alphas[index] * self.recommenders[index]._compute_item_score(user_id_array,
                                                                                                items_to_compute)
        return result


class LinearZScoreNormalizedHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "LinearZScoreNormalizedHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'ZScoreNormHybridRecommender'

        super(LinearZScoreNormalizedHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alphas=None):
        self.alphas = alphas
        self.mean_predictions = []
        self.std_predictions = []

        # Fit each individual recommender
        for recommender in self.recommenders:
            # Compute mean and standard deviation of predictions
            predictions = recommender._compute_item_score(np.arange(self.URM_train.shape[0]))
            self.mean_predictions.append(np.mean(predictions))
            self.std_predictions.append(np.std(predictions))

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        result = self.alphas[0] * self.normalize_predictions(self.recommenders[0]._compute_item_score(user_id_array, items_to_compute), 0)
        for index in range(1, len(self.alphas)):
            result = result + self.alphas[index] * self.normalize_predictions(self.recommenders[index]._compute_item_score(user_id_array, items_to_compute), index)
        return result

    def normalize_predictions(self, predictions, index):
        # Apply z-score normalization using mean and std of predictions from each recommender
        return (predictions - self.mean_predictions[index]) / self.std_predictions[index]

class DifferentLossScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        self.RECOMMENDER_NAME = ''
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender_1.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender_2.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'DifferentLossScoresHybridRecommender'

        super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, norm, alpha=0.5):

        self.alpha = alpha
        self.norm = norm

    def _compute_item_score(self, user_id_array, items_to_compute):

        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)

        if norm_item_weights_1 == 0:
            raise ValueError(
                "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))

        if norm_item_weights_2 == 0:
            raise ValueError(
                "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))

        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (
                1 - self.alpha)

        return item_weights


class MajVotHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "MajorityVotingHybrid"

    def __init__(self, URM_train, recommenders):
        super(MajVotHybridRecommender, self).__init__(URM_train)
        self.recommenders = recommenders

    def fit(self):
        # Fit each individual recommender
        for recommender in self.recommenders:
            recommender.fit()

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        Computes the score for each item for the given users.

        :param user_id_array: array containing the user indices whose recommendations need to be computed
        :param items_to_compute: array containing the items whose scores are to be computed.
                                 If None, all items are computed.
        :return: array (len(user_id_array), n_items) with the score.
        """
        # Initialize an array to store the aggregated votes
        aggregated_votes = None

        # Iterate over each recommender
        for recommender in self.recommenders:
            # Compute scores for each recommender
            scores = recommender._compute_item_score(user_id_array, items_to_compute)

            # Initialize the aggregated_votes array with the shape of scores from the first recommender
            if aggregated_votes is None:
                aggregated_votes = np.zeros_like(scores)

            # Convert scores to binary votes (1 for recommended items, 0 otherwise)
            recommender_votes = scores > 0
            aggregated_votes += recommender_votes

        return aggregated_votes

    # Other methods of the class


class ScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3, verbose=True):
        self.RECOMMENDER_NAME = ''
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender_1.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender_2.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender_3.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'ScoresHybridRecommender'

        super(ScoresHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3

    def fit(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta + item_weights_3 * (
                    1 - self.alpha - self.beta)

        return item_weights

class TwoScoresHybridRecommender(BaseRecommender):
    """ TwoScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "TwoScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2, verbose=True):
        self.RECOMMENDER_NAME = ''
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender_1.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender_2.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'TwoScoresHybridRecommender'

        super(TwoScoresHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)
        return item_weights

