import numpy as np
from scipy.spatial.distance import cdist


def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    """
    Returns a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the
    i-th query descriptor. The descriptor vectors are MxQ and MxD where M is the descriptor dimension and Q and D the
    amount of query and database descriptors respectively. matches(i) will be -1 if there is no database descriptor
    with an SSD < lambda * min(SSD). No elements of matches will be equal except for the -1 elements.
    """
    # shape: (Q, D) -- in this case (200, 200)
    # distance from each query descriptor to each database descriptor
    dists = cdist(query_descriptors.T, database_descriptors.T, "euclidean")

    # shape: (200, 1)
    # for each query_descriptor, which db_descriptor (index) is closest (argmin)
    matches = np.argmin(dists, axis=1)

    # shape: (200, 1)
    # keep only distances that matched in `matches`
    dists = dists[np.arange(matches.shape[0]), matches]

    # scalar
    # min distance between any two descriptors across both sets
    min_non_zero_dist = dists.min()

    # keep only descriptors with small distance
    # adaptive threshold (because there should be at least one match)
    matches[dists >= match_lambda * min_non_zero_dist] = -1

    # remove double matches:
    # if a db_descriptor was assigned to several query_descriptors, keep only 1 match
    unique_matches = np.ones_like(matches) * -1
    _, unique_match_idxs = np.unique(matches, return_index=True)
    unique_matches[unique_match_idxs] = matches[unique_match_idxs]

    return unique_matches
