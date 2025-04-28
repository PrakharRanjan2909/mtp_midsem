# diagnostics.py

import numpy as np

def assign_holes_to_clusters(hmms, hole_sequences):
    """
    For each hole, picks the HMM with the highest log-likelihood.
    
    Returns:
      assignments: list of int (index of the winning HMM for each hole)
      loglikes: list of float (the winning log-likelihood)
    """
    assignments = []
    loglikes = []
    K = len(hmms)
    for seq in hole_sequences:
        ll_values = [h.log_likelihood(seq) for h in hmms]
        winner_idx = np.argmax(ll_values)
        assignments.append(winner_idx)
        loglikes.append(ll_values[winner_idx])
    return assignments, loglikes


def label_clusters_temporally(db_to_holes, assignments):
    """
    A simple heuristic:
      - For each drill-bit, we see the order in which clusters appear 
        from first hole -> last hole.
      - We gather each cluster's "first appearance index" across bits.
      - We sort clusters by that average first-appearance => 
        earliest => 'good', next => 'medium', last => 'bad'.

    Returns: cluster_order (list of int),
             e.g. [2, 0, 1] means cluster #2 is the earliest, #0 next, #1 last
    """
    from collections import defaultdict
    cluster_positions = defaultdict(list)  # cluster -> list of earliest appearances

    for db_name, hole_indices in db_to_holes.items():
        # hole_indices are in chronological order for that DB
        cluster_seq = [assignments[h] for h in hole_indices]
        first_appearance = {}
        prev = None
        for i, c in enumerate(cluster_seq):
            if c not in first_appearance:
                first_appearance[c] = i
        # record them
        for c, pos in first_appearance.items():
            cluster_positions[c].append(pos)

    # compute average position
    cluster_avg = []
    for c, pos_list in cluster_positions.items():
        avgpos = sum(pos_list)/len(pos_list)
        cluster_avg.append( (c, avgpos) )

    # sort by ascending avgpos
    cluster_avg.sort(key=lambda x: x[1])
    cluster_order = [x[0] for x in cluster_avg]
    return cluster_order
