""" This library of supporting functions are written to perform graph partitioning according to the following reference

    References
    ----------
        .. [1] Peixoto, Tiago P. 'Entropy of stochastic blockmodel ensembles.'
               Physical Review E 85, no. 5 (2012): 056122.
        .. [2] Peixoto, Tiago P. 'Parsimonious module inference in large networks.'
               Physical review letters 110, no. 14 (2013): 148701.
        .. [3] Karrer, Brian, and Mark EJ Newman. 'Stochastic blockmodels and community structure in networks.'
               Physical Review E 83, no. 1 (2011): 016107."""
import numpy as np
from scipy import sparse as sparse
import scipy.misc as misc
from munkres import Munkres # for correctness evaluation
import sys
from multiprocessing import sharedctypes
import ctypes
from compute_delta_entropy import compute_delta_entropy

use_graph_tool_options = False # for visualiziing graph partitions (optional)
if use_graph_tool_options:
    import graph_tool.all as gt

import random
def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

def load_graph(input_filename, load_true_partition, strm_piece_num=None, out_neighbors=None, in_neighbors=None, permutate=False):
    """Load the graph from a TSV file with standard format, and the truth partition if available

        Parameters
        ----------
        input_filename : str
                input file name not including the .tsv extension
        true_partition_available : bool
                whether the truth partition is available
        strm_piece_num : int, optional
                specify which stage of the streaming graph to load
        out_neighbors, in_neighbors : list of ndarray, optional
                existing graph to add to. This is used when loading the streaming graphs one stage at a time. Note that
                the truth partition is loaded all together at once.

        Returns
        -------
        out_neighbors : list of ndarray; list length is N, the number of nodes
                each element of the list is a ndarray of out neighbors, where the first column is the node indices
                and the second column the corresponding edge weights
        in_neighbors : list of ndarray; list length is N, the number of nodes
                each element of the list is a ndarray of in neighbors, where the first column is the node indices
                and the second column the corresponding edge weights
        N : int
                number of nodes in the graph
        E : int
                number of edges in the graph
        true_b : ndarray (int) optional
                array of truth block assignment for each node

        Notes
        -----
        The standard tsv file has the form for each row: "from to [weight]" (tab delimited). Nodes are indexed from 0
        to N-1. If available, the true partition is stored in the file `filename_truePartition.tsv`."""

    # read the entire graph CSV into rows of edges
    if (strm_piece_num == None):
        edge_rows = np.loadtxt('{}.tsv'.format(input_filename), delimiter='\t', dtype=np.int64)
    else:
        edge_rows = np.loadtxt('{}_{}.tsv'.format(input_filename, strm_piece_num), delimiter='\t', dtype=np.int64)

    if (out_neighbors == None):  # no previously loaded streaming pieces
        N = edge_rows[:, 0:2].max()  # number of nodes
        out_neighbors = [[] for i in range(N)]
        in_neighbors = [[] for i in range(N)]
    else:  # add to previously loaded streaming pieces
        N = max(edge_rows[:, 0:2].max(), len(out_neighbors))  # number of nodes
        out_neighbors = [list(out_neighbors[i]) for i in range(len(out_neighbors))]
        out_neighbors.extend([[] for i in range(N - len(out_neighbors))])
        in_neighbors = [list(in_neighbors[i]) for i in range(len(in_neighbors))]
        in_neighbors.extend([[] for i in range(N - len(in_neighbors))])
    weights_included = edge_rows.shape[1] == 3

    if permutate:
        permutation = [i for i in random_permutation(range(N))]

    # load edges to list of lists of out and in neighbors
    for i in range(edge_rows.shape[0]):
        if weights_included:
            edge_weight = edge_rows[i, 2]
        else:
            edge_weight = 1
        # -1 on the node index since Python is 0-indexed and the standard graph TSV is 1-indexed
        from_idx = edge_rows[i, 0] - 1
        to_idx = edge_rows[i, 1] - 1

        if permutate:
            from_idx = permutation[from_idx]
            to_idx = permutation[to_idx]

        out_neighbors[from_idx].append([to_idx, edge_weight])
        in_neighbors [to_idx].append([from_idx, edge_weight])

    # convert each neighbor list to neighbor numpy arrays for faster access
    for i in range(N):
        if len(out_neighbors[i]) > 0:
            out_neighbors[i] = np.array(out_neighbors[i], dtype=int)
        else:
            out_neighbors[i] = np.array(out_neighbors[i], dtype=int).reshape((0,2))
    for i in range(N):
        if len(in_neighbors[i]) > 0:
            in_neighbors[i] = np.array(in_neighbors[i], dtype=int)
        else:
            in_neighbors[i] = np.array(in_neighbors[i], dtype=int).reshape((0,2))

    E = sum(len(v) for v in out_neighbors)  # number of edges

    if load_true_partition:
        # read the entire true partition CSV into rows of partitions
        true_b_rows = np.loadtxt('{}_truePartition.tsv'.format(input_filename), delimiter='\t', dtype=np.int64)
        true_b = np.ones(true_b_rows.shape[0], dtype=int) * -1  # initialize truth assignment to -1 for 'unknown'
        for i in range(true_b_rows.shape[0]):
            true_b[true_b_rows[i, 0] - 1] = int(
                true_b_rows[i, 1] - 1)  # -1 since Python is 0-indexed and the TSV is 1-indexed

    if permutate:
        #in_neighbors = [in_neighbors[i] for i in permutation]
        #out_neighbors = [out_neighbors[i] for i in permutation]
        if load_true_partition:
            true_b = [true_b[i] for i in permutation]

    if load_true_partition:
        return out_neighbors, in_neighbors, N, E, true_b
    else:
        return out_neighbors, in_neighbors, N, E

def decimate_graph(out_neighbors, in_neighbors, true_partition, decimation, decimated_piece):
    """
    """
    in_neighbors = in_neighbors[decimated_piece::decimation]
    out_neighbors = out_neighbors[decimated_piece::decimation]
    true_partition = true_partition[decimated_piece::decimation]
    E = sum(len(v) for v in out_neighbors)
    N = np.int64(len(in_neighbors))

    for i in range(N):
        xx = (in_neighbors[i][:,0] % decimation) == decimated_piece
        in_neighbors[i] = in_neighbors[i][xx, :]
        xx = (out_neighbors[i][:,0] % decimation) == decimated_piece
        out_neighbors[i] = out_neighbors[i][xx, :]

    for i in range(N):
        in_neighbors[i][:,0] = in_neighbors[i][:,0] / decimation
        out_neighbors[i][:,0] = out_neighbors[i][:,0] / decimation

    return out_neighbors, in_neighbors, N, E, true_partition


def initialize_partition_variables():
    """Initialize variables for the iterations to find the best partition with the optimal number of blocks

        Returns
        -------
        optimal_B_found : bool
                    flag for whether the optimal block has been found
        old_b : list of length 3
                    holds the best three partitions so far
        old_M : list of length 3
                    holds the edge count matrices for the best three partitions so far
        old_d : list of length 3
                        holds the block degrees for the best three partitions so far
        old_d_out : list of length 3
                    holds the out block degrees for the best three partitions so far
        old_d_in : list of length 3
                    holds the in block degrees for the best three partitions so far
        old_S : list of length 3
                    holds the overall entropy for the best three partitions so far
        old_B : list of length 3
                    holds the number of blocks for the best three partitions so far
        graph_object : list
                    empty for now and will store the graph object if graphs will be visualized"""

    optimal_B_found = False
    old_b = [[], [], []]  # partition for the high, best, and low number of blocks so far
    old_M = [[], [], []]  # edge count matrix for the high, best, and low number of blocks so far
    old_d = [[], [], []]  # block degrees for the high, best, and low number of blocks so far
    old_d_out = [[], [], []]  # out block degrees for the high, best, and low number of blocks so far
    old_d_in = [[], [], []]  # in block degrees for the high, best, and low number of blocks so far
    old_S = [np.Inf, np.Inf, np.Inf] # overall entropy for the high, best, and low number of blocks so far
    old_B = [[], [], []]  # number of blocks for the high, best, and low number of blocks so far
    graph_object = None
    return optimal_B_found, old_b, old_M, old_d, old_d_out, old_d_in, old_S, old_B, graph_object


def initialize_edge_counts(out_neighbors, B, b, use_sparse = False):
    """Initialize the edge count matrix and block degrees according to the current partition

        Parameters
        ----------
        out_neighbors : list of ndarray; list length is N, the number of nodes
                    each element of the list is a ndarray of out neighbors, where the first column is the node indices
                    and the second column the corresponding edge weights
        B : int
                    total number of blocks in the current partition
        b : ndarray (int)
                    array of block assignment for each node
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        d_out : ndarray (int)
                    the current out degree of each block
        d_in : ndarray (int)
                    the current in degree of each block
        d : ndarray (int)
                    the current total degree of each block

        Notes
        -----
        Compute the edge count matrix and the block degrees from scratch"""

    if use_sparse: # store interblock edge counts as a sparse matrix
        M = sparse.lil_matrix((B, B), dtype=int)
    else:
        M = np.zeros((B,B), dtype=int)

    # compute the initial interblock edge count
    for v in range(len(out_neighbors)):
        if len(out_neighbors[v]) > 0:
            k1 = b[v]
            k2, inverse_idx = np.unique(b[out_neighbors[v][:, 0]], return_inverse=True)
            count = np.bincount(inverse_idx, weights=out_neighbors[v][:, 1]).astype(int)
            M[k1, k2] += count
    # compute initial block degrees
    d_out = np.asarray(M.sum(axis=1)).ravel()
    d_in = np.asarray(M.sum(axis=0)).ravel()
    d = d_out + d_in
    return M, d_out, d_in, d


def propose_new_partition(r, neighbors_out, neighbors_in, b, M, d, B, agg_move, use_sparse, n_proposals=10):
    """Propose a new block assignment for the current node or block

        Parameters
        ----------
        r : int
                    current block assignment for the node under consideration
        neighbors_out : ndarray (int) of two columns
                    out neighbors array where the first column is the node indices and the second column is the edge weight
        neighbors_in : ndarray (int) of two columns
                    in neighbors array where the first column is the node indices and the second column is the edge weight
        b : ndarray (int)
                    array of block assignment for each node
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        d : ndarray (int)
                    total number of edges to and from each block
        B : int
                    total number of blocks
        agg_move : bool
                    whether the proposal is a block move
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        s : int
                    proposed block assignment for the node under consideration
        k_out : int
                    the out degree of the node
        k_in : int
                    the in degree of the node
        k : int
                    the total degree of the node

        Notes
        -----
        - d_u: degree of block u

        Randomly select a neighbor of the current node, and obtain its block assignment u. With probability \frac{B}{d_u + B}, randomly propose
        a block. Otherwise, randomly selects a neighbor to block u and propose its block assignment. For block (agglomerative) moves,
        avoid proposing the current block."""
    neighbors = np.concatenate((neighbors_out, neighbors_in))
    k_out = sum(neighbors_out[:,1])
    k_in = sum(neighbors_in[:,1])
    k = k_out + k_in
    draws = n_proposals

    # xxx no neighbor available
    if k == 0:
        return r, k_out, k_in, k

    try:
        rand_neighbor = np.random.choice(neighbors[:,0], p=neighbors[:,1]/float(k), size=draws)
    except:
        print("An exception occured:")
        print("k = %s" % str(k))
        print("neighbors[:,0] = %s" % str(neighbors[:,0]))
        print("neighbors[:,1] = %s" % str(neighbors[:,1]))

    u = b[rand_neighbor]
    probs = (np.random.uniform(size=draws) <= B/(d[u].astype(float)+B))

    if agg_move:
        # force proposals to be different from current block via a random offset and modulo
        s1 = (r + 1 + np.random.randint(B - 1, size=draws)) % B
    else:
        s1 = np.random.randint(B, size=draws)

    # proposals by random draw from neighbors of block partition[rand_neighbor]
    multinomial_prob = (M[u, :].T + M[:, u]) / d[u].astype(float)
    if agg_move: # force proposal to be different from current block
        multinomial_prob[r] = 0

    if multinomial_prob.sum() == 0:
        # the current block has no neighbors. randomly propose a different block
        s2 = (r + 1 + np.random.randint(B - 1, size=draws)) % B
    else:
        multinomial_prob /= multinomial_prob.sum()
        # numpy random.multinomial does not support multi-dimensional draws
        c = multinomial_prob.cumsum(axis=0)
        u = np.random.uniform(size=draws)
        s2 = (np.argmax((u < c), axis=0))

    #s = (probs & s1) | (~probs & s2)
    s = np.zeros(draws, dtype=int)
    s[np.where(probs)] = s1
    s[np.where(~probs)] = s2
    # print(r, probs, s1, s2, s)
    return s, k_out, k_in, k

    # propose a new block randomly
    if np.random.uniform(size=draws) <= B/float(d[u]+B):  # chance inversely prop. to block_degree
        if agg_move:
            # force proposal to be different from current block via a random offset and modulo
            s = (r + 1 + np.random.randint(B - 1)) % B
            # s = np.array([s]) # keep while r is scalar
        else:
            s = np.array([np.random.randint(B)])
    else:  # propose by random draw from neighbors of block partition[rand_neighbor]
        if use_sparse:
            multinomial_prob = (M[u, :].toarray().transpose() + M[:, u].toarray()) / float(d[u])
        else:
            multinomial_prob = (M[u, :].transpose() + M[:, u]) / float(d[u])
        if agg_move:  # force proposal to be different from current block
            multinomial_prob[r] = 0
            if multinomial_prob.sum() == 0:  # the current block has no neighbors. randomly propose a different block
                s = (r + 1 + np.random.randint(B - 1)) % B
                # s = np.array([s]) # keep while r is scalar
                return s, k_out, k_in, k
            else:
                multinomial_prob = multinomial_prob / multinomial_prob.sum()
        candidates = multinomial_prob.nonzero()[0]
        s = candidates[np.flatnonzero(np.random.multinomial(1, multinomial_prob[candidates].ravel()))[0]]
        s = np.array([s])
    return s, k_out, k_in, k


def compute_new_rows_cols_interblock_edge_count_matrix(M, r, s, b_out, count_out, b_in, count_in, count_self,
                                                       agg_move, use_sparse, debug=0):
    """Compute the two new rows and cols of the edge count matrix under the proposal for the current node or block

        Parameters
        ----------
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        r : int
                    current block assignment for the node under consideration
        s : int
                    proposed block assignment for the node under consideration
        b_out : ndarray (int)
                    blocks of the out neighbors (of a vertex)
        count_out : ndarray (int)
                    edge counts to the out neighbor blocks
        b_in : ndarray (int)
                    blocks of the in neighbors (of a vertex)
        count_in : ndarray (int)
                    edge counts to the in neighbor blocks
        count_self : int
                    edge counts to self
        agg_move : bool
                    whether the proposal is a block move
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        M_r_row : ndarray or sparse matrix (int)
                    the current block row of the new edge count matrix under proposal
        M_s_row : ndarray or sparse matrix (int)
                    the proposed block row of the new edge count matrix under proposal
        M_r_col : ndarray or sparse matrix (int)
                    the current block col of the new edge count matrix under proposal
        M_s_col : ndarray or sparse matrix (int)
                    the proposed block col of the new edge count matrix under proposal

        Notes
        -----
        The updates only involve changing the entries to and from the neighboring blocks"""

    if 0:
        print("sum(count_in), sum(count_out)", np.sum(count_in), np.sum(count_out))

    B = M.shape[0]
    if agg_move:  # the r row and column are simply empty after this merge move
        M_r_row = np.zeros((1, B), dtype=int)
        M_r_col = np.zeros((B, 1), dtype=int)
    else:
        M_r_row = M[r, :].copy().reshape(1, B)
        M_r_col = M[:, r].copy().reshape(B, 1)

        M_r_row[0, b_out] -= count_out
        where_b_in_r = np.where(b_in == r)
        
        M_r_row[0, r] -= np.sum(count_in[where_b_in_r])
        M_r_row[0, s] += np.sum(count_in[where_b_in_r])

        M_r_col[b_in, 0] -= count_in.reshape(M_r_col[b_in, 0].shape)

        where_b_out_r = np.where(b_out == r)
        M_r_col[r, 0] -= np.sum(count_out[where_b_out_r])
        M_r_col[s, 0] += np.sum(count_out[where_b_out_r])

    M_s_row = M[s, :].copy()
    M_s_col = M[:, s].copy()

    M_s_row[:, b_out] += count_out

    for i in range(len(s)):
        where_b_in_s = np.where(b_in == s[i])
        M_s_row[i, r]  -= np.sum(count_in[where_b_in_s])
        M_s_row[i, s[i]] += np.sum(count_in[where_b_in_s])

    M_s_row[:, r] -= count_self
    for i in range(len(s)):
        M_s_row[i, s[i]] += count_self

    if 1:
        # repeat count_in for each col
        c_in = np.broadcast_to(count_in.T, M_s_col[b_in, :].T.shape).T
        M_s_col[b_in, :] += c_in
    else:
        M_s_col[b_in, :] += count_in.reshape(M_s_col[b_in, :].shape)

    for i in range(len(s)):
        where_b_out_s = np.where(b_out == s[i])
        M_s_col[r, i]  -= np.sum(count_out[where_b_out_s])
        M_s_col[s[i], i] += np.sum(count_out[where_b_out_s])

    M_s_col[r, :] -= count_self

    for i in range(len(s)):
        M_s_col[s[i], i] += count_self

    return M_r_row, M_s_row, M_r_col, M_s_col


def compute_new_block_degrees(r, s, d_out, d_in, d, k_out, k_in, k):
    """Compute the new block degrees under the proposal for the current node or block

        Parameters
        ----------
        r : int
                    current block assignment for the node under consideration
        s : int
                    proposed block assignment for the node under consideration
        d_out : ndarray (int)
                    the current out degree of each block
        d_in : ndarray (int)
                    the current in degree of each block
        d : ndarray (int)
                    the current total degree of each block
        k_out : int
                    the out degree of the node
        k_in : int
                    the in degree of the node
        k : int
                    the total degree of the node

        Returns
        -------
        d_out_new : ndarray (int)
                    the new out degree of each block under proposal
        d_in_new : ndarray (int)
                    the new in degree of each block under proposal
        d_new : ndarray (int)
                    the new total degree of each block under proposal

        Notes
        -----
        The updates only involve changing the degrees of the current and proposed block"""

    if 0:
        new = []
        for old, degree in zip([d_out, d_in, d], [k_out, k_in, k]):
            new_d = old.copy()
            new_d[r] -= degree
            new_d[s] += degree
            new.append(new_d)
            #print([k_out, k_in, k]) # depends only on r
            #print([x[0:10] for x in [d_out, d_in, d]])
            #print([x[0:10] for x in new])
        return new
    else:
        d_outs = np.broadcast_to(d_out, (s.shape[0], d_out.shape[0])).copy()
        d_ins  = np.broadcast_to(d_in,  (s.shape[0],  d_in.shape[0])).copy()
        ds     = np.broadcast_to(d,     (s.shape[0],     d.shape[0])).copy()

        for i,S in enumerate(s):
            d_outs[i, r] -= k_out
            d_outs[i, S] += k_out
            d_ins[i, r] -= k_in
            d_ins[i, S] += k_in
            ds[i, r] -= k
            ds[i, S] += k

        if s.shape[0] == 1: #xxx
            d_outs = d_outs.ravel()
            d_ins = d_ins.ravel()
            ds = ds.ravel()
        return (d_outs, d_ins, ds)



def compute_Hastings_correction(b_out, count_out, b_in, count_in, s, M, M_r_row, M_r_col, B, d, d_new, use_sparse):
    """Compute the Hastings correction for the proposed block from the current block

        Parameters
        ----------
        b_out : ndarray (int)
                    blocks of the out neighbors
        count_out : ndarray (int)
                    edge counts to the out neighbor blocks
        b_in : ndarray (int)
                    blocks of the in neighbors
        count_in : ndarray (int)
                    edge counts to the in neighbor blocks
        s : int
                    proposed block assignment for the node under consideration
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        M_r_row : ndarray or sparse matrix (int)
                    the current block row of the new edge count matrix under proposal
        M_r_col : ndarray or sparse matrix (int)
                    the current block col of the new edge count matrix under proposal
        B : int
                    total number of blocks
        d : ndarray (int)
                    total number of edges to and from each block
        d_new : ndarray (int)
                    new block degrees under the proposal
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        Hastings_correction : float
                    term that corrects for the transition asymmetry between the current block and the proposed block

        Notes
        -----
        - p_{i, s \rightarrow r} : for node i, probability of proposing block r if its current block is s
        - p_{i, r \rightarrow s} : for node i, probability of proposing block s if its current block is r
        - r : current block for node i
        - s : proposed block for node i
        - M^-: current edge count matrix between the blocks
        - M^+: new edge count matrix under the proposal
        - d^-_t: current degree of block t
        - d^+_t: new degree of block t under the proposal
        - \mathbf{b}_{\mathcal{N}_i}: the neighboring blocks to node i
        - k_i: the degree of node i
        - k_{i,t} : the degree of node i to block t (i.e. number of edges to and from block t)
        - B : the number of blocks

        The Hastings correction is:

        \frac{p_{i, s \rightarrow r}}{p_{i, r \rightarrow s}}

        where

        p_{i, r \rightarrow s} = \sum_{t \in \{\mathbf{b}_{\mathcal{N}_i}^-\}} \left[ {\frac{k_{i,t}}{k_i} \frac{M_{ts}^- + M_{st}^- + 1}{d^-_t+B}}\right]

        p_{i, s \rightarrow r} = \sum_{t \in \{\mathbf{b}_{\mathcal{N}_i}^-\}} \left[ {\frac{k_{i,t}}{k_i} \frac{M_{tr}^+ + M_{rt}^+ +1}{d_t^++B}}\right]

        summed over all the neighboring blocks t"""

    t, idx = np.unique(np.append(b_out, b_in), return_inverse=True)  # find all the neighboring blocks
    count = np.bincount(idx, weights=np.append(count_out, count_in)).astype(int)  # count edges to neighboring blocks
    if use_sparse:
        M_t_s = M[t, s].toarray().ravel()
        M_s_t = M[s, t].toarray().ravel()
        M_r_row = M_r_row[t].toarray().ravel()
        M_r_col = M_r_col[t].toarray().ravel()
    else:
        M_t_s = M[t, s]
        M_s_t = M[s, t]
        M_r_row = M_r_row[0, t].ravel()
        M_r_col = M_r_col[t, 0].ravel()
        
    p_forward = np.sum(count*(M_t_s + M_s_t + 1) / (d[t] + float(B)))
    p_backward = np.sum(count*(M_r_row + M_r_col + 1) / (d_new[t] + float(B)))
    return p_backward / p_forward

def carry_out_best_merges(delta_entropy_for_each_block, best_merges, best_merge_for_each_block, b, B, B_to_merge, verbose=False):
    """Execute the best merge (agglomerative) moves to reduce a set number of blocks

        Parameters
        ----------
        delta_entropy_for_each_block : ndarray (float)
                    the delta entropy for merging each block
        best_merge_for_each_block : ndarray (int)
                    the best block to merge with for each block
        b : ndarray (int)
                    array of block assignment for each node
        B : int
                    total number of blocks in the current partition
        B_to_merge : int
                    the number of blocks to merge

        Returns
        -------
        b : ndarray (int)
                    array of new block assignment for each node after the merge
        B : int
                    total number of blocks after the merge"""

    block_map = np.arange(B)
    num_merge = 0
    counter = 0
    while num_merge < B_to_merge:
        mergeFrom = best_merges[counter]
        mergeTo = block_map[best_merge_for_each_block[best_merges[counter]]]
        counter += 1
        if mergeTo != mergeFrom:
            if verbose:
                print("Merge from block %s to block %s" % (mergeFrom, mergeTo))
            block_map[np.where(block_map == mergeFrom)] = mergeTo
            b[np.where(b == mergeFrom)] = mergeTo
            num_merge += 1
    remaining_blocks = np.unique(b)
    mapping = -np.ones(B, dtype=int)
    mapping[remaining_blocks] = np.arange(len(remaining_blocks))
    b = mapping[b]
    B -= B_to_merge
    return b, B


def update_partition(b, ni, r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out_new, d_in_new, d_new, use_sparse):
    """Move the current node to the proposed block and update the edge counts

        Parameters
        ----------
        b : ndarray (int)
                    current array of new block assignment for each node
        ni : int
                    current node index
        r : int
                    current block assignment for the node under consideration
        s : int
                    proposed block assignment for the node under consideration
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        M_r_row : ndarray or sparse matrix (int)
                    the current block row of the new edge count matrix under proposal
        M_s_row : ndarray or sparse matrix (int)
                    the proposed block row of the new edge count matrix under proposal
        M_r_col : ndarray or sparse matrix (int)
                    the current block col of the new edge count matrix under proposal
        M_s_col : ndarray or sparse matrix (int)
                    the proposed block col of the new edge count matrix under proposal
        d_out_new : ndarray (int)
                    the new out degree of each block under proposal
        d_in_new : ndarray (int)
                    the new in degree of each block under proposal
        d_new : ndarray (int)
                    the new total degree of each block under proposal
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        b : ndarray (int)
                    array of block assignment for each node after the move
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks after the move
        d_out_new : ndarray (int)
                    the out degree of each block after the move
        d_in_new : ndarray (int)
                    the in degree of each block after the move
        d_new : ndarray (int)
                    the total degree of each block after the move"""
    b[ni] = s
    M[r, :] = M_r_row
    M[s, :] = M_s_row
    if use_sparse:
        M[:, r] = M_r_col
        M[:, s] = M_s_col
    else:
        M[:, r] = M_r_col.reshape(M[:, r].shape)
        M[:, s] = M_s_col.reshape(M[:, s].shape)

    return b, M, d_out_new, d_in_new, d_new


def compute_overall_entropy(M, d_out, d_in, B, N, E, use_sparse):
    """Compute the overall entropy, including the model entropy as well as the data entropy, on the current partition.
       The best partition with an optimal number of blocks will minimize this entropy.

        Parameters
        ----------
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        d_out : ndarray (int)
                    the current out degrees of each block
        d_in : ndarray (int)
                    the current in degrees of each block
        B : int
                    the number of blocks in the partition
        N : int
                    number of nodes in the graph
        E : int
                    number of edges in the graph
        use_sparse : bool
                    whether the edge count matrix is stored as a sparse matrix

        Returns
        -------
        S : float
                    the overall entropy of the current partition

        Notes
        -----
        - M: current edge count matrix
        - d_{t, out}: current out degree of block t
        - d_{t, in}: current in degree of block t
        - B: number of blocks
        - C: some constant invariant to the partition
        
        The overall entropy of the partition is computed as:
        
        S = E\;h\left(\frac{B^2}{E}\right) + N \ln(B) - \sum_{t_1, t_2} {M_{t_1 t_2} \ln\left(\frac{M_{t_1 t_2}}{d_{t_1, out} d_{t_2, in}}\right)} + C
        
        where the function h(x)=(1+x)\ln(1+x) - x\ln(x) and the sum runs over all entries (t_1, t_2) in the edge count matrix"""

    nonzeros = M.nonzero()  # all non-zero entries
    edge_count_entries = M[nonzeros[0], nonzeros[1]]
    if use_sparse:
        edge_count_entries = edge_count_entries.toarray()

    entries = edge_count_entries * np.log(edge_count_entries / (d_out[nonzeros[0]] * d_in[nonzeros[1]]).astype(float))
    data_S = -np.sum(entries)
    model_S_term = B**2 / float(E)
    model_S = E * (1 + model_S_term) * np.log(1 + model_S_term) - model_S_term * np.log(model_S_term) + N*np.log(B)
    S = model_S + data_S
    return S


def prepare_for_partition_on_next_num_blocks(S, b, M, d, d_out, d_in, B, old_b, old_M, old_d, old_d_out, old_d_in,
                                             old_S, old_B, B_rate):
    """Checks to see whether the current partition has the optimal number of blocks. If not, the next number of blocks
       to try is determined and the intermediate variables prepared.

        Parameters
        ----------
        S : float
                the overall entropy of the current partition
        b : ndarray (int)
                    current array of block assignment for each node
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        d : ndarray (int)
                    the current total degree of each block
        d_out : ndarray (int)
                    the current out degree of each block
        d_in : ndarray (int)
                    the current in degree of each block
        B : int
                    the number of blocks in the current partition
        old_b : list of length 3
                    holds the best three partitions so far
        old_M : list of length 3
                    holds the edge count matrices for the best three partitions so far
        old_d : list of length 3
                    holds the block degrees for the best three partitions so far
        old_d_out : list of length 3
                    holds the out block degrees for the best three partitions so far
        old_d_in : list of length 3
                    holds the in block degrees for the best three partitions so far
        old_S : list of length 3
                    holds the overall entropy for the best three partitions so far
        old_B : list of length 3
                    holds the number of blocks for the best three partitions so far
        B_rate : float
                    the ratio on the number of blocks to reduce before the golden ratio bracket is established

        Returns
        -------
        b : ndarray (int)
                starting array of block assignment on each node for the next number of blocks to try
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    starting edge count matrix for the next number of blocks to try
        d : ndarray (int)
                    the starting total degree of each block for the next number of blocks to try
        d_out : ndarray (int)
                    the starting out degree of each block for the next number of blocks to try
        d_in : ndarray (int)
                    the starting in degree of each block for the next number of blocks to try
        B : int
                    the starting number of blocks before the next block merge
        B_to_merge : int
                    number of blocks to merge next
        old_b : list of length 3
                    holds the best three partitions including the current partition
        old_M : list of length 3
                    holds the edge count matrices for the best three partitions including the current partition
        old_d : list of length 3
                    holds the block degrees for the best three partitions including the current partition
        old_d_out : list of length 3
                    holds the out block degrees for the best three partitions including the current partition
        old_d_in : list of length 3
                    holds the in block degrees for the best three partitions including the current partition
        old_S : list of length 3
                    holds the overall entropy for the best three partitions including the current partition
        old_B : list of length 3
                    holds the number of blocks for the best three partitions including the current partition
        optimal_B_found : bool
                    flag for whether the optimal block has been found

        Notes
        -----
        The holders for the best three partitions so far and their statistics will be stored in the order of the number
        of blocks, starting from the highest to the lowest. The middle entry is always the best so far. The number of
        blocks is reduced by a fixed rate until the golden ratio bracket (three best partitions with the middle one
        being the best) is established. Once the golden ratio bracket is established, perform golden ratio search until
        the bracket is narrowed to consecutive number of blocks where the middle one is identified as the optimal
        number of blocks."""

    optimal_B_found = False
    B_to_merge = 0

    # update the best three partitions so far and their statistics
    if S <= old_S[1]:  # if the current partition is the best so far
        # if the current number of blocks is smaller than the previous best number of blocks
        old_index = 0 if old_B[1] > B else 2
        old_b[old_index] = old_b[1]
        old_M[old_index] = old_M[1]
        old_d[old_index] = old_d[1]
        old_d_out[old_index] = old_d_out[1]
        old_d_in[old_index] = old_d_in[1]
        old_S[old_index] = old_S[1]
        old_B[old_index] = old_B[1]

        index = 1
    else:  # the current partition is not the best so far
        # if the current number of blocks is smaller than the best number of blocks so far
        index = 2 if old_B[1] > B else 0

    old_b[index] = b
    old_M[index] = M
    old_d[index] = d
    old_d_out[index] = d_out
    old_d_in[index] = d_in
    old_S[index] = S
    old_B[index] = B

    # find the next number of blocks to try using golden ratio bisection
    if old_S[2] == np.Inf:  # if the three points in the golden ratio bracket has not yet been established
        B_to_merge = int(B*B_rate)
        if (B_to_merge==0): # not enough number of blocks to merge so done
            optimal_B_found = True
        b = old_b[1].copy()
        M = old_M[1].copy()
        d = old_d[1].copy()
        d_out = old_d_out[1].copy()
        d_in = old_d_in[1].copy()
    else:  # golden ratio search bracket established
        if old_B[0] - old_B[2] == 2:  # we have found the partition with the optimal number of blocks
            optimal_B_found = True
            B = old_B[1]
            b = old_b[1]
        else:  # not done yet, find the next number of block to try according to the golden ratio search
            if (old_B[0]-old_B[1]) >= (old_B[1]-old_B[2]):  # the higher segment in the bracket is bigger
                index = 0
            else:  # the lower segment in the bracket is bigger
                index = 1
            next_B_to_try = old_B[index + 1] + np.round((old_B[index] - old_B[index + 1]) * 0.618).astype(int)
            B_to_merge = old_B[index] - next_B_to_try
            B = old_B[index]
            b = old_b[index].copy()
            M = old_M[index].copy()
            d = old_d[index].copy()
            d_out = old_d_out[index].copy()
            d_in = old_d_in[index].copy()
    return b, M, d, d_out, d_in, B, B_to_merge, old_b, old_M, old_d, old_d_out, old_d_in, old_S, old_B, optimal_B_found


def plot_graph_with_partition(out_neighbors, b, graph_object=None, pos=None):
    """Plot the graph with force directed layout and color/shape each node according to its block assignment

        Parameters
        ----------
        out_neighbors : list of ndarray; list length is N, the number of nodes
                    each element of the list is a ndarray of out neighbors, where the first column is the node indices
                    and the second column the corresponding edge weights
        b : ndarray (int)
                    array of block assignment for each node
        graph_object : graph tool object, optional
                    if a graph object already exists, use it to plot the graph
        pos : ndarray (float) shape = (#nodes, 2), optional
                    if node positions are given, plot the graph using them

        Returns
        -------
        graph_object : graph tool object
                    the graph tool object containing the graph and the node position info"""

    if len(out_neighbors) <= 5000:
        if graph_object is None:
            graph_object = gt.Graph()
            edge_list = [(i, j) for i in range(len(out_neighbors)) if len(out_neighbors[i]) > 0 for j in
                         out_neighbors[i][:, 0]]
            graph_object.add_edge_list(edge_list)
            if pos is None:
                graph_object.vp['pos'] = gt.sfdp_layout(graph_object)
            else:
                graph_object.vp['pos'] = graph_object.new_vertex_property("vector<float>")
                for v in graph_object.vertices():
                    graph_object.vp['pos'][v] = pos[graph_object.vertex_index[v], :]
        block_membership = graph_object.new_vertex_property("int")
        vertex_shape = graph_object.new_vertex_property("int")
        block_membership.a = b[0:len(out_neighbors)]
        vertex_shape.a = np.mod(block_membership.a, 10)
        gt.graph_draw(graph_object, inline=True, output_size=(400, 400), pos=graph_object.vp['pos'],
                      vertex_shape=vertex_shape,
                      vertex_fill_color=block_membership, edge_pen_width=0.1, edge_marker_size=1, vertex_size=7)
    else:
        print('That\'s a big graph!')
    return graph_object


def evaluate_partition(true_b, alg_b):
    """Evaluate the output partition against the truth partition and report the correctness metrics.
       Compare the partitions using only the nodes that have known truth block assignment.

        Parameters
        ----------
        true_b : ndarray (int)
                array of truth block assignment for each node. If the truth block is not known for a node, -1 is used
                to indicate unknown blocks.
        alg_b : ndarray (int)
                array of output block assignment for each node. The length of this array corresponds to the number of
                nodes observed and processed so far."""

    blocks_b1 = true_b
    blocks_b1_set = set(true_b)
    blocks_b1_set.discard(-1)  # -1 is the label for 'unknown'
    B_b1 = len(blocks_b1_set)

    blocks_b2 = alg_b
    B_b2 = max(blocks_b2) + 1

    print('\nPartition Correctness Evaluation\n')
    print('Number of nodes: {}'.format(len(alg_b)))
    print('Number of partitions in truth partition: {}'.format(B_b1))
    print('Number of partitions in alg. partition: {}'.format(B_b2))

    # populate the confusion matrix between the two partitions
    contingency_table = np.zeros((B_b1, B_b2))
    for i in range(len(alg_b)):  # evaluation based on nodes observed so far
        if true_b[i] != -1:  # do not include nodes without truth in the evaluation
            contingency_table[blocks_b1[i], blocks_b2[i]] += 1
    N = contingency_table.sum()

    # associate the labels between two partitions using linear assignment
    assignment = Munkres()  # use the Hungarian algorithm / Kuhn-Munkres algorithm
    if B_b1 > B_b2:  # transpose matrix for linear assignment (this implementation assumes #col >= #row)
        contingency_table = contingency_table.transpose()
    indexes = assignment.compute(-contingency_table)
    total = 0
    contingency_table_before_assignment = np.array(contingency_table)
    for row, column in indexes:
        contingency_table[:, row] = contingency_table_before_assignment[:, column]
        total += contingency_table[row, row]
    # fill in the un-associated columns
    unassociated_col = set(range(contingency_table.shape[1])) - set(np.array(indexes)[:, 1])
    counter = 0;
    for column in unassociated_col:
        contingency_table[:, contingency_table.shape[0] + counter] = contingency_table_before_assignment[:, column]
        counter += 1
    if B_b1 > B_b2:  # transpose back
        contingency_table = contingency_table.transpose()
    print('Contingency Table: \n{}'.format(contingency_table))
    joint_prob = contingency_table / sum(
        sum(contingency_table))  # joint probability of the two partitions is just the normalized contingency table
    accuracy = sum(joint_prob.diagonal())
    print('Accuracy (with optimal partition matching): {}'.format(accuracy))
    print('\n')

    # Compute pair-counting-based metrics
    def nchoose2(a):
        return misc.comb(a, 2)

    num_pairs = nchoose2(N)
    colsum = np.sum(contingency_table, axis=0)
    rowsum = np.sum(contingency_table, axis=1)
    # compute counts of agreements and disagreement (4 types) and the regular rand index
    sum_table_squared = sum(sum(contingency_table ** 2))
    sum_colsum_squared = sum(colsum ** 2)
    sum_rowsum_squared = sum(rowsum ** 2)
    count_in_each_b1 = np.sum(contingency_table, axis=1)
    count_in_each_b2 = np.sum(contingency_table, axis=0)
    num_same_in_b1 = sum(count_in_each_b1 * (count_in_each_b1 - 1)) / 2
    num_same_in_b2 = sum(count_in_each_b2 * (count_in_each_b2 - 1)) / 2
    num_agreement_same = 0.5 * sum(sum(contingency_table * (contingency_table - 1)));
    num_agreement_diff = 0.5 * (N ** 2 + sum_table_squared - sum_colsum_squared - sum_rowsum_squared);
    num_agreement = num_agreement_same + num_agreement_diff
    rand_index = num_agreement / num_pairs

    vectorized_nchoose2 = np.vectorize(nchoose2)
    sum_table_choose_2 = sum(sum(vectorized_nchoose2(contingency_table)))
    sum_colsum_choose_2 = sum(vectorized_nchoose2(colsum))
    sum_rowsum_choose_2 = sum(vectorized_nchoose2(rowsum))
    adjusted_rand_index = (sum_table_choose_2 - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs) / (
        0.5 * (sum_rowsum_choose_2 + sum_colsum_choose_2) - sum_rowsum_choose_2 * sum_colsum_choose_2 / num_pairs)
    print('Rand Index: {}'.format(rand_index))
    print('Adjusted Rand Index: {}'.format(adjusted_rand_index))
    print('Pairwise Recall: {}'.format(num_agreement_same / (num_same_in_b1)))
    print('Pairwise Precision: {}'.format(num_agreement_same / (num_same_in_b2)))
    print('\n')

    # compute the information theoretic metrics
    marginal_prob_b2 = np.sum(joint_prob, 0)
    marginal_prob_b1 = np.sum(joint_prob, 1)
    idx1 = np.nonzero(marginal_prob_b1)
    idx2 = np.nonzero(marginal_prob_b2)
    conditional_prob_b2_b1 = np.zeros(joint_prob.shape)
    conditional_prob_b1_b2 = np.zeros(joint_prob.shape)
    conditional_prob_b2_b1[idx1, :] = joint_prob[idx1, :] / marginal_prob_b1[idx1, None]
    conditional_prob_b1_b2[:, idx2] = joint_prob[:, idx2] / marginal_prob_b2[None, idx2]
    # compute entropy of the non-partition2 and the partition2 version
    H_b2 = -np.sum(marginal_prob_b2[idx2] * np.log(marginal_prob_b2[idx2]))
    H_b1 = -np.sum(marginal_prob_b1[idx1] * np.log(marginal_prob_b1[idx1]))

    # compute the conditional entropies
    idx = np.nonzero(joint_prob)
    H_b2_b1 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b2_b1[idx])))
    H_b1_b2 = -np.sum(np.sum(joint_prob[idx] * np.log(conditional_prob_b1_b2[idx])))
    # compute the mutual information (symmetric)
    marginal_prod = np.dot(marginal_prob_b1[:, None], np.transpose(marginal_prob_b2[:, None]))
    MI_b1_b2 = np.sum(np.sum(joint_prob[idx] * np.log(joint_prob[idx] / marginal_prod[idx])))

    if H_b1 > 0:
        fraction_missed_info = H_b1_b2 / H_b1
    else:
        fraction_missed_info = 0
    if H_b2 > 0:
        fraction_err_info = H_b2_b1 / H_b2
    else:
        fraction_err_info = 0
    print('Entropy of truth partition: {}'.format(abs(H_b1)))
    print('Entropy of alg. partition: {}'.format(abs(H_b2)))
    print('Conditional entropy of truth partition given alg. partition: {}'.format(abs(H_b1_b2)))
    print('Conditional entropy of alg. partition given truth partition: {}'.format(abs(H_b2_b1)))
    print('Mututal informationion between truth partition and alg. partition: {}'.format(abs(MI_b1_b2)))
    print('Fraction of missed information: {}'.format(abs(fraction_missed_info)))
    print('Fraction of erroneous information: {}'.format(abs(fraction_err_info)))
