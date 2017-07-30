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
import scipy.sparse
import scipy.misc as misc
from munkres import Munkres # for correctness evaluation
import sys
from multiprocessing import sharedctypes
import ctypes
from compute_delta_entropy import compute_delta_entropy
from collections import defaultdict
from fast_sparse_array import fast_sparse_array, nonzero_slice, take_nonzero, nonzero_dict
from collections import Iterable
use_sparse_dict = 1

use_graph_tool_options = False # for visualiziing graph partitions (optional)
if use_graph_tool_options:
    import graph_tool.all as gt

def assert_close(x, y, tol=1e-9):
    if np.abs(x - y) > tol:
        raise Exception("Equality assertion failed: %s %s" % (x,y))

def coo_to_flat(x, size):
    x_i, x_v = x
    f = np.zeros(size)
    f[x_i] = x_v
    return f

def is_sorted(x):
    return len(x) == 1 or (x[1:] >= x[0:-1]).all()

def is_in_sorted(needle, haystack):
    #assert(is_sorted(haystack))
    if len(haystack) == 0:
        return np.zeros(needle.shape, dtype=bool)
    elif len(needle) == 0:
        return np.array([], dtype=bool)
    loc = np.searchsorted(haystack, needle)
    loc[(loc == len(haystack))] = len(haystack) - 1
    res = (haystack[loc] == needle)
    res = res.reshape(needle.shape)
    return res


search_array = is_in_sorted
#search_array = np.in1d


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

    old_b = [[], [], []]  # partition for the high, best, and low number of blocks so far
    old_M = [[], [], []]  # edge count matrix for the high, best, and low number of blocks so far
    old_d = [[], [], []]  # block degrees for the high, best, and low number of blocks so far
    old_d_out = [[], [], []]  # out block degrees for the high, best, and low number of blocks so far
    old_d_in = [[], [], []]  # in block degrees for the high, best, and low number of blocks so far
    old_S = [np.Inf, np.Inf, np.Inf] # overall entropy for the high, best, and low number of blocks so far
    old_B = [[], [], []]  # number of blocks for the high, best, and low number of blocks so far
    graph_object = None
    hist = (old_b, old_M, old_d, old_d_out, old_d_in, old_S, old_B)
    return hist, graph_object


def initialize_edge_counts(out_neighbors, B, b, sparse):
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

    import time
    print("Initialize edge counts for size %d" % (B))
    t0 = time.time()

    if not sparse:
        M = np.zeros((B,B), dtype=int)
        # compute the initial interblock edge count
        for v in range(len(out_neighbors)):
            k1 = b[v]
            if len(out_neighbors[v]) > 0:
                k2, inverse_idx = np.unique(b[out_neighbors[v][:, 0]], return_inverse=True)
                count = np.bincount(inverse_idx, weights=out_neighbors[v][:, 1]).astype(int)
                M[k1, k2] += count

        # compute initial block degrees
        d_out = np.asarray(M.sum(axis=1)).ravel()
        d_in = np.asarray(M.sum(axis=0)).ravel()
        d = d_out + d_in
        print("density(M) = %s" % (len(M.nonzero()[0]) / (B ** 2.)))
    else:
        M = fast_sparse_array((B,B))
        d_out = np.zeros(B, dtype=int)
        d_in = np.zeros(B, dtype=int)
        M_d = defaultdict(int)
        for v in range(len(out_neighbors)):
            k1 = b[v]
            if len(out_neighbors[v]) > 0:
                k2, inverse_idx = np.unique(b[out_neighbors[v][:, 0]], return_inverse=True)
                count = np.bincount(inverse_idx, weights=out_neighbors[v][:, 1]).astype(int)
                for k,c in zip(k2, count):
                    M_d[(k1, k)] += c
        for (i,j),w in M_d.items():
            M[i,j] = w
            d_in[j] += w
            d_out[i] += w
        d = d_out + d_in

    t1 = time.time()

    print("M initialization took %s" % (t1-t0))

    return M, d_out, d_in, d


def propose_new_partition(r, neighbors, neighbor_weights, n_neighbors, b, M, d, B, agg_move):
    """Propose a new block assignment for the current node or block

        Parameters
        ----------
        r : int
                    current block assignment for the node under consideration
        neighbors : ndarray (int)
                    neighbors of this vertex
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
        Returns
        -------
        s : int
                    proposed block assignment for the node under consideration
        Notes
        -----
        - d_u: degree of block u

        Randomly select a neighbor of the current node, and obtain its block assignment u. With probability \frac{B}{d_u + B}, randomly propose
        a block. Otherwise, randomly selects a neighbor to block u and propose its block assignment. For block (agglomerative) moves,
        avoid proposing the current block."""

    # xxx no neighbor available
    if n_neighbors == 0:
        return r

    rand_neighbor = np.random.choice(neighbors, p=neighbor_weights / float(n_neighbors))

    #rand_neighbor = np.random.choice(neighbors[:,0])

    u = b[rand_neighbor]

    if np.random.uniform() <= B / (d[u].astype(float) + B):
        if agg_move:
            # force proposals to be different from current block via a random offset and modulo
            s1 = (r + 1 + np.random.randint(B - 1)) % B
        else:
            s1 = np.random.randint(B)

        return s1
    else:
        # proposals by random draw from neighbors of block partition[rand_neighbor]
        if 1:
            Mu_row_i, Mu_row = take_nonzero(M, u, 0)
            Mu_col_i, Mu_col = take_nonzero(M, u, 1)
            multinomial_choices = np.concatenate((Mu_row_i, Mu_col_i))
            multinomial_probs = np.concatenate((Mu_row, Mu_col)).astype(float)
            if agg_move: # force proposal to be different from current block
                multinomial_probs[ (multinomial_choices == r) ] = 0.0

            if len(multinomial_choices) == 0:
                # the current block has no neighbors. randomly propose a different block
                s2 = (r + 1 + np.random.randint(B - 1)) % B
            else:
                multinomial_probs /= multinomial_probs.sum()
                s2 = np.random.choice(multinomial_choices, p = multinomial_probs)
                #c = multinomial_probs.cumsum(axis=0)
                #print("constant",(np.abs(multinomial_probs[0] - multinomial_probs) < 1e-6).all())
                #print("p,c=",multinomial_probs,c)
                #u = np.random.uniform()
                #s2i = np.argmax((u < c), axis=0)
                #s2_new = multinomial_choices[s2i]
                #print("s2=",s2)

        if 0:
            multinomial_prob = (M[u, :] + M[:, u]).astype(float) #/ d[u].astype(float)

            if agg_move: # force proposal to be different from current block
                multinomial_prob[r] = 0

            nz = multinomial_prob.nonzero()[0]

            if len(nz) == 0:
                # the current block has no neighbors. randomly propose a different block
                s2 = (r + 1 + np.random.randint(B - 1)) % B
            else:
                multinomial_prob[nz] /= multinomial_prob[nz].sum()
                # s2 = np.random.choice(nz, p = multinomial_prob[nz])

                # numpy random.multinomial does not support multi-dimensional draws
                c = multinomial_prob.cumsum(axis=0)
                #u = np.random.uniform()
                s2 = np.argmax((u < c), axis=0)

                assert(s2 == s2_new)

        return s2


def compute_new_rows_cols_interblock_edge_count_matrix(M, r, s, b_out, count_out, b_in, count_in, count_self,
                                                       agg_move, sparse):

    verify_sparse = 0
    B = M.shape[0]
    if agg_move:  # the r row and column are simply empty after this merge move
        if sparse:
            M_r_row_i, M_r_row_v = (np.empty((0,), dtype=int), np.empty((0,), dtype=int))
            M_r_col_i, M_r_col_v = (np.empty((0,), dtype=int), np.empty((0,), dtype=int))
            new_M_r_row = (M_r_row_i, M_r_row_v)
            new_M_r_col = (M_r_col_i, M_r_col_v)
#            new_M_r_row = {}
#            new_M_r_col = {}
        else:
            M_r_row = np.zeros(B, dtype=int)
            M_r_col = np.zeros(B, dtype=int)
            new_M_r_row = M_r_row
            new_M_r_col = M_r_col
    else:
        where_b_in_r = np.where(b_in == r)
        where_b_out_r = np.where(b_out == r)

        offset = np.sum(count_in[where_b_in_r])

        if not sparse or verify_sparse:
            M_r_row = M[r, :].copy()
            M_r_row[b_out] -= count_out
            M_r_row[r] -= offset
            M_r_row[s] += offset
            new_M_r_row = M_r_row

        if sparse:
            if use_sparse_dict:
                M_r_row_d = M.take_dict(r, 0).copy()
                for k,v in zip(b_out,count_out):
                    M_r_row_d[k] -= v
                M_r_row_d[r] -= offset
                M_r_row_d[s] += offset
                new_M_r_row = M_r_row_d
            else:
                M_r_row_i, M_r_row_v = take_nonzero(M, r, 0)
                M_r_row_in_b_out = search_array(M_r_row_i, b_out)
                M_r_row_v[M_r_row_in_b_out] -= count_out

                x = search_array(M_r_row_i, np.array([r]))
                M_r_row_v[x] -= offset

                x = search_array(M_r_row_i, np.array([s]))
                if x.any():
                    M_r_row_v[x] += offset
                else:
                    M_r_row_v = np.append(M_r_row_v, offset)
                    M_r_row_i = np.append(M_r_row_i, s)

                # After modifying the entries, elements of the rows and columns may have become zero.
                nz = (M_r_row_v != 0)
                M_r_row_i = M_r_row_i[nz]
                M_r_row_v = M_r_row_v[nz]
                new_M_r_row = (M_r_row_i, M_r_row_v)

            if verify_sparse:
                L = np.argsort(M_r_row_i)
                M_r_row_i = M_r_row_i[L]
                M_r_row_v = M_r_row_v[L]
                M_r_row_i_test, M_r_row_v_test = nonzero_slice(M_r_row)

                if (M_r_row_i_test.shape != M_r_row_i.shape) or (M_r_row_i_test != M_r_row_i).any():
                    print("r = %s s = %s" % (r, s))
                    print("indices")
                    print(M_r_row_i)
                    print(M_r_row_i_test)
                    print("values")
                    print(M_r_row_v)
                    print(M_r_row_v_test)
                    print("")
                    raise Exception("Array M_r_row mismatch encountered")

        if not sparse or verify_sparse:
            M_r_col = M[:, r].copy()
            M_r_col[b_in] -= count_in
            M_r_col[r] -= np.sum(count_out[where_b_out_r])
            M_r_col[s] += np.sum(count_out[where_b_out_r])
            new_M_r_col = M_r_col

        if sparse:
            if use_sparse_dict:
                M_r_col_d = M.take_dict(r, 1).copy()
                for k,v in zip(b_in,count_in):
                    M_r_col_d[k] -= v
                M_r_col_d[r] -= np.sum(count_out[where_b_out_r])
                M_r_col_d[s] += np.sum(count_out[where_b_out_r])
                new_M_r_col = M_r_col_d
            else:
                M_r_col_i, M_r_col_v = take_nonzero(M, r, 1)

                M_r_col_in_b_in = search_array(M_r_col_i, b_in)
                b_in_in_M_r_col = search_array(b_in, M_r_col_i)

                M_r_col_v[M_r_col_in_b_in] -= count_in[b_in_in_M_r_col]

                x = search_array(M_r_col_i, np.array([r]))
                M_r_col_v[x] -= np.sum(count_out[where_b_out_r])

                x = search_array(M_r_col_i, np.array([s]))
                if x.any():
                    M_r_col_v[x] += np.sum(count_out[where_b_out_r])
                else:
                    M_r_col_v = np.append(M_r_col_v, np.sum(count_out[where_b_out_r]))
                    M_r_col_i = np.append(M_r_col_i, s)

                # After modifying the entries, elements of the rows and columns may have become zero.
                nz = (M_r_col_v != 0)
                M_r_col_i = M_r_col_i[nz]
                M_r_col_v = M_r_col_v[nz]
                new_M_r_col = (M_r_col_i, M_r_col_v)

            if verify_sparse:
                L = np.argsort(M_r_col_i)
                M_r_col_i = M_r_col_i[L]
                M_r_col_v = M_r_col_v[L]
                M_r_col_i_test, M_r_col_v_test = nonzero_slice(M_r_col)

                if (M_r_col_i_test.shape != M_r_col_i.shape) or (M_r_col_i_test != M_r_col_i).any():
                    print("r = %s s = %s" % (r, s))
                    print("indices")
                    print(M_r_col_i)
                    print(M_r_col_i_test)
                    print("values")
                    print(M_r_col_v)
                    print(M_r_col_v_test)
                    print("")
                    raise Exception("Array M_r_col mismatch encountered")

    where_b_in_s = np.where(b_in == s)
    where_b_out_s = np.where(b_out == s)

    # Compute M_s_row
    offset = np.sum(count_in[where_b_in_s]) + count_self
    if not sparse or verify_sparse:
        M_s_row = M[s, :].copy()
        M_s_row[b_out] += count_out
        M_s_row[r] -= offset
        M_s_row[s] += offset
        new_M_s_row = M_s_row

    if sparse:
        if use_sparse_dict:
            M_s_row_d = M.take_dict(s, 0).copy()
            for k,v in zip(b_out,count_out):
                M_s_row_d[k] += v
            M_s_row_d[r] -= offset
            M_s_row_d[s] += offset
            new_M_s_row = M_s_row_d
        else:
            M_s_row_i, M_s_row_v = take_nonzero(M, s, 0)
            M_s_row_in_b_out = search_array(M_s_row_i, b_out)
            b_out_in_M_s_row = search_array(b_out, M_s_row_i)

            # xxx insert all the missing entries
            if M_s_row_in_b_out.any():
                M_s_row_v[M_s_row_in_b_out] += count_out[b_out_in_M_s_row]

            M_s_row_i = np.append(M_s_row_i, b_out[~b_out_in_M_s_row])
            M_s_row_v = np.append(M_s_row_v, count_out[~b_out_in_M_s_row])

            x = search_array(M_s_row_i, np.array([r]))
            M_s_row_v[x] -= offset

            x = search_array(M_s_row_i, np.array([s]))

            if x.any():
                M_s_row_v[x] += offset
            else:
                M_s_row_i = np.append(M_s_row_i, s)
                M_s_row_v = np.append(M_s_row_v, offset)

            # After modifying the entries, elements of the rows and columns may have become zero.
            nz = (M_s_row_v != 0)
            M_s_row_i = M_s_row_i[nz]
            M_s_row_v = M_s_row_v[nz]
            new_M_s_row = (M_s_row_i, M_s_row_v)

        if verify_sparse:
            L = np.argsort(M_s_row_i)
            M_s_row_i = M_s_row_i[L]
            M_s_row_v = M_s_row_v[L]
            M_s_row_i_test, M_s_row_v_test = nonzero_slice(M_s_row)

            if (M_s_row_i_test.shape != M_s_row_i.shape) or (M_s_row_i_test != M_s_row_i).any():
                print("s = %s adj=%s" % (s, offset))
                print("count_out.shape", count_out.shape)
                print("b_out.shape", b_out.shape)

                print("M_s_row_in_b_out",M_s_row_in_b_out.astype(int))
                print(M_s_row_i)

                print("Indices (actual vs. expected):")
                print(M_s_row_i)
                print(M_s_row_i_test)

                print("Values (actual vs. expected):")
                print(M_s_row_v)
                print(M_s_row_v_test)
                print("")

                raise Exception("Array M_s_row mismatch encountered")
                assert((M_s_row_v_test == M_s_row_v).all())

    # Compute M_s_col
    offset = np.sum(count_out[where_b_out_s]) + count_self

    if not sparse or verify_sparse:
        M_s_col = M[:, s].copy()
        M_s_col[b_in] += count_in
        M_s_col[r] -= offset
        M_s_col[s] += offset
        new_M_s_col = M_s_col

    if sparse:
        if use_sparse_dict:
            M_s_col_d = M.take_dict(s, 1).copy()
            for k,v in zip(b_in,count_in):
                M_s_col_d[k] += v
            M_s_col_d[r] -= offset
            M_s_col_d[s] += offset
            new_M_s_col = M_s_col_d
        else:
            M_s_col_i, M_s_col_v = take_nonzero(M, s, 1)
            M_s_col_in_b_in = search_array(M_s_col_i, b_in)
            b_in_in_M_s_col = search_array(b_in, M_s_col_i)

            # xxx insert all the missing entries
            if M_s_col_in_b_in.any():
                M_s_col_v[M_s_col_in_b_in] += count_in[b_in_in_M_s_col]

            M_s_col_i = np.append(M_s_col_i, b_in[~b_in_in_M_s_col])
            M_s_col_v = np.append(M_s_col_v, count_in[~b_in_in_M_s_col])

            x = search_array(M_s_col_i, np.array([r]))
            M_s_col_v[x] -= offset

            x = search_array(M_s_col_i, np.array([s]))
            if x.any():
                M_s_col_v[x] += offset
            else:
                M_s_col_i = np.append(M_s_col_i, s)
                M_s_col_v = np.append(M_s_col_v, offset)

            # After modifying the entries, elements of the rows and columns may have become zero.
            nz = (M_s_col_v != 0)
            M_s_col_i = M_s_col_i[nz]
            M_s_col_v = M_s_col_v[nz]
            new_M_s_col = (M_s_col_i, M_s_col_v)

        if verify_sparse:
            L = np.argsort(M_s_col_i)
            M_s_col_i = M_s_col_i[L]
            M_s_col_v = M_s_col_v[L]
            M_s_col_i_test, M_s_col_v_test = nonzero_slice(M_s_col)

            if (M_s_col_i_test.shape != M_s_col_i.shape) or (M_s_col_i_test != M_s_col_i).any():
                print("r = %s s = %s adj=%s" % (r, s, offset))
                print("count_in.shape", count_in.shape)
                print("b_in.shape", b_in.shape)
                print("b_in=",b_in)
                print("")
                print("M_s_col_in_b_in", M_s_col_in_b_in.astype(int))
                print(M_s_col_i)
                print("")
                print("Indices (actual vs. expected):")
                print(M_s_col_i)
                print(M_s_col_i_test)
                print("")
                print("Values (actual vs. expected):")
                print(M_s_col_v)
                print(M_s_col_v_test)
                print("")

                raise Exception("Array M_s_col mismatch encountered")

    return new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col


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

    if 1:
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


def compute_Hastings_correction(b_out, count_out, b_in, count_in, r, s, M, M_r_row, M_r_col, B, d, d_new):
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
    B = float(B)

    if 0:
        M_t_s = M[t, s]
        M_s_t = M[s, t]
    elif use_sparse_dict:
        # t appears to already be sorted, verified with assert(is_sorted(t))
        # t = np.sort(t)
        M_s_row_d = M.take_dict(s, 0)
        M_s_col_d = M.take_dict(s, 1)
        M_t_s = np.fromiter((M_s_row_d[i] for i in t), dtype=int)
        M_s_t = np.fromiter((M_s_col_d[i] for i in t), dtype=int)
    else:
        # t = np.sort(t)
        M_s_row_i, M_s_row_v = take_nonzero(M, s, 0)
        M_s_col_i, M_s_col_v = take_nonzero(M, s, 1)
        M_t_s = coo_to_flat((M_s_col_i, M_s_col_v), M.shape[0])[t]
        M_s_t = coo_to_flat((M_s_row_i, M_s_row_v), M.shape[0])[t]

    p_forward = np.sum(count * (M_t_s + M_s_t + 1) / (d[t] + B))

    p_backward = 0.0

    if type(M_r_row) is nonzero_dict:
        M_r_row_i = np.fromiter(M_r_row.keys(), dtype=int)
        M_r_row_v = np.fromiter((M_r_row[k] for k in M_r_row_i), dtype=int)
    elif type(M_r_row) is tuple:
        M_r_row_i, M_r_row_v = M_r_row
    else:
        M_r_row_i, M_r_row_v = nonzero_slice(M_r_row)

    if type(M_r_col) is nonzero_dict:
        M_r_col_i = np.fromiter(M_r_col.keys(), dtype=int)
        M_r_col_v = np.fromiter((M_r_col[k] for k in M_r_col_i), dtype=int)
    elif type(M_r_col) is tuple:
        M_r_col_i, M_r_col_v = M_r_col
    else:
        M_r_col_i, M_r_col_v = nonzero_slice(M_r_col)


    if type(M_r_row) is tuple:
        in_t = search_array(M_r_row_i, t)
        in_M_r_row = search_array(t, M_r_row_i)
        p_backward += np.sum(count[in_M_r_row] * M_r_row_v[in_t] / (d_new[t[in_M_r_row]] + B))
    elif type(M_r_row) is nonzero_dict:
        in_t = search_array(M_r_row_i, t)
        in_M_r_row = np.fromiter((i in M_r_row for i in t), dtype=bool)
        p_backward += np.sum(count[in_M_r_row] * M_r_row_v[in_t] / (d_new[t[in_M_r_row]] + B))
    else:
        p_backward += np.sum(count * M_r_row[t] / (d_new[t] + B))

    if type(M_r_col) is tuple:
        in_t = search_array(M_r_col_i, t)
        in_M_r_col = search_array(t, M_r_col_i)
        p_backward += np.sum(count[in_M_r_col] * (M_r_col_v[in_t] + 1) / (d_new[t[in_M_r_col]] + B))
    elif type(M_r_col) is nonzero_dict:
        in_t = search_array(M_r_col_i, t)
        in_M_r_col = np.fromiter((i in M_r_col for i in t), dtype=bool)
        p_backward += np.sum(count[in_M_r_col] * (M_r_col_v[in_t] + 1) / (d_new[t[in_M_r_col]] + B)) 
    else:
        p_backward += np.sum(count * (M_r_col[t] + 1) / (d_new[t] + B))

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
        if counter == len(best_merges):
            if verbose:
                print("No more merges possible")
            break

        mergeFrom = best_merges[counter]
        mergeTo = block_map[best_merge_for_each_block[best_merges[counter]]]
        counter += 1
        if mergeTo != mergeFrom:
            if verbose:
                print("Merge %d of %d from block %s to block %s" % (num_merge, B_to_merge, mergeFrom, mergeTo))
            block_map[np.where(block_map == mergeFrom)] = mergeTo
            b[np.where(b == mergeFrom)] = mergeTo
            num_merge += 1

    remaining_blocks = np.unique(b)
    mapping = -np.ones(B, dtype=int)
    mapping[remaining_blocks] = np.arange(len(remaining_blocks))
    b = mapping[b]
    B -= num_merge
    return b, B


def update_partition(b, ni, r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out_new, d_in_new, d_new):
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
    M[:, r] = M_r_col.reshape(M[:, r].shape)
    M[:, s] = M_s_col.reshape(M[:, s].shape)

    return b, M, d_out_new, d_in_new, d_new


def compute_overall_entropy(M, d_out, d_in, B, N, E):
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

    if type(M) is not fast_sparse_array:
        nonzeros = M.nonzero()
        edge_count_entries = M[nonzeros]
        entries = edge_count_entries * np.log(edge_count_entries / (d_out[nonzeros[0]] * d_in[nonzeros[1]]).astype(float))
        data_S = -np.sum(entries)
    else:
        data_S = 0.0
        for i in range(B):
            M_row_i, M_row_v = take_nonzero(M, i, 0, sort=False)
            entries = M_row_v * np.log(M_row_v / (d_out[i] * d_in[M_row_i]).astype(float))
            data_S += -np.sum(entries)

    model_S_term = B**2 / float(E)
    model_S = E * (1 + model_S_term) * np.log(1 + model_S_term) - model_S_term * np.log(model_S_term) + N*np.log(B)
    S = model_S + data_S

    return S


def prepare_for_partition_on_next_num_blocks(S, b, M, d, d_out, d_in, B, hist, B_rate):
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

    old_b, old_M, old_d, old_d_out, old_d_in, old_S, old_B = hist
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
            next_B_to_try = old_B[index + 1] + np.round((old_B[index] - old_B[index + 1]) * 0.61803399).astype(int)
            B_to_merge = old_B[index] - next_B_to_try
            B = old_B[index]
            b = old_b[index].copy()
            M = old_M[index].copy()
            d = old_d[index].copy()
            d_out = old_d_out[index].copy()
            d_in = old_d_in[index].copy()

    hist = old_b, old_M, old_d, old_d_out, old_d_in, old_S, old_B
    return b, M, d, d_out, d_in, B, B_to_merge, hist, optimal_B_found


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
