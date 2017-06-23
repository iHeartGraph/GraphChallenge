from partition_baseline_support import *
import multiprocessing as mp
from multiprocessing import Pool, Value, current_process
from functools import reduce

use_timeit = True # for timing runs (optional)
if use_timeit:
    import timeit
import os, sys, argparse
import time
import traceback
import numpy.random
from compute_delta_entropy import compute_delta_entropy

parser = argparse.ArgumentParser()
parser.add_argument("-P", "--parallel-phase", type=int, required=False, default=3)
parser.add_argument("-t", "--threads", type=int, required=False, default=0)
parser.add_argument("-p", "--parts", type=int, required=False, default=0)
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument("-b", "--node-move-batch-size", type=int, required=False, default=1)
parser.add_argument("-s", "--sort", type=int, required=False, default=0)
parser.add_argument("-S", "--seed", type=int, required=False, default=-1)
parser.add_argument("input_filename", nargs="?", type=str, default="../../data/static/simulated_blockmodel_graph_500_nodes")
args = parser.parse_args()

import random
def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def entropy_max_argsort(x):
    a = np.argsort(x)
    y = x[a]
    m = np.zeros(y.shape, dtype=bool)

    # Positions of transition starts
    m[0]  = 1
    m[1:] = y[1:] != y[0:-1]
    starts = np.where(m)[0]

    # Positions of transition ends
    ends = np.empty(starts.shape, dtype=int)
    ends[0:-1] = starts[1:]
    ends[-1] = y.shape[0]

    # Run the counters
    ii = starts.copy()
    out = np.empty(x.shape, dtype=int)
    k = 0

    while k < x.shape[0]:
        for idx,elm in enumerate(starts):
            if ii[idx] < ends[idx]:
                out[k] = a[ii[idx]]
                ii[idx] += 1
                k += 1
    return out


def compute_best_merge_and_entropy_wrapper(tup):
    (blocks, num_blocks) = tup

    interblock_edge_count = syms['interblock_edge_count']
    block_partition = syms['block_partition']
    block_degrees = syms['block_degrees']
    block_degrees_out = syms['block_degrees_out']
    block_degrees_in = syms['block_degrees_in']
    partition = syms['partition']

    return compute_best_merge_and_entropy(blocks, num_blocks, interblock_edge_count, block_partition, block_degrees, args.n_proposal, block_degrees_out, block_degrees_in)

def compute_best_merge_and_entropy(blocks, num_blocks, M, block_partition, block_degrees, n_proposal, block_degrees_out, block_degrees_in):
    best_overall_merge = [-1 for i in blocks]
    best_overall_delta_entropy = [np.Inf for i in blocks]
    n_proposals_evaluated = 0

    for current_block_idx,current_block in enumerate(blocks):
        if current_block is None:
            break

        current_block = np.array([current_block])
        ii = M[:, current_block].nonzero()
        oo = M[current_block, :].nonzero()
        in_blocks = np.vstack((ii[0], M[ii[0], current_block])).T
        out_blocks = np.vstack((oo[1], M[current_block, oo[1]])).T

        for proposal_idx in range(10):

            # propose a new block to merge with
            proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
                current_block,
                out_blocks, in_blocks, block_partition, M, block_degrees, num_blocks,
                1, use_sparse = 0,
                n_proposals = n_proposal)

            # compute the two new rows and columns of the interblock edge count matrix
            new_M_current_block_row, new_M_new_block_row, new_M_current_block_col, new_M_new_block_col = \
                                                                                                                                                                                         compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, proposal,
                                                                                                                                                                                                                                            out_blocks[:, 0], out_blocks[:, 1], in_blocks[:, 0],
                                                                                                                                                                                                                                            in_blocks[:, 1],
                                                                                                                                                                                                                                            M[current_block, current_block],
                                                                                                                                                                                                                                            agg_move = 1, use_sparse = 0, debug = args.debug)
            n_proposals_evaluated += len(proposal)
            # compute change in entropy / posterior

            delta_entropy = np.empty(n_proposal)
            for pi,pp in enumerate(proposal):
                s = np.array([pp])

                # compute the two new rows and columns of the interblock edge count matrix
                new_M_current_block_row_2, new_M_new_block_row_2, new_M_current_block_col_2, new_M_new_block_col_2 = \
                            compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, s,
                                                                               out_blocks[:, 0], out_blocks[:, 1], in_blocks[:, 0],
                                                                               in_blocks[:, 1],
                                                                               M[current_block, current_block],
                                                                               agg_move = 1, use_sparse = 0, debug=args.debug)

                # compute new block degrees
                block_degrees_out_new_2, block_degrees_in_new_2, block_degrees_new_2 = compute_new_block_degrees(current_block,
                                                                                                                 s,
                                                                                                                 block_degrees_out,
                                                                                                                 block_degrees_in,
                                                                                                                 block_degrees,
                                                                                                                 num_out_neighbor_edges,
                                                                                                                 num_in_neighbor_edges,
                                                                                                                 num_neighbor_edges)
                delta_entropy[pi] = compute_delta_entropy(current_block, s, M,
                                                          new_M_current_block_row_2,
                                                          new_M_new_block_row_2,
                                                          new_M_current_block_col_2,
                                                          new_M_new_block_col_2,
                                                          block_degrees_out,
                                                          block_degrees_in,
                                                          block_degrees_out_new_2,
                                                          block_degrees_in_new_2)

            mi = np.argmin(delta_entropy)
            best_entropy = delta_entropy[mi]

            if best_entropy < best_overall_delta_entropy[current_block_idx]:
                best_overall_merge[current_block_idx] = proposal[mi]
                best_overall_delta_entropy[current_block_idx] = best_entropy

    return blocks, best_overall_merge, best_overall_delta_entropy, n_proposals_evaluated

update_id = -1
def propose_node_movement_wrapper(tup):
    global update_id, partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in

    current_node = tup

    state = syms['state']
    lock = syms['lock']

    pid = current_process().pid

    (update_id_shared, partition_shared, interblock_edge_count_shared, num_blocks, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared, where_modified_shared) = state

    is_non_seq = 0

    lock.acquire()

    if update_id != update_id_shared - 1:
        is_non_seq = 1

    if update_id != update_id_shared:
        if update_id == -1:
            (interblock_edge_count, partition, block_degrees, block_degrees_out, block_degrees_in) \
                = (i.copy() for i in (interblock_edge_count_shared, partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared))
        elif is_non_seq:
            interblock_edge_count[:] = interblock_edge_count_shared[:]
            block_degrees_in[:] = block_degrees_in_shared[:]
            block_degrees_out[:] = block_degrees_out_shared[:]
            block_degrees[:] = block_degrees_shared[:]
            partition[:] = partition_shared[:]            
        else:
            w = np.where(where_modified_shared)
            interblock_edge_count[w, :] = interblock_edge_count_shared[w, :]
            interblock_edge_count[:, w] = interblock_edge_count_shared[:, w]
            block_degrees_in[w] = block_degrees_in_shared[w]
            block_degrees_out[w] = block_degrees_out_shared[w]
            block_degrees[w] = block_degrees_shared[w]
            partition[:] = partition_shared[:]

        update_id = update_id_shared.copy()

    lock.release()

    # Ensure every worker has a different random seed.
    numpy.random.seed((pid + int(time.time() * 1e6)) % 4294967295)

    t0 = time.time()

    res = propose_node_movement(current_node, partition, out_neighbors, in_neighbors,
                                interblock_edge_count, num_blocks, block_degrees, block_degrees_out, block_degrees_in)

    t1 = time.time()
    t_elapsed_ms = (t1 - t0) * 1e3

    return res + (t_elapsed_ms, not is_non_seq,is_non_seq)

def propose_node_movement(current_node, partition, out_neighbors, in_neighbors, interblock_edge_count, num_blocks, block_degrees, block_degrees_out, block_degrees_in):
    #child_id = current_process().pid
    #sys.stdout.write("Enter propose_node_movement %s\n" % child_id)
    #time.sleep(child_id % 10)

    current_block = np.array([partition[current_node]])
    # propose a new block for this node
    proposal, num_out_neighbor_edges, num_in_neighbor_edges, num_neighbor_edges = propose_new_partition(
        current_block,
        out_neighbors[current_node], in_neighbors[current_node], partition,
        interblock_edge_count, block_degrees, num_blocks, 0, use_sparse = 0, n_proposals=1)

    # determine whether to accept or reject the proposal
    if (proposal == current_block):
        accepted = 0
        delta_entropy = 0
        return current_node, current_block, int(proposal), delta_entropy, accepted
    else:
        # compute block counts of in and out neighbors
        blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[current_node][:, 0]],
                                                return_inverse=True)

        count_out = np.bincount(inverse_idx_out, weights=out_neighbors[current_node][:, 1]).astype(int)
        blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[current_node][:, 0]], return_inverse=True)
        count_in = np.bincount(inverse_idx_in, weights=in_neighbors[current_node][:, 1]).astype(int)

        if 0:
            print("current_node %s proposal %s count_out %s" % (current_node, proposal, count_out))

        # compute the two new rows and columns of the interblock edge count matrix
        self_edge_weight = np.sum(out_neighbors[current_node][np.where(
            out_neighbors[current_node][:, 0] == current_node), 1])  # check if this node has a self edge

        new_interblock_edge_count_current_block_row, new_interblock_edge_count_new_block_row, new_interblock_edge_count_current_block_col, new_interblock_edge_count_new_block_col = \
            compute_new_rows_cols_interblock_edge_count_matrix(interblock_edge_count, current_block, proposal,
                                                               blocks_out, count_out, blocks_in, count_in,
                                                               self_edge_weight, 0, use_sparse = 0)

        # compute new block degrees
        block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(
            current_block, proposal, block_degrees_out, block_degrees_in, block_degrees, num_out_neighbor_edges,
            num_in_neighbor_edges, num_neighbor_edges)

        # compute the Hastings correction
        Hastings_correction = compute_Hastings_correction(blocks_out, count_out, blocks_in, count_in, proposal,
                                                          interblock_edge_count,
                                                          new_interblock_edge_count_current_block_row,
                                                          new_interblock_edge_count_current_block_col,
                                                          num_blocks, block_degrees,
                                                          block_degrees_new, use_sparse = 0)

        # compute change in entropy / posterior
        delta_entropy = compute_delta_entropy(current_block, proposal, interblock_edge_count,
                                              new_interblock_edge_count_current_block_row,
                                              new_interblock_edge_count_new_block_row,
                                              new_interblock_edge_count_current_block_col,
                                              new_interblock_edge_count_new_block_col, block_degrees_out,
                                              block_degrees_in, block_degrees_out_new, block_degrees_in_new)
        # compute probability of acceptance
        p_accept = np.min([np.exp(-args.beta * delta_entropy) * Hastings_correction, 1])

        #sys.stdout.write("exit propose_node_movement %s\n" % child_id)

    return current_node, current_block, int(proposal), delta_entropy, p_accept



def update_partition_single(b, ni, s, M, M_r_row, M_s_row, M_r_col, M_s_col):
    r = b[ni]
    b[ni] = s
    M[r, :] = M_r_row
    M[s, :] = M_s_row
    M[:, r] = M_r_col.reshape(M[:, r].shape)
    M[:, s] = M_s_col.reshape(M[:, s].shape)
    return b, M

def update_partition_batch(b, ni, S, mask, M, M_r_rows, M_s_rows, M_r_cols, M_s_cols):
    """
    Move the current nodes to the proposed blocks and update the edge counts
    Parameters
        b : ndarray (int)
                    current array of new block assignment for each node
        ni : ndarray (int)
                    index for each node
        S : ndarray (int)
                    proposed block assignment for each node
        mask:
                    which nodes are under consideration
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
    """

    # Idea: apply cumulative changes for M_r_row, M_s_row, M_r_col, M_s_col

    dM = np.zeros(M.shape, dtype=int)
    ii = np.where(mask != 0)[0]
    shape = M[0, :].shape
    for i in ii:
        r = b[ni[i]]
        s = S[i]
        b[ni[i]] = s
        if 0:
            print("i = %s ni[i] = %s r = %s s = %s mask.shape = %s M.shape = %s" % (i, ni[i], r, s, mask.shape, M.shape))
        dM[r, :] += (M_r_rows[i] - M[r, :]).reshape(shape)
        dM[s, :] += (M_s_rows[i] - M[s, :]).reshape(shape)
        dM[:, s] += (M_s_cols[i].reshape(M[:, s].shape) - M[:, s])
        dM[:, r] += (M_r_cols[i].reshape(M[:, r].shape) - M[:, r])

        # Avoid double counting
        dM[r, r] -= (M_r_rows[i][0, r] - M[r, r])
        dM[r, s] -= (M_r_rows[i][0, s] - M[r, s])
        dM[s, r] -= (M_s_rows[i][0, r] - M[s, r])
        dM[s, s] -= (M_s_rows[i][0, s] - M[s, s])

    M += dM
    d_out = np.asarray(M.sum(axis=1)).ravel()
    d_in = np.asarray(M.sum(axis=0)).ravel()
    d = d_out + d_in
    return b, M, d_out, d_in, d

def shared_memory_copy(z):
    prod = reduce((lambda x,y : x*y), (i for i in z.shape))
    ctype = {"float64" : ctypes.c_double, "int64" : ctypes.c_int64, "bool" : ctypes.c_bool}[str(z.dtype)]
    raw = sharedctypes.RawArray(ctype, prod)
    a = np.frombuffer(raw, dtype=z.dtype).reshape(z.shape)
    a[:] = z
    return a


def nodal_moves_sequential(batch_size, max_num_nodal_itr, delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy_cur, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks):
    global block_sum_time_cum

    total_num_nodal_moves_itr = 0
    itr_delta_entropy = np.zeros(max_num_nodal_itr)

    for itr in range(max_num_nodal_itr):
        num_nodal_moves = 0
        block_sum_time = 0.0
        itr_delta_entropy[itr] = 0

        propose_time_ms_cum = 0
        propose_time_workers_ms_cum = 0.0
        merge_time_ms_cum = 0
        update_partition_time_ms_cum = 0.0

        if args.sort:
            #L = np.argsort(partition)
            L = entropy_max_argsort(partition)
        else:
            L = range(0, N)
            
        t_propose_start = time.time()

        movements = (propose_node_movement(i, partition, out_neighbors, in_neighbors, M, num_blocks, block_degrees, block_degrees_out, block_degrees_in) for i in L)

        t_propose_end = time.time()
        propose_time_ms = (t_propose_end - t_propose_start) * 1e3
        propose_time_ms_cum += propose_time_ms

        t_merge_start = time.time()

        proposal_cnt = 0
        next_batch_cnt = num_nodal_moves + batch_size

        modified = np.zeros(M.shape[0], dtype=bool)
        cnt_seq_workers = 0
        cnt_non_seq_workers = 0
        update_id_cnt = 0

        for result in movements:
            (ni, current_block, proposal, delta_entropy, p_accept) = result
            accept = (np.random.uniform() <= p_accept)
            proposal_cnt += 1

            if not accept:
                continue

            t_update_partition_beg = time.time()

            total_num_nodal_moves_itr += 1
            num_nodal_moves += 1
            itr_delta_entropy[itr] += delta_entropy

            modified[partition[ni]] = True
            modified[proposal] = True

            current_block = np.array([partition[ni]])

            if 0:
                print("Move %s from block %s to block %s." % (ni, current_block, proposal))

            blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[ni][:, 0]], return_inverse=True)
            count_out = np.bincount(inverse_idx_out, weights=out_neighbors[ni][:, 1]).astype(int)
            blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[ni][:, 0]], return_inverse=True)
            count_in = np.bincount(inverse_idx_in, weights=in_neighbors[ni][:, 1]).astype(int)
            self_edge_weight = np.sum(out_neighbors[ni][np.where(out_neighbors[ni][:, 0] == ni), 1])

           
            (new_M_r_row, new_M_s_row,new_M_r_block_col, new_M_s_col) = \
                                compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, [proposal],
                                                                                   blocks_out, count_out, blocks_in, count_in,
                                                                                   self_edge_weight, agg_move = 0, use_sparse = 0)

            partition, M = update_partition_single(partition, ni, proposal, M,
                                                   new_M_r_row, new_M_s_row, new_M_r_block_col, new_M_s_col)

            t_update_partition_end = time.time()

            update_partition_time_ms_cum += (t_update_partition_end - t_update_partition_beg) * 1e3

            btime = time.time()
            where_modified = np.where(modified)
            block_degrees_out[where_modified] = np.sum(M[modified, :], axis = 1)
            block_degrees_in[where_modified] = np.sum(M[:, modified], axis = 0)
            block_degrees[where_modified] = block_degrees_out[where_modified] + block_degrees_in[where_modified]

            btime_end = time.time()
            block_sum_time += btime_end - btime
            block_sum_time_cum += btime_end - btime

        t_merge_end = time.time()
        merge_time_ms = (t_merge_end - t_merge_start) * 1e3
        merge_time_ms_cum += merge_time_ms

        if num_nodal_moves != 0:
            merge_rate_ms = merge_time_ms / num_nodal_moves
        else:
            merge_rate_ms = 0.0

        if args.verbose:
            print("Processed %d nodal movements in %3.4f ms rate = %f per ms." % (num_nodal_moves, merge_time_ms, merge_rate_ms))

            print("Node propose time is %3.2f ms, merge time is %3.2f ms, block sum time is %3.2f ms, partition update time is %3.2f ms, ratio propose to merge %3.2f"
              % (propose_time_ms_cum, merge_time_ms_cum, block_sum_time * 1e3, update_partition_time_ms_cum,  (propose_time_ms_cum) / merge_time_ms_cum))
            print("Itr: {:3d}, number of nodal moves: {:3d}, delta S: {:0.9f}".format(itr, num_nodal_moves,
                                                                                itr_delta_entropy[itr] / float(
                                                                                    overall_entropy_cur)))

        # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
        if itr >= (delta_entropy_moving_avg_window - 1):  
            if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold * overall_entropy_cur)):
                    break

    return total_num_nodal_moves_itr


def nodal_moves_parallel(n_thread, batch_size, max_num_nodal_itr, delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy_cur, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks):
    global syms, block_sum_time_cum

    total_num_nodal_moves_itr = 0
    itr_delta_entropy = np.zeros(max_num_nodal_itr)

    syms = {}
    lock = mp.Lock()

    parallel_phase2 = (args.parallel_phase & 2) != 0

    for itr in range(max_num_nodal_itr):
        num_nodal_moves = 0
        block_sum_time = 0.0
        itr_delta_entropy[itr] = 0

        propose_time_ms_cum = 0
        propose_time_workers_ms_cum = 0.0
        merge_time_ms_cum = 0
        update_partition_time_ms_cum = 0.0
        t_useless = 0.0
        update_shared_time = 0.0

        if parallel_phase2 and n_thread > 0:
            where_modified_shared = shared_memory_copy(np.zeros(M.shape[0], dtype=bool))
            (update_id_shared, M_shared, partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared) \
                = (shared_memory_copy(i) for i in (np.array([0]), M, partition, block_degrees, block_degrees_out, block_degrees_in))

        if args.sort:
            #L = np.argsort(partition)
            L = entropy_max_argsort(partition)
        else:
            L = range(0, N)
            
        t_propose_start = time.time()

        if parallel_phase2 and n_thread > 0:
            state = (update_id_shared, partition_shared, M_shared, num_blocks, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared, where_modified_shared)

            syms['lock'] = lock
            syms['state'] = state

            pool_size = min(n_thread, len(L))
            pool = Pool(n_thread)

            #print("About to fork %d workers for batch %s." % (pool_size,L))

            movements = pool.imap_unordered(propose_node_movement_wrapper, L)
        else:
            movements = (propose_node_movement(i, partition, out_neighbors, in_neighbors, M, num_blocks, block_degrees, block_degrees_out, block_degrees_in) for i in L)

        t_propose_end = time.time()
        propose_time_ms = (t_propose_end - t_propose_start) * 1e3
        propose_time_ms_cum += propose_time_ms

        t_merge_start = time.time()

        proposal_cnt = 0
        next_batch_cnt = num_nodal_moves + batch_size
        modified = np.zeros(M.shape[0], dtype=bool)
        cnt_seq_workers = 0
        cnt_non_seq_workers = 0
        update_id_cnt = 0

        for result in movements:
            useless_time_beg = time.time()

            (ni, current_block, proposal, delta_entropy, p_accept, propose_time_worker_ms, is_seq, is_non_seq) = result

            cnt_seq_workers += is_seq
            cnt_non_seq_workers += is_non_seq

            #print("Got a result for index %d from pid %d" % (ni,pid))

            accept = (np.random.uniform() <= p_accept)
            proposal_cnt += 1
            propose_time_workers_ms_cum += propose_time_worker_ms

            useless_time_end = time.time()
            t_useless += useless_time_end - useless_time_beg

            if not accept:
                continue

            t_update_partition_beg = time.time()

            total_num_nodal_moves_itr += 1
            num_nodal_moves += 1
            itr_delta_entropy[itr] += delta_entropy

            modified[partition[ni]] = True
            modified[proposal] = True

            current_block = np.array([partition[ni]])

            if 0:
                print("Move %s from block %s to block %s." % (ni, current_block, proposal))

            blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[ni][:, 0]], return_inverse=True)
            count_out = np.bincount(inverse_idx_out, weights=out_neighbors[ni][:, 1]).astype(int)
            blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[ni][:, 0]], return_inverse=True)
            count_in = np.bincount(inverse_idx_in, weights=in_neighbors[ni][:, 1]).astype(int)
            self_edge_weight = np.sum(out_neighbors[ni][np.where(out_neighbors[ni][:, 0] == ni), 1])

           
            (new_M_r_row, new_M_s_row,new_M_r_block_col, new_M_s_col) = \
                                compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, [proposal],
                                                                                   blocks_out, count_out, blocks_in, count_in,
                                                                                   self_edge_weight, agg_move = 0, use_sparse = 0)

            partition, M = update_partition_single(partition, ni, proposal, M,
                                                   new_M_r_row, new_M_s_row, new_M_r_block_col, new_M_s_col)

            t_update_partition_end = time.time()

            update_partition_time_ms_cum += (t_update_partition_end - t_update_partition_beg) * 1e3

            if parallel_phase2 and n_thread > 0 and num_nodal_moves >= next_batch_cnt:
                btime = time.time()
                where_modified = np.where(modified)
                block_degrees_out[where_modified] = np.sum(M[modified, :], axis = 1)
                block_degrees_in[where_modified] = np.sum(M[:, modified], axis = 0)
                block_degrees[where_modified] = block_degrees_out[where_modified] + block_degrees_in[where_modified]
                next_batch_cnt = num_nodal_moves + batch_size

                btime_end = time.time()
                block_sum_time += btime_end - btime
                block_sum_time_cum += btime_end - btime

                update_beg = time.time()
                update_id_cnt += 1

                lock.acquire()

                M_shared[where_modified, :] = M[where_modified, :]
                M_shared[:, where_modified] = M[:, where_modified]
                block_degrees_in_shared[where_modified] = block_degrees_in[where_modified]
                block_degrees_out_shared[where_modified] = block_degrees_out[where_modified]
                block_degrees_shared[where_modified] = block_degrees[where_modified]

                where_modified_shared[:] = modified[:]
                partition_shared[:] = partition[:]
                update_id_shared[:] = np.array([update_id_cnt])[:]

                lock.release()

                modified = np.zeros(M.shape[0], dtype=bool)

                update_end = time.time()
                update_shared_time += update_end - update_beg


        btime = time.time()
        where_modified = np.where(modified)
        block_degrees_out[where_modified] = np.sum(M[modified, :], axis = 1)
        block_degrees_in[where_modified] = np.sum(M[:, modified], axis = 0)
        block_degrees[where_modified] = block_degrees_out[where_modified] + block_degrees_in[where_modified]

        btime_end = time.time()
        block_sum_time += btime_end - btime
        block_sum_time_cum += btime_end - btime

        update_beg = time.time()

        if parallel_phase2 and n_thread > 0:
            pool.close()
            pool.join()

        update_end = time.time()
        update_shared_time += update_end - update_beg

        t_merge_end = time.time()
        merge_time_ms = (t_merge_end - t_merge_start) * 1e3
        merge_time_ms_cum += merge_time_ms

        if num_nodal_moves != 0:
            merge_rate_ms = merge_time_ms / num_nodal_moves
        else:
            merge_rate_ms = 0.0

        if args.verbose:
            print("Processed %d nodal movements in %3.4f ms rate = %f per ms." % (num_nodal_moves, merge_time_ms, merge_rate_ms))

            print("Node propose time is %3.2f ms (master) propose time %3.2f ms (workers) merge time is %3.2f ms block sum time is %3.2f ms partition update time is %3.2f useless time is %3.2f update_shared time is %3.2f ratio propose to merge %3.2f"
              % (propose_time_ms_cum, propose_time_workers_ms_cum, merge_time_ms_cum, block_sum_time * 1e3, update_partition_time_ms_cum, t_useless * 1e3, update_shared_time * 1e3, (propose_time_ms_cum + propose_time_workers_ms_cum) / merge_time_ms_cum))
            print("Worker update counts sequential %d non-sequential %d" % (cnt_seq_workers, cnt_non_seq_workers))
            print("Itr: {:3d}, number of nodal moves: {:3d}, delta S: {:0.9f}".format(itr, num_nodal_moves,
                                                                                itr_delta_entropy[itr] / float(
                                                                                    overall_entropy_cur)))

        # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
        if itr >= (delta_entropy_moving_avg_window - 1):  
            if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold * overall_entropy_cur)):
                    break

    return total_num_nodal_moves_itr


def entropy_for_block_count(num_blocks, num_target_blocks, delta_entropy_threshold, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, partition):
    global syms, block_sum_time_cum

    # Dramatis Personae
    M = interblock_edge_count

    t_start = time.time()
    parallel_phase1 = (args.parallel_phase & 1) != 0

    n_thread = args.threads

    n_merges = 0
    n_proposals_evaluated = 0

    # begin agglomerative partition updates (i.e. block merging)
    if args.verbose:
        print("\nMerging down blocks from {} to {}".format(num_blocks, num_target_blocks))

    best_merge_for_each_block = np.ones(num_blocks, dtype=int) * -1  # initialize to no merge
    delta_entropy_for_each_block = np.ones(num_blocks) * np.Inf  # initialize criterion
    block_partition = np.arange(num_blocks)
    n_merges += 1

    if parallel_phase1 and n_thread > 0:
        syms = {}
        syms['interblock_edge_count'] = M
        syms['block_partition'] = block_partition
        syms['block_degrees'] = block_degrees
        syms['block_degrees_out'] = block_degrees_out
        syms['block_degrees_in'] = block_degrees_in
        syms['partition'] = partition

        L = range(num_blocks)
        pool_size = min(n_thread, num_blocks)

        with Pool(processes=pool_size) as pool:
            for current_blocks,best_merge,best_delta_entropy,fresh_proposals_evaluated in pool.imap_unordered(compute_best_merge_and_entropy_wrapper, [((i,),num_blocks) for i in L]):
                for current_block_idx,current_block in enumerate(current_blocks):
                    best_merge_for_each_block[current_block] = best_merge[current_block_idx]
                    delta_entropy_for_each_block[current_block] = best_delta_entropy[current_block_idx]
                n_proposals_evaluated += fresh_proposals_evaluated                

    else:
        current_blocks,best_merge,best_delta_entropy,fresh_proposals_evaluated = compute_best_merge_and_entropy(range(num_blocks), num_blocks, M, block_partition, block_degrees, args.n_proposal, block_degrees_out, block_degrees_in)
        n_proposals_evaluated += fresh_proposals_evaluated
        for current_block_idx,current_block in enumerate(current_blocks):
            if current_block is not None:
                best_merge_for_each_block[current_block] = best_merge[current_block_idx]
                delta_entropy_for_each_block[current_block] = best_delta_entropy[current_block_idx]

    if (n_proposals_evaluated == 0):
        raise Exception("No proposals evaluated.")

    overall_entropy_per_num_blocks = np.empty(len(num_target_blocks))
    state_per_num_blocks = [None] * len(num_target_blocks)

    best_overall_entropy = np.Inf
    best_merges = delta_entropy_for_each_block.argsort()

    for i,t in enumerate(num_target_blocks):
        num_blocks_to_merge = num_blocks - t

        # carry out the best merges
        (partition_t, num_blocks_t) = carry_out_best_merges(delta_entropy_for_each_block, best_merges, best_merge_for_each_block, partition,
                                                            num_blocks, num_blocks_to_merge)

        # re-initialize edge counts and block degrees
        M_t, block_degrees_out_t, block_degrees_in_t, block_degrees_t = initialize_edge_counts(out_neighbors,
                                                                                               num_blocks_t,
                                                                                               partition_t,
                                                                                               use_sparse = 0)
        # compute the global entropy for MCMC convergence criterion
        overall_entropy = compute_overall_entropy(M_t, block_degrees_out_t, block_degrees_in_t, num_blocks_t, N,
                                                  E, use_sparse = 0)

        overall_entropy_per_num_blocks[i] = overall_entropy
        state_per_num_blocks[i] = (overall_entropy, partition_t, num_blocks_t, M_t, block_degrees_out_t, block_degrees_in_t, block_degrees_t)

    S = overall_entropy_per_num_blocks[::-1]
    dS_dn = 0.5 * (S[2] - S[1]) + 0.5 * (S[1] - S[0])

    optimal_stop_found = False
    if len(num_target_blocks) == 3 \
       and overall_entropy_per_num_blocks[1] < overall_entropy_per_num_blocks[0] \
       and overall_entropy_per_num_blocks[1] < overall_entropy_per_num_blocks[2]:
        print("Optimal stopping criterion found at %d blocks derivative %s at time %s." % (num_target_blocks[1], dS_dn, time.time() - t_prog_start))
        optimal_stop_found = True
        best_idx = 1
    else:
        extrapolated_newton = num_target_blocks[1] - 0.5 * (S[2] - S[0]) / (S[2] - S[1] - (S[1] - S[0]))
        print("Stopping criterion not found at %d blocks extrapolate to %d blocks derivative %s." % (num_target_blocks[1], extrapolated_newton, dS_dn))
        best_idx = np.argsort(overall_entropy)
        best_idx = 1 # xxx

    (overall_entropy, partition, num_blocks_t, M, block_degrees_out, block_degrees_in, block_degrees) = state_per_num_blocks[best_idx]

    num_blocks_merged = num_blocks - num_blocks_t
    num_blocks = num_blocks_t

    print("Best num_blocks = %s" % num_blocks)
    print("blocks %s entropy %s" % (num_target_blocks, overall_entropy_per_num_blocks))

    if args.verbose:
        print("Beginning nodal updates")

    batch_size = args.node_move_batch_size

    parallel_phase2 = (args.parallel_phase & 2) != 0

    if parallel_phase2 and n_thread > 0:
        total_num_nodal_moves_itr = nodal_moves_parallel(n_thread, batch_size, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks)
    else:
        total_num_nodal_moves_itr = nodal_moves_sequential(batch_size, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks)

    # compute the global entropy for determining the optimal number of blocks
    overall_entropy = compute_overall_entropy(M, block_degrees_out, block_degrees_in, num_blocks, N,
                                              E, use_sparse = 0)

    if args.verbose:
        t_end = time.time()
        time_taken = t_end - t_start
        print(
            "Total number of nodal moves: {:3d}, overall_entropy: {:0.2f}".format(total_num_nodal_moves_itr, overall_entropy))
        print("Time taken: {:3.4f}".format(time_taken))

    if args.visualize_graph:
        graph_object = plot_graph_with_partition(out_neighbors, partition, graph_object)

    return overall_entropy, n_proposals_evaluated, n_merges, total_num_nodal_moves_itr, M, block_degrees, block_degrees_out, block_degrees_in, num_blocks_merged, partition, optimal_stop_found, dS_dn


def do_main(args):
    global N, E, out_neighbors, in_neighbors, true_partition, t_prog_start

    input_filename = args.input_filename
    true_partition_available = True
    args.visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions

    np.set_printoptions(linewidth=159)

    if not os.path.isfile(input_filename + '.tsv') and not os.path.isfile(input_filename + '_1.tsv'):
            print("File doesn't exist: '{}'!".format(input_filename))
            sys.exit(1)

    if args.parts >= 1:
            print('\nLoading partition 1 of {} ({}) ...'.format(args.parts, input_filename + "_1.tsv"))
            out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available, strm_piece_num=1)
            for part in xrange(2, args.parts + 1):
                    print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
                    out_neighbors, in_neighbors, N, E = load_graph(input_filename, load_true_partition=False, strm_piece_num=part, out_neighbors=out_neighbors, in_neighbors=in_neighbors)
    else:
            out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available)


    if args.verbose:
        print('Number of nodes: {}'.format(N))
        print('Number of edges: {}'.format(E))
        print('Nodal updates based on actual moves.')

    if use_timeit:
        t_prog_start = time.time()

    # initialize by putting each node in its own block (N blocks)
    num_blocks = N
    partition = np.arange(num_blocks, dtype=int)

    # partition update parameters
    args.beta = 3  # exploitation versus exploration (higher value favors exploitation)
    use_sparse_matrix = False  # whether to represent the edge count matrix using sparse matrix
                               # Scipy's sparse matrix is slow but this may be necessary for large graphs

    # agglomerative partition update parameters
    num_agg_proposals_per_block = 10  # number of proposals per block
    num_block_reduction_rate = 0.5  # fraction of blocks to reduce until the golden ratio bracket is established

    # nodal partition updates parameters
    args.max_num_nodal_itr = 100  # maximum number of iterations
    delta_entropy_threshold1 = 5e-4  # stop iterating when the change in entropy falls below this fraction of the overall entropy
                                     # lowering this threshold results in more nodal update iterations and likely better performance, but longer runtime
    delta_entropy_threshold2 = 1e-4  # threshold after the golden ratio bracket is established (typically lower to fine-tune to partition)
    args.delta_entropy_moving_avg_window = 3  # width of the moving average window for the delta entropy convergence criterion

    # initialize edge counts and block degrees
    interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees = initialize_edge_counts(out_neighbors,
                                                                                                       num_blocks,
                                                                                                       partition,
                                                                                                       use_sparse = 0)

    # initialize items before iterations to find the partition with the optimal number of blocks
    optimal_num_blocks_found, old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks, graph_object = initialize_partition_variables()
    num_blocks_to_merge = int(num_blocks * num_block_reduction_rate)

    # begin partitioning by finding the best partition with the optimal number of blocks
    n_proposals_evaluated = 0
    total_num_nodal_moves = 0
    args.debug = 0
    args.n_proposal = 1

    while not optimal_num_blocks_found:
        if not np.all(np.isfinite(old_overall_entropy)):
            # golden ratio bracket not yet established
            delta_entropy_threshold = delta_entropy_threshold1
        else:
            delta_entropy_threshold = delta_entropy_threshold2


        # Must be in decreasing order because reducing by carrying out merges modifies state.
        target_blocks = [num_blocks - num_blocks_to_merge + 1, num_blocks - num_blocks_to_merge, num_blocks - num_blocks_to_merge - 1]

        (overall_entropy,n_proposals_itr,n_merges_itr,total_num_nodal_moves_itr, \
         interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks_merged, partition, optimal_num_blocks_found, dS_dn) \
         = entropy_for_block_count(num_blocks, target_blocks,
                                   delta_entropy_threshold,
                                   interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in,
                                   partition)

        print("num_blocks = %s num_blocks_merged = %s" % (num_blocks, num_blocks_merged))
        num_blocks -= num_blocks_merged
        total_num_nodal_moves += total_num_nodal_moves_itr
        n_proposals_evaluated += n_proposals_itr

        # check whether the partition with optimal number of block has been found; if not, determine and prepare for the next number of blocks to try

        # if not optimal_num_blocks_found:
        if 1:
            partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks, num_blocks_to_merge, old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks, optimal_num_blocks_found = \
                prepare_for_partition_on_next_num_blocks(overall_entropy, partition, interblock_edge_count, block_degrees,
                                                         block_degrees_out, block_degrees_in, num_blocks, old_partition,
                                                         old_interblock_edge_count, old_block_degrees, old_block_degrees_out,
                                                         old_block_degrees_in, old_overall_entropy, old_num_blocks,
                                                         num_block_reduction_rate)
        if args.verbose:
            print('Overall entropy: {}'.format(old_overall_entropy))
            print('Number of blocks: {}'.format(old_num_blocks))
            if optimal_num_blocks_found:
                print('\nOptimal partition found with {} blocks'.format(num_blocks))
            print('Proposals evaluated: {}'.format(n_proposals_evaluated))
            print('Overall nodal moves: {}'.format(total_num_nodal_moves))

    if use_timeit:
        t_prog_end = time.time()
        print('\nGraph partition took {} seconds'.format(t_prog_end - t_prog_start))

    # evaluate output partition against the true partition
    evaluate_partition(true_partition, partition)

block_sum_time_cum = 0
if __name__ == '__main__':
    # np.seterr(all='raise')
    if args.seed != -1:
        numpy.random.seed(args.seed % 4294967295)

    do_main(args)
    print("Block sum time = %s" % block_sum_time_cum)
