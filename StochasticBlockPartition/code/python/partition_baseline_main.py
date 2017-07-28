from partition_baseline_support import *
import multiprocessing as mp
import multiprocessing.pool
from multiprocessing import Pool, Value, current_process
from functools import reduce
import pickle
import timeit
import os, sys, argparse
import time, struct
import traceback
import numpy.random
from compute_delta_entropy import compute_delta_entropy
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


def compute_best_block_merge_wrapper(tup):
    (blocks, num_blocks) = tup

    interblock_edge_count = syms['interblock_edge_count']
    block_partition = syms['block_partition']
    block_degrees = syms['block_degrees']
    block_degrees_out = syms['block_degrees_out']
    block_degrees_in = syms['block_degrees_in']
    partition = syms['partition']

    return compute_best_block_merge(blocks, num_blocks, interblock_edge_count, block_partition, block_degrees, args.n_proposal, block_degrees_out, block_degrees_in)


def compute_best_block_merge(blocks, num_blocks, M, block_partition, block_degrees, n_proposal, block_degrees_out, block_degrees_in):
    best_overall_merge = [-1 for i in blocks]
    best_overall_delta_entropy = [np.Inf for i in blocks]
    n_proposals_evaluated = 0

    for current_block_idx,current_block in enumerate(blocks):
        if current_block is None:
            break

        # Index of non-zero block entries and their associated weights
        in_idx, in_weight = take_nonzero(M, current_block, 1)
        out_idx, out_weight = take_nonzero(M, current_block, 0)

        block_neighbors = np.concatenate((in_idx, out_idx))
        block_neighbor_weights = np.concatenate((in_weight, out_weight))

        num_out_block_edges = sum(out_weight)
        num_in_block_edges = sum(in_weight)
        num_block_edges = num_out_block_edges + num_in_block_edges

        n_proposal = 10
        delta_entropy = np.empty(n_proposal)


        proposals = np.empty(n_proposal, dtype=int)

        # propose new blocks to merge with
        for proposal_idx in range(n_proposal):
            s = propose_new_partition(
                current_block,
                block_neighbors,
                block_neighbor_weights,
                num_block_edges,
                block_partition, M, block_degrees, num_blocks,
                agg_move = 1)

            s = int(s)
            proposals[proposal_idx] = s

            # compute the two new rows and columns of the interblock edge count matrix
            new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col \
                = compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, s,
                                                    out_idx, out_weight,
                                                    in_idx, in_weight,
                                                    M[current_block, current_block],
                                                    agg_move = 1,
                                                    sparse = args.sparse)

            # compute change in entropy / posterior
            block_degrees_out_new, block_degrees_in_new, block_degrees_new \
                = compute_new_block_degrees(current_block,
                                            s,
                                            block_degrees_out,
                                            block_degrees_in,
                                            block_degrees,
                                            num_out_block_edges,
                                            num_in_block_edges,
                                            num_block_edges)

            delta_entropy[proposal_idx] = compute_delta_entropy(current_block, s, M,
                                                                new_M_r_row,
                                                                new_M_s_row,
                                                                new_M_r_col,
                                                                new_M_s_col,
                                                                block_degrees_out,
                                                                block_degrees_in,
                                                                block_degrees_out_new,
                                                                block_degrees_in_new)

        mi = np.argmin(delta_entropy)
        best_entropy = delta_entropy[mi]
        n_proposals_evaluated += n_proposal

        if best_entropy < best_overall_delta_entropy[current_block_idx]:
            best_overall_merge[current_block_idx] = proposals[mi]
            best_overall_delta_entropy[current_block_idx] = best_entropy

    return blocks, best_overall_merge, best_overall_delta_entropy, n_proposals_evaluated

update_id = -1
def propose_node_movement_wrapper(tup):

    global update_id, partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in

    start,stop,step = tup

    state = syms['state']
    lock = syms['lock']
    n_thread = syms['n_thread']

    results = syms['results']
    (results_proposal, results_delta_entropy, results_accept, results_propose_time_worker_ms, results_is_non_seq) = syms['results']

    if args.pipe:
        pipe = syms['pipe']

    pid = current_process().pid

    (update_id_shared, partition_shared, interblock_edge_count_shared, num_blocks, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared, where_modified_shared, out_neighbors, in_neighbors, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors) = state

    is_non_seq = 0

    lock.acquire()

    if update_id != update_id_shared.value - 1:
        is_non_seq = 1

    if update_id != update_id_shared.value:
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

        update_id = update_id_shared.value

    lock.release()

    # Ensure every worker has a different random seed.
    numpy.random.seed((pid + int(timeit.default_timer() * 1e6)) % 4294967295)

    for current_node in range(start, stop, step):

        t0 = timeit.default_timer()

        res = propose_node_movement(current_node, partition, out_neighbors, in_neighbors,
                interblock_edge_count, num_blocks, block_degrees, block_degrees_out, block_degrees_in,
                vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors)

        t1 = timeit.default_timer()
        t_elapsed_ms = (t1 - t0) * 1e3
        (ni, current_block, proposal, delta_entropy, p_accept) = res
        accept = (np.random.uniform() <= p_accept)

        results_proposal[ni] = proposal
        results_delta_entropy[ni] = delta_entropy
        results_accept[ni] = accept
        results_propose_time_worker_ms[ni] = t_elapsed_ms
        results_is_non_seq[ni] = is_non_seq

    if args.pipe:
        os.write(pipe[1], start.to_bytes(4, byteorder='little') + stop.to_bytes(4, byteorder='little') + step.to_bytes(4, byteorder='little'))
        return
    else:
        return start,stop,step

def propose_node_movement(current_node, partition, out_neighbors, in_neighbors, interblock_edge_count, num_blocks, block_degrees, block_degrees_out, block_degrees_in, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors):

    current_block = partition[current_node]
    # propose a new block for this node
    proposal = propose_new_partition(
        current_block,
        vertex_neighbors[current_node][:, 0],
        vertex_neighbors[current_node][:, 1],
        vertex_num_neighbor_edges[current_node],
        partition,
        interblock_edge_count, block_degrees, num_blocks, agg_move = 0)

    num_out_neighbor_edges = vertex_num_out_neighbor_edges[current_node]
    num_in_neighbor_edges = vertex_num_in_neighbor_edges[current_node]
    num_neighbor_edges = vertex_num_neighbor_edges[current_node]

    proposal = int(proposal)

    # determine whether to accept or reject the proposal
    if proposal == current_block:
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

        new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col = \
            compute_new_rows_cols_interblock_edge_count_matrix(interblock_edge_count, current_block, proposal,
                                                               blocks_out, count_out, blocks_in, count_in,
                                                               self_edge_weight, agg_move = 0, sparse = args.sparse)

        # compute new block degrees
        block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(
            current_block, proposal, block_degrees_out, block_degrees_in, block_degrees,
            num_out_neighbor_edges,
            num_in_neighbor_edges,
            num_neighbor_edges)

        # compute the Hastings correction
        Hastings_correction = compute_Hastings_correction(blocks_out, count_out, blocks_in, count_in,
                                                          current_block,
                                                          proposal,
                                                          interblock_edge_count,
                                                          new_M_r_row,
                                                          new_M_r_col,
                                                          num_blocks, block_degrees,
                                                          block_degrees_new)

        # compute change in entropy / posterior
        delta_entropy = compute_delta_entropy(current_block, proposal,
                                              interblock_edge_count,
                                              new_M_r_row,
                                              new_M_s_row,
                                              new_M_r_col,
                                              new_M_s_col, block_degrees_out,
                                              block_degrees_in, block_degrees_out_new, block_degrees_in_new)

        # compute probability of acceptance

        # Clamp to avoid under- and overflow
        if delta_entropy > 10.0:
            delta_entropy = 10.0
        elif delta_entropy < -10.0:
            delta_entropy = -10.0

        p_accept = np.min([np.exp(-args.beta * delta_entropy) * Hastings_correction, 1])

    return current_node, current_block, int(proposal), delta_entropy, p_accept

def coo_to_flat(x, size):
    x_i, x_v = x
    f = np.zeros(size)
    f[x_i] = x_v
    return f

def update_partition_single(b, ni, s, M, M_r_row, M_s_row, M_r_col, M_s_col):
    r = b[ni]
    b[ni] = s

    if type(M) is fast_sparse_array:
        if type(M_r_row) is nonzero_dict:
            M.set_axis_dict(r, 0, M_r_row)
        else:
            M.set_row_nonzeros(r, M_r_row[0], M_r_row[1])

        if type(M_s_row) is nonzero_dict:
            M.set_axis_dict(s, 0, M_s_row)
        else:
            M.set_row_nonzeros(s, M_s_row[0], M_s_row[1])

        if type(M_r_col) is nonzero_dict:
            M.set_axis_dict(r, 1, M_r_col)
        else:
            M.set_col_nonzeros(r, M_r_col[0], M_r_col[1])

        if type(M_s_col) is nonzero_dict:
            M.set_axis_dict(s, 1, M_s_col)
        else:
            M.set_col_nonzeros(s, M_s_col[0], M_s_col[1])

        return b, M

    if type(M_r_row) is tuple:
        M[r, :] = coo_to_flat(M_r_row, M.shape[0])
    else:
        M[r, :] = M_r_row

    if type(M_r_col) is tuple:
        M[:, r] = coo_to_flat(M_r_col, M.shape[0])
    else:
        M[:, r] = M_r_col

    if type(M_s_row) is tuple:
        M[s, :] = coo_to_flat(M_s_row, M.shape[0])
    else:
        M[s, :] = M_s_row

    if type(M_s_col) is tuple:
        M[:, s] = coo_to_flat(M_s_col, M.shape[0])
    else:
        M[:, s] = M_s_col

    return b, M

def shared_memory_copy(z):
    prod = reduce((lambda x,y : x*y), (i for i in z.shape))
    ctype = {"float64" : ctypes.c_double, "int64" : ctypes.c_int64, "bool" : ctypes.c_bool}[str(z.dtype)]
    raw = sharedctypes.RawArray(ctype, prod)
    a = np.frombuffer(raw, dtype=z.dtype).reshape(z.shape)
    a[:] = z
    return a

def shared_memory_empty(shape, dtype='int64'):
    prod = reduce((lambda x,y : x*y), (i for i in shape))
    ctype = {"float64" : ctypes.c_double, "float" : ctypes.c_double, "int64" : ctypes.c_int64, "int" : ctypes.c_int, "bool" : ctypes.c_bool}[str(dtype)]
    raw = sharedctypes.RawArray(ctype, prod)
    a = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return a


def nodal_moves_sequential(batch_size, max_num_nodal_itr, delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy_cur, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose):
    global block_sum_time_cum

    total_num_nodal_moves_itr = 0
    itr_delta_entropy = np.zeros(max_num_nodal_itr)

    for itr in range(max_num_nodal_itr):
        num_nodal_moves = 0
        block_sum_time = 0.0
        itr_delta_entropy[itr] = 0

        propose_time_cum = 0
        merge_time_cum = 0
        update_partition_time_ms_cum = 0.0

        if args.sort:
            #L = np.argsort(partition)
            L = entropy_max_argsort(partition)
        else:
            L = range(0, N)

        proposal_cnt = 0
        update_id_cnt = 0

        modified = np.zeros(M.shape[0], dtype=bool)

        t_merge_start = timeit.default_timer()

        for i in L:
            t_propose_start = timeit.default_timer()
            movement = propose_node_movement(i, partition, out_neighbors, in_neighbors, M, num_blocks, block_degrees, block_degrees_out, block_degrees_in, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors)

            t_propose_end = timeit.default_timer()
            propose_time = (t_propose_end - t_propose_start)
            propose_time_cum += propose_time

            proposal_cnt += 1

            (ni, current_block, proposal, delta_entropy, p_accept) = movement
            accept = (np.random.uniform() <= p_accept)

            if not accept:
                continue

            t_update_partition_beg = timeit.default_timer()

            total_num_nodal_moves_itr += 1
            num_nodal_moves += 1
            itr_delta_entropy[itr] += delta_entropy

            modified[partition[ni]] = True
            modified[proposal] = True

            current_block = partition[ni]

            if verbose > 2:
                print("Move %5d from block %5d to block %5d." % (ni, current_block, proposal))

            blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[ni][:, 0]], return_inverse=True)
            count_out = np.bincount(inverse_idx_out, weights=out_neighbors[ni][:, 1]).astype(int)
            blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[ni][:, 0]], return_inverse=True)
            count_in = np.bincount(inverse_idx_in, weights=in_neighbors[ni][:, 1]).astype(int)
            self_edge_weight = np.sum(out_neighbors[ni][np.where(out_neighbors[ni][:, 0] == ni), 1])

            (new_M_r_row, new_M_s_row,new_M_r_block_col, new_M_s_col) = \
                                compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, proposal,
                                                                                   blocks_out, count_out, blocks_in, count_in,
                                                                                   self_edge_weight, agg_move = 0, sparse = args.sparse)

            partition, M = update_partition_single(partition, ni, proposal, M,
                                                   new_M_r_row, new_M_s_row, new_M_r_block_col, new_M_s_col)

            t_update_partition_end = timeit.default_timer()

            update_partition_time_ms_cum += (t_update_partition_end - t_update_partition_beg) * 1e3

            btime = timeit.default_timer()
            where_modified = np.where(modified)[0]

            if 0: #type(M) is not fast_sparse_array:
                block_degrees_out[where_modified] = np.sum(M[where_modified, :], axis = 1)
                block_degrees_in[where_modified] = np.sum(M[:, where_modified], axis = 0)
            elif 1:
                for w in where_modified:
                    nz_i, nz_v = take_nonzero(M, w, 0, sort=False)
                    block_degrees_out[w] = np.sum(nz_v)
                    nz_i, nz_v = take_nonzero(M, w, 1, sort=False)
                    block_degrees_in[w] = np.sum(nz_v)

            block_degrees[where_modified] = block_degrees_out[where_modified] + block_degrees_in[where_modified]
            btime_end = timeit.default_timer()
            block_sum_time += btime_end - btime
            block_sum_time_cum += btime_end - btime

        t_merge_end = timeit.default_timer()
        merge_time = (t_merge_end - t_merge_start)
        merge_time_cum += merge_time

        if num_nodal_moves != 0:
            merge_rate_ms = merge_time * 1e-3 / num_nodal_moves
        else:
            merge_rate_ms = 0.0

        if verbose:
            print("Processed %d nodal movements in %3.4f ms rate = %f per ms." % (num_nodal_moves, merge_time * 1e-3, merge_rate_ms))

            print("Node propose time is %3.2f ms, merge time is %3.2f ms, block sum time is %3.2f ms, partition update time is %3.2f ms, ratio propose to merge %3.2f"
              % (propose_time_cum * 1e-3, merge_time_cum * 1e-3, block_sum_time * 1e3, update_partition_time_ms_cum,  (propose_time_cum) / merge_time_cum))
            print("Itr: {:3d}, number of nodal moves: {:3d}, delta S: {:0.9f}".format(itr, num_nodal_moves,
                                                                                itr_delta_entropy[itr] / float(
                                                                                    overall_entropy_cur)))

        # exit MCMC if the recent change in entropy falls below a small fraction of the overall entropy
        if itr >= (delta_entropy_moving_avg_window - 1):  
            if (-np.mean(itr_delta_entropy[(itr - delta_entropy_moving_avg_window + 1):itr]) < (
                    delta_entropy_threshold * overall_entropy_cur)):
                    break

    return total_num_nodal_moves_itr


def nodal_moves_parallel(n_thread, batch_size, max_num_nodal_itr, delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy_cur, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose = False):
    global syms, block_sum_time_cum

    total_num_nodal_moves_itr = 0
    itr_delta_entropy = np.zeros(max_num_nodal_itr)

    lock = mp.Lock()

    modified = np.zeros(M.shape[0], dtype=bool)
    where_modified_shared = shared_memory_copy(modified)
    update_id_shared = Value('i', 0)

    (M_shared, partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared) \
        = (shared_memory_copy(i) for i in (M, partition, block_degrees, block_degrees_out, block_degrees_in))

    state = (update_id_shared, partition_shared, M_shared, num_blocks, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared, where_modified_shared, out_neighbors, in_neighbors, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors)


    shape = partition.shape
    results_proposal = shared_memory_empty(shape)
    results_delta_entropy = shared_memory_empty(shape, dtype='float')
    results_accept = shared_memory_empty(shape, dtype='bool')
    results_propose_time_worker_ms = shared_memory_empty(shape, dtype='float')
    results_is_non_seq = shared_memory_empty(shape, dtype='bool')

    syms = {}
    syms['results'] = (results_proposal, results_delta_entropy, results_accept, results_propose_time_worker_ms, results_is_non_seq)
    syms['lock'] = lock
    syms['state'] = state
    syms['n_thread'] = n_thread

    if args.pipe:
        pipe = os.pipe()
        try:
            os.set_inheritable(pipe[0], True)
            os.set_inheritable(pipe[1], True)
        except AttributeError:
            raise Exception("Python2 no longer supports heritable pipes.")
        syms['pipe'] = pipe

    pool = Pool(n_thread)
    update_id_cnt = 0

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

        if args.sort:
            #L = np.argsort(partition)
            L = entropy_max_argsort(partition)
        else:
            L = range(0, N)
            
        t_propose_start = timeit.default_timer()

        propose_movement_batch_size = 10

        group_size = args.node_propose_batch_size
        chunks = [(i, min(i+group_size, N), 1) for i in range(0,N,group_size)]

        if 1:
            movements = pool.imap_unordered(propose_node_movement_wrapper, chunks)
        else:
            status = [pool.apply_async(propose_node_movement_wrapper, (j,)) for j in chunks]

        t_propose_end = timeit.default_timer()
        propose_time_ms = (t_propose_end - t_propose_start) * 1e3
        propose_time_ms_cum += propose_time_ms

        t_merge_start = timeit.default_timer()

        proposal_cnt = 0
        next_batch_cnt = num_nodal_moves + batch_size

        cnt_seq_workers = 0
        cnt_non_seq_workers = 0

        while proposal_cnt < N:

            if args.pipe:
                buf = os.read(pipe[0], 4*3)
                start,stop,step = struct.unpack('<iii', buf)
            else:
                start,stop,step = movements.next()
    
            for ni in range(start,stop,step):

                useless_time_beg = timeit.default_timer()

                proposal = results_proposal[ni]
                delta_entropy = results_delta_entropy[ni]
                accept = results_accept[ni]
                propose_time_worker_ms = results_propose_time_worker_ms[ni]
                is_non_seq = results_is_non_seq[ni]
                is_seq = not is_non_seq

                cnt_seq_workers += is_seq
                cnt_non_seq_workers += is_non_seq

                #print("Got a result for index %d from pid %d" % (ni,pid))

                proposal_cnt += 1
                propose_time_workers_ms_cum += propose_time_worker_ms

                useless_time_end = timeit.default_timer()
                t_useless += useless_time_end - useless_time_beg

                if accept:
                    t_update_partition_beg = timeit.default_timer()

                    total_num_nodal_moves_itr += 1
                    num_nodal_moves += 1
                    itr_delta_entropy[itr] += delta_entropy

                    modified[partition[ni]] = True
                    modified[proposal] = True

                    current_block = partition[ni]

                    if 0:
                        print("Move %s from block %s to block %s." % (ni, current_block, proposal))

                    blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[ni][:, 0]], return_inverse=True)
                    count_out = np.bincount(inverse_idx_out, weights=out_neighbors[ni][:, 1]).astype(int)
                    blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[ni][:, 0]], return_inverse=True)
                    count_in = np.bincount(inverse_idx_in, weights=in_neighbors[ni][:, 1]).astype(int)
                    self_edge_weight = np.sum(out_neighbors[ni][np.where(out_neighbors[ni][:, 0] == ni), 1])


                    (new_M_r_row, new_M_s_row,new_M_r_block_col, new_M_s_col) = \
                                        compute_new_rows_cols_interblock_edge_count_matrix(
                                            M, current_block, proposal,
                                            blocks_out, count_out, blocks_in, count_in,
                                            self_edge_weight, agg_move = 0, sparse = args.sparse)

                    partition, M = update_partition_single(partition, ni, proposal, M,
                                                           new_M_r_row, new_M_s_row, new_M_r_block_col, new_M_s_col)

                    t_update_partition_end = timeit.default_timer()

                    update_partition_time_ms_cum += (t_update_partition_end - t_update_partition_beg) * 1e3

                if num_nodal_moves >= next_batch_cnt or proposal_cnt == N:
                    btime = timeit.default_timer()
                    where_modified = np.where(modified)[0]
                    block_degrees_out[where_modified] = np.sum(M[where_modified, :], axis = 1)
                    block_degrees_in[where_modified] = np.sum(M[:, where_modified], axis = 0)
                    block_degrees[where_modified] = block_degrees_out[where_modified] + block_degrees_in[where_modified]
                    next_batch_cnt = num_nodal_moves + batch_size

                    btime_end = timeit.default_timer()
                    block_sum_time += btime_end - btime
                    block_sum_time_cum += btime_end - btime

                    update_beg = timeit.default_timer()
                    update_id_cnt += 1

                    block = (proposal_cnt == N)

                    if lock.acquire(block=block):
                        where_modified_shared[:] = modified[:]

                        M_shared[where_modified, :] = M[where_modified, :]
                        M_shared[:, where_modified] = M[:, where_modified]
                        block_degrees_in_shared[where_modified] = block_degrees_in[where_modified]
                        block_degrees_out_shared[where_modified] = block_degrees_out[where_modified]
                        block_degrees_shared[where_modified] = block_degrees[where_modified]

                        partition_shared[:] = partition[:]
                        update_id_shared.value = update_id_cnt

                        lock.release()

                        modified[where_modified] = False

                    update_end = timeit.default_timer()
                    update_shared_time += update_end - update_beg

        t_merge_end = timeit.default_timer()
        merge_time_ms = (t_merge_end - t_merge_start) * 1e3
        merge_time_ms_cum += merge_time_ms

        if num_nodal_moves != 0:
            merge_rate_ms = merge_time_ms / num_nodal_moves
        else:
            merge_rate_ms = 0.0

        if verbose:
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

    pool.close()
    return total_num_nodal_moves_itr


def entropy_for_block_count(num_blocks, num_target_blocks, delta_entropy_threshold, M, block_degrees, block_degrees_out, block_degrees_in, out_neighbors, in_neighbors, N, E, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, partition, verbose = False):
    global syms, block_sum_time_cum

    t_start = timeit.default_timer()
    parallel_phase1 = (args.parallel_phase & 1) != 0

    n_thread = args.threads

    n_merges = 0
    n_proposals_evaluated = 0

    # begin agglomerative partition updates (i.e. block merging)
    if verbose:
        print("\nMerging down blocks from {} to {} at time {:4.4f}".format(num_blocks, num_target_blocks, timeit.default_timer() - t_prog_start))

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

        pool = Pool(n_thread)
        for current_blocks,best_merge,best_delta_entropy,fresh_proposals_evaluated in pool.imap_unordered(compute_best_block_merge_wrapper, [((i,),num_blocks) for i in L]):
            for current_block_idx,current_block in enumerate(current_blocks):
                best_merge_for_each_block[current_block] = best_merge[current_block_idx]
                delta_entropy_for_each_block[current_block] = best_delta_entropy[current_block_idx]
            n_proposals_evaluated += fresh_proposals_evaluated                
        pool.close()
    else:
        current_blocks,best_merge,best_delta_entropy,fresh_proposals_evaluated \
            = compute_best_block_merge(range(num_blocks), num_blocks, M,
                        block_partition, block_degrees, args.n_proposal, block_degrees_out, block_degrees_in)

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
                                                                                               args.sparse)
        # compute the global entropy for MCMC convergence criterion
        overall_entropy = compute_overall_entropy(M_t, block_degrees_out_t, block_degrees_in_t, num_blocks_t, N,
                                                  E)

        overall_entropy_per_num_blocks[i] = overall_entropy
        state_per_num_blocks[i] = (overall_entropy, partition_t, num_blocks_t, M_t, block_degrees_out_t, block_degrees_in_t, block_degrees_t)

    S = overall_entropy_per_num_blocks[::-1]
    dS_dn = 0.5 * (S[2] - S[1]) + 0.5 * (S[1] - S[0])

    optimal_stop_found = False
    if len(num_target_blocks) == 3 \
       and overall_entropy_per_num_blocks[1] < overall_entropy_per_num_blocks[0] \
       and overall_entropy_per_num_blocks[1] < overall_entropy_per_num_blocks[2]:
        if verbose:
            print("Optimal stopping criterion found at %d blocks derivative %s at time %4.4f." % (num_target_blocks[1], dS_dn, timeit.default_timer() - t_prog_start))

        optimal_stop_found = True
        best_idx = 1
    else:
        extrapolated_newton = num_target_blocks[1] - 0.5 * (S[2] - S[0]) / (S[2] - S[1] - (S[1] - S[0]))

        if verbose:
            print("Stopping criterion not found at %s blocks extrapolate to %s blocks derivative %s." % (num_target_blocks[1], extrapolated_newton, dS_dn))

        best_idx = np.argsort(overall_entropy)
        best_idx = 1 # xxx

    (overall_entropy, partition, num_blocks_t, M, block_degrees_out, block_degrees_in, block_degrees) = state_per_num_blocks[best_idx]

    num_blocks_merged = num_blocks - num_blocks_t
    num_blocks = num_blocks_t

    if verbose:
        print("Best num_blocks = %s" % num_blocks)
        print("blocks %s entropy %s" % (num_target_blocks, overall_entropy_per_num_blocks))
        print("Beginning nodal updates")

    batch_size = args.node_move_update_batch_size

    parallel_phase2 = (args.parallel_phase & 2) != 0

    if parallel_phase2 and n_thread > 0:
        total_num_nodal_moves_itr = nodal_moves_parallel(n_thread, batch_size, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose)
    else:
        total_num_nodal_moves_itr = nodal_moves_sequential(batch_size, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose)

    # compute the global entropy for determining the optimal number of blocks
    overall_entropy = compute_overall_entropy(M, block_degrees_out, block_degrees_in, num_blocks, N, E)

    if verbose:
        t_end = timeit.default_timer()
        time_taken = t_end - t_start
        print(
            "Total number of nodal moves: {:3d}, overall_entropy: {:0.2f}".format(total_num_nodal_moves_itr, overall_entropy))
        print("Time taken: {:3.4f}".format(time_taken))

    if args.visualize_graph:
        graph_object = plot_graph_with_partition(out_neighbors, partition, graph_object)

    return overall_entropy, n_proposals_evaluated, n_merges, total_num_nodal_moves_itr, M, block_degrees, block_degrees_out, block_degrees_in, num_blocks_merged, partition, optimal_stop_found, dS_dn


def load_graph_parts(input_filename):
    true_partition_available = True
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
            out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available, permutate=0)
    return out_neighbors, in_neighbors, N, E, true_partition

def find_optimal_partition(out_neighbors, in_neighbors, N, E, stop_at_bracket = False, verbose = False, partition_bracket = [], num_block_reduction_rate = 0.5):

    if verbose:
        print('Number of nodes: {}'.format(N))
        print('Number of edges: {}'.format(E))

    # partition update parameters
    args.beta = 3.0  # exploitation versus exploration (higher value favors exploitation)

    # agglomerative partition update parameters
    num_agg_proposals_per_block = 10  # number of agglomerative merge proposals per block
    # num_block_reduction_rate is fraction of blocks to reduce until the golden ratio bracket is established

    # nodal partition updates parameters
    args.max_num_nodal_itr = 100  # maximum number of iterations
    delta_entropy_threshold1 = 5e-4  # stop iterating when the change in entropy falls below this fraction of the overall entropy
                                     # lowering this threshold results in more nodal update iterations and likely better performance, but longer runtime
    delta_entropy_threshold2 = 1e-4  # threshold after the golden ratio bracket is established (typically lower to fine-tune to partition)
    args.delta_entropy_moving_avg_window = 3  # width of the moving average window for the delta entropy convergence criterion

    vertex_num_in_neighbor_edges = np.empty(N, dtype=int)
    vertex_num_out_neighbor_edges = np.empty(N, dtype=int)
    vertex_num_neighbor_edges = np.empty(N, dtype=int)
    vertex_neighbors = [np.concatenate((out_neighbors[i], in_neighbors[i])) for i in range(N)]

    for i in range(N):
        vertex_num_out_neighbor_edges[i] = sum(out_neighbors[i][:,1])
        vertex_num_in_neighbor_edges[i] = sum(in_neighbors[i][:,1])
        vertex_num_neighbor_edges[i] = vertex_num_out_neighbor_edges[i] + vertex_num_in_neighbor_edges[i]


    optimal_num_blocks_found = False

    if not partition_bracket:
        # initialize by putting each node in its own block (N blocks)
        num_blocks = N
        partition = np.arange(num_blocks, dtype=int)

        # initialize edge counts and block degrees
        interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees \
            = initialize_edge_counts(out_neighbors,
                                     num_blocks,
                                     partition,
                                     args.sparse)
        # initialize items before iterations to find the partition with the optimal number of blocks
        hist, graph_object = initialize_partition_variables()
        num_blocks_to_merge = int(num_blocks * num_block_reduction_rate)
        golden_ratio_bracked_established = False
        delta_entropy_threshold = delta_entropy_threshold1
    else:
        # resume search from a previous partition state
        if len(partition_bracket) == 3:
            (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) \
                = ([], [], [], [], [], [], [])

            for partition in partition_bracket:
                num_blocks = 1 + np.max(partition)
                interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees \
                    = initialize_edge_counts(out_neighbors, num_blocks, partition, args.sparse)

                overall_entropy = compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in, num_blocks, N, E)

                for i,j in zip((partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, overall_entropy, num_blocks),
                        (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks)):
                    j.append(i)

                print("Resuming with num_blocks = %s overall_entropy = %s" % (old_num_blocks, old_overall_entropy))
                hist = (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks)

        elif len(partition_bracket) == 1:
            hist, graph_object = initialize_partition_variables()
            partition = partition_bracket[0]
            num_blocks = 1 + np.max(partition)
            interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees \
                = initialize_edge_counts(out_neighbors, num_blocks, partition, args.sparse)
            overall_entropy = compute_overall_entropy(interblock_edge_count, block_degrees_out, block_degrees_in, num_blocks, N, E)
            

        print("optimal = %s num_blocks = %s" % (optimal_num_blocks_found, num_blocks))
        partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks, num_blocks_to_merge, hist, optimal_num_blocks_found = \
                                                        prepare_for_partition_on_next_num_blocks(overall_entropy, partition, interblock_edge_count, block_degrees,
                                                                                                 block_degrees_out, block_degrees_in, num_blocks, hist,
                                                                                                 num_block_reduction_rate)
        print("optimal = %s" % optimal_num_blocks_found)

        # golden ratio bracket was previously established
        golden_ratio_bracked_established = False
        delta_entropy_threshold = delta_entropy_threshold2

    n_proposals_evaluated = 0
    total_num_nodal_moves = 0
    args.n_proposal = 1

    while not optimal_num_blocks_found:

        # Must be in decreasing order because reducing by carrying out merges modifies state.
        target_blocks = [num_blocks - num_blocks_to_merge + 1, num_blocks - num_blocks_to_merge, num_blocks - num_blocks_to_merge - 1]

        (overall_entropy,n_proposals_itr,n_merges_itr,total_num_nodal_moves_itr, \
         interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks_merged, partition, optimal_num_blocks_found, dS_dn) \
         = entropy_for_block_count(num_blocks, target_blocks,
                                   delta_entropy_threshold,
                                   interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in,
                                   out_neighbors, in_neighbors, N, E,
                                   vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors,
                                   partition, verbose)
        if verbose:
            print("num_blocks = %s num_blocks_merged = %s M.shape = %s" % (num_blocks, num_blocks_merged, str(interblock_edge_count.shape)))

        num_blocks -= num_blocks_merged
        total_num_nodal_moves += total_num_nodal_moves_itr
        n_proposals_evaluated += n_proposals_itr

        # check whether the partition with optimal number of block has been found; if not, determine and prepare for the next number of blocks to try

        # if not optimal_num_blocks_found:
        if 1:
            partition, interblock_edge_count, block_degrees, block_degrees_out, block_degrees_in, num_blocks, num_blocks_to_merge, hist, optimal_num_blocks_found = \
                prepare_for_partition_on_next_num_blocks(overall_entropy, partition, interblock_edge_count, block_degrees,
                                                         block_degrees_out, block_degrees_in, num_blocks, hist,
                                                         num_block_reduction_rate)

            (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) = hist

        if np.all(np.isfinite(old_overall_entropy)):
            if not golden_ratio_bracked_established:
                golden_ratio_bracked_established = True
                print("Golden ratio found at blocks %s at time %4.4f entropy %s" % (old_num_blocks, timeit.default_timer() - t_prog_start, old_overall_entropy))
            if stop_at_bracket:
                break
            delta_entropy_threshold = delta_entropy_threshold2

        if verbose:
            print('Overall entropy: {}'.format(old_overall_entropy))
            print('Number of blocks: {}'.format(old_num_blocks))
            if optimal_num_blocks_found:
                print('\nOptimal partition found with {} blocks'.format(num_blocks))
            print('Proposals evaluated: {}'.format(n_proposals_evaluated))
            print('Overall nodal moves: {}'.format(total_num_nodal_moves))


    partition_bracket = old_partition

    print("Partition blocks is %s" % (1 + np.max(partition_bracket[0])))

    return partition_bracket, old_interblock_edge_count[1]

def find_optimal_partition_wrapper(tup):
    args.threads = max(1, args.threads // args.decimation)
    out_neighbors, in_neighbors, N, E, true_partition = tup
    return find_optimal_partition(out_neighbors, in_neighbors, N, E, stop_at_bracket = True, verbose = min(0, args.verbose - 1))


# See: https://stackoverflow.com/questions/17223301/python-multiprocessing-is-it-possible-to-have-a-pool-inside-of-a-pool
class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NonDaemonicPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def merge_partitions(partitions, stop_pieces, out_neighbors, verbose):
    """
    Create a unified graph block partition from the supplied partition pieces into a partiton of size stop_pieces.
    """

    pieces = len(partitions)
    N = sum(len(i) for i in partitions)

    # The temporary partition variable is for the purpose of computing M.
    # The axes of M are concatenated block ids from each partition.
    # And each partition[i] will have an offset added to so all the interim partition ranges are globally unique.
    #
    partition = np.zeros(N, dtype=int)

    while pieces > stop_pieces:

        Bs = [max(partitions[i]) + 1 for i in range(pieces)]
        B =  sum(Bs)

        partition_offsets = np.zeros(pieces, dtype=int)
        partition_offsets[1:] = np.cumsum(Bs)[:-1]

        if verbose > 1:
            print("")
            print("Reconstitute graph from %d pieces B[piece] = %s" % (pieces,Bs))
            print("partition_offsets = %s" % partition_offsets)

        # It would likely be faster to re-use already computed values of M from pieces:
        #     M[ 0:B0,     0:B0   ] = M_0
        #     M[B0:B0+B1, B0:B0+B1] = M_1
        # Instead of relying on initialize_edge_counts.

        M, block_degrees_out, block_degrees_in, block_degrees \
            = initialize_edge_counts(out_neighbors, B, partition, args.sparse)

        if args.verbose > 2:
            print("M.shape = %s, M = \n%s" % (str(M.shape),M))

        next_partitions = []
        for i in range(0, pieces, 2):
            print("Merge piece %d and %d into %d" % (i, i + 1, i // 2))
            partitions[i],_ = merge_two_partitions(M, block_degrees_out, block_degrees_out, block_degrees_out,
                                                   partitions[i], partitions[i + 1],
                                                   partition_offsets[i], partition_offsets[i + 1],
                                                   Bs[i], Bs[i + 1])
            next_partitions.append(np.concatenate((partitions[i], partitions[i+1])))

        partitions = next_partitions
        pieces //= 2

    return partitions



def merge_two_partitions(M, block_degrees_out, block_degrees_in, block_degrees, partition0, partition1, partition_offset_0, partition_offset_1, B0, B1):
    """
    Merge two partitions each from a decimated piece of the graph.
    Note
        M : ndarray or sparse matrix (int), shape = (#blocks, #blocks)
                    block count matrix between all the blocks, of which partition 0 and partition 1 are just subsets
        partition_offset_0 and partition_offset_1 are the starting offsets within M of each partition piece
    """
    # Now reduce by merging down blocks from partition 0 into partition 1.
    # This requires computing delta_entropy over all of M (hence the partition_offsets are needed).

    delta_entropy = np.empty((B0,B1))

    for r in range(B0):
        current_block = r + partition_offset_0

        # Index of non-zero block entries and their associated weights
        in_idx, in_weight = take_nonzero(M, current_block, 1)
        out_idx, out_weight = take_nonzero(M, current_block, 0)

        block_neighbors = np.concatenate((in_idx, out_idx))
        block_neighbor_weights = np.concatenate((in_weight, out_weight))

        num_out_block_edges = sum(out_weight)
        num_in_block_edges = sum(in_weight)
        num_block_edges = num_out_block_edges + num_in_block_edges

        for s in range(B1):
            proposal = s + partition_offset_1

            new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col \
                = compute_new_rows_cols_interblock_edge_count_matrix(M, current_block, proposal,
                                                                     out_idx, out_weight,
                                                                     in_idx, in_weight,
                                                                     M[current_block, current_block], agg_move = 1)

            block_degrees_out_new, block_degrees_in_new, block_degrees_new \
                = compute_new_block_degrees(current_block,
                                            proposal,
                                            block_degrees_out,
                                            block_degrees_in,
                                            block_degrees,
                                            num_out_block_edges,
                                            num_in_block_edges,
                                            num_block_edges)

            delta_entropy[r, s] = compute_delta_entropy(current_block, proposal, M,
                                                        new_M_r_row,
                                                        new_M_s_row,
                                                        new_M_r_col,
                                                        new_M_s_col,
                                                        block_degrees_out,
                                                        block_degrees_in,
                                                        block_degrees_out_new,
                                                        block_degrees_in_new)

    best_merge_for_each_block = np.argmin(delta_entropy, axis = 1)

    if args.verbose > 2:
        print("delta_entropy = \n%s" % delta_entropy)
        print("best_merge_for_each_block = %s" % best_merge_for_each_block)

    delta_entropy_for_each_block = delta_entropy[np.arange(delta_entropy.shape[0]), best_merge_for_each_block]

    # Global number of blocks (when all pieces are considered together).
    num_blocks = M.shape[0]
    num_blocks_to_merge = B0
    best_merges = delta_entropy_for_each_block.argsort()

    # Note: partition0 will be modified in carry_out_best_merges
    (partition, num_blocks) = carry_out_best_merges(delta_entropy_for_each_block,
                                                    best_merges,
                                                    best_merge_for_each_block + partition_offset_1,
                                                    partition0,
                                                    num_blocks,
                                                    num_blocks_to_merge, verbose=(args.verbose > 2))

    return partition, num_blocks


def do_main(args):
    input_filename = args.input_filename
    args.visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions

    np.set_printoptions(linewidth=159)

    out_neighbors, in_neighbors, N, E, true_partition = load_graph_parts(input_filename)

    if args.verbose > 1:
        from collections import Counter
        print("Overall true partition statistics:")
        print("[" + "".join(("%5d : %3d, " % (i,e,) for i,e in sorted([(e,i) for (i,e) in Counter(true_partition).items()]))) + "]\n")


    if args.predecimation > 1:
        out_neighbors, in_neighbors, N, E, true_partition = decimate_graph(out_neighbors, in_neighbors, true_partition,
                                                                           decimation = args.predecimation, decimated_piece = 0)

    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_procs = 2**int(np.log2(comm.size))

        # MPI-based decimation is only supported for powers of 2
        if comm.rank >= mpi_procs:
            comm.Barrier()
            return

        print("Hello! I am rank %4d from %4d running in total limit is %d" % (comm.rank, comm.size, mpi_procs))

        decimation = mpi_procs
        out_neighbors_piece, in_neighbors_piece, N_piece, E_piece, true_partition_piece \
            = decimate_graph(out_neighbors, in_neighbors, true_partition,
                             decimation, decimated_piece = comm.rank)

        t_prog_start = timeit.default_timer()
        partition_bracket, M = find_optimal_partition(out_neighbors_piece, in_neighbors_piece, N_piece, E_piece, stop_at_bracket = False, verbose = args.verbose)
        t_prog_end = timeit.default_timer()

        partition = partition_bracket[0]

        if comm.rank != 0:
            comm.send(true_partition_piece, dest=0, tag=11)
            comm.send(partition, dest=0, tag=11)
            comm.Barrier()
            return
        else:
            true_partitions = [true_partition_piece] + [comm.recv(source=i, tag=11) for i in range(1, mpi_procs)]
            partitions = [partition] + [comm.recv(source=i, tag=11) for i in range(1, mpi_procs)]
            comm.Barrier()

    elif args.decimation > 1:
        decimation = args.decimation

        # Re-start timer after decimation is complete
        t_prog_start = timeit.default_timer()

        pieces = [decimate_graph(out_neighbors, in_neighbors, true_partition, decimation, i) for i in range(decimation)]
        _,_,_,_,true_partitions = zip(*pieces)

        if args.verbose > 1:
            for j,_ in enumerate(true_partitions):
                print("Overall true partition %d statistics:" % (j))
                print("[" + "".join(("%5d : %3d, " % (i,e,) for i,e in sorted([(e,i) for (i,e) in Counter(true_partitions[j]).items()]))) + "]\n")


        pool = NonDaemonicPool(decimation)

        results = pool.map(find_optimal_partition_wrapper, pieces)
        partition_brackets,Ms = (list(i) for i in zip(*results))

        pool.close()
        partitions = [partition_brackets[i][0] for i in range(decimation)]
    else:
        decimation = 1
        t_prog_start = timeit.default_timer()

        if 0:
            partition_bracket, M_bracket = find_optimal_partition(out_neighbors, in_neighbors, N, E, stop_at_bracket = False, verbose = args.verbose)
        else:
            # xxx stop
            partition_bracket, M_bracket = find_optimal_partition(out_neighbors, in_neighbors, N, E, stop_at_bracket = True, verbose = args.verbose)
            print("")
            print("Resume bracket search.")
            print("")
            partition_bracket, M_bracket = find_optimal_partition(out_neighbors, in_neighbors, N, E, stop_at_bracket = False, verbose = args.verbose, partition_bracket = partition_bracket)
            partition = partition_bracket[1]

        t_prog_end = timeit.default_timer()

        if args.test_decimation > 0:
            decimation = args.test_decimation
            true_partitions = [true_partition[i::decimation] for i in range(decimation)]
            partitions = [partition[i::decimation] for i in range(decimation)]


    # Either multiprocess pool or MPI results need merging
    if decimation > 1:
        if args.verbose > 2:
            for i in range(decimation):
                print("")
                print("Evaluate subgraph %d:" % i)
                evaluate_partition(true_partitions[i], partitions[i])

        t_decimation_merge_start = timeit.default_timer()


        # Merge all pieces into a smaller number.
        partitions = merge_partitions(partitions,
                                      4, out_neighbors, args.verbose)

        # Now merge all remaining pieces into one big partition and then merge down.
        Bs = [max(i) + 1 for i in partitions]
        partition = np.zeros(N, dtype=int)
        partition_offsets = np.zeros(len(partitions), dtype=int)
        partition_offsets[1:] = np.cumsum(Bs)[:-1]

        partition = np.concatenate([partitions[i] + partition_offsets[i] for i in range(len(partitions))])

        t_decimation_merge_end = timeit.default_timer()
        print("Decimation merge time is %3.5f" % (t_decimation_merge_end - t_decimation_merge_start))

        t_final_partition_search_start = timeit.default_timer()

        partition_bracket, M_bracket = find_optimal_partition(out_neighbors, in_neighbors, N, E,
                                                              stop_at_bracket = False, verbose = args.verbose,
                                                              partition_bracket = [partition],
                                                              num_block_reduction_rate = 0.35)

        t_final_partition_search_end = timeit.default_timer()

        partition = partition_bracket[1]
        t_prog_end = timeit.default_timer()
        print("Final partition search took %3.5f" % (t_final_partition_search_end - t_final_partition_search_start))

    print('\nGraph partition took {} seconds'.format(t_prog_end - t_prog_start))
    evaluate_partition(true_partition, partition)

block_sum_time_cum = 0
# xxx global for use in merge down blocks incremental time
# does not affect overall graph partition time report
t_prog_start = timeit.default_timer()
print("Program start at %s sec." % (t_prog_start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--parallel-phase", type=int, required=False, default=3)
    parser.add_argument("-t", "--threads", type=int, required=False, default=0)
    parser.add_argument("-p", "--parts", type=int, required=False, default=0)
    parser.add_argument("-d", "--decimation", type=int, required=False, default=0)
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0)
    parser.add_argument("-b", "--node-move-update-batch-size", type=int, required=False, default=1)
    parser.add_argument("-g", "--node-propose-batch-size", type=int, required=False, default=4)
    parser.add_argument("--sparse", type=int, required=False, default=0)
    parser.add_argument("-s", "--sort", type=int, required=False, default=0)
    parser.add_argument("-S", "--seed", type=int, required=False, default=-1)
    parser.add_argument("-m", "--merge-method", type=int, required=False, default=0)
    parser.add_argument("--mpi", action="store_true", default=False)
    parser.add_argument("input_filename", nargs="?", type=str, default="../../data/static/simulated_blockmodel_graph_500_nodes")

    # Debugging options
    parser.add_argument("--profile", type=str, required=False, default="")
    parser.add_argument("--pipe", type=int, required=False, default=0)
    parser.add_argument("--test-decimation", type=int, required=False, default=0)
    parser.add_argument("--predecimation", type=int, required=False, default=0)

    args = parser.parse_args()

    np.seterr(all='raise')
    args.debug = 0

    if args.verbose > 0:
        d = vars(args)
        args_sorted = sorted([i for i in d.items()])
        print("Arguments: {" + "".join(("%s : %s, " % (k,v) for k,v in args_sorted)) + "}\n")

    print("multiprocessing cpu count is %d" % mp.cpu_count())

    if args.seed != -1:
        numpy.random.seed(args.seed % 4294967295)

    if args.profile:
        import cProfile
        cProfile.run('do_main(args)', filename=args.profile)
    else:
        do_main(args)

    print("Block sum time = %s" % block_sum_time_cum)
