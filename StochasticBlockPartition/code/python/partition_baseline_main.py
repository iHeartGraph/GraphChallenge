from partition_baseline_support import *
import multiprocessing as mp
import multiprocessing.pool
from multiprocessing import Pool, Value, Semaphore, Manager, Queue, current_process
from functools import reduce
import pickle
import timeit
import os, sys, argparse
import time, struct
import traceback
import numpy.random
from compute_delta_entropy import compute_delta_entropy
import random
import shutil
import queue

compressed_threshold = 5000

def is_compressed(M):
    return not isinstance(M, np.ndarray)

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
    args = syms['args']

    return compute_best_block_merge(blocks, num_blocks, interblock_edge_count, block_partition, block_degrees, args.n_proposal, block_degrees_out, block_degrees_in, args)


def compute_best_block_merge(blocks, num_blocks, M, block_partition, block_degrees, n_proposal, block_degrees_out, block_degrees_in, args):
    best_overall_merge = [-1 for i in blocks]
    best_overall_delta_entropy = [np.Inf for i in blocks]
    n_proposals_evaluated = 0

    for current_block_idx,current_block in enumerate(blocks):
        if current_block is None:
            break

        # Index of non-zero block entries and their associated weights
        in_idx, in_weight = take_nonzero(M, current_block, 1, sort = False)
        out_idx, out_weight = take_nonzero(M, current_block, 0, sort = False)

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
                                                    use_sparse_alg = args.sparse_algorithm)

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

    global update_id, partition, M, block_degrees, block_degrees_out, block_degrees_in, mypid

    rank,start,stop,step = tup

    args = syms['args']
    lock = syms['lock']
    n_thread = syms['n_thread']

    results = syms['results']
    (results_proposal, results_delta_entropy, results_accept, results_propose_time_worker_ms) = syms['results']

    if args.pipe:
        pipe = syms['pipe']

    (num_blocks, out_neighbors, in_neighbors, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors) = syms['static_state']

    (update_id_shared, M_shared, partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared, block_modified_time_shared) = syms['nodal_move_state']

    lock.acquire()

    if update_id != update_id_shared.value:
        if update_id == -1:
            (partition, block_degrees, block_degrees_out, block_degrees_in) \
                = (shared_memory_to_private(i) for i in (partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared))

            M = M_shared.copy()

            # Ensure every worker has a different random seed.
            mypid = current_process().pid
            numpy.random.seed((mypid + int(timeit.default_timer() * 1e6)) % 4294967295)
        else:
            w = np.where(block_modified_time_shared > update_id)[0]
            M[w, :] = M_shared[w, :]
            M[:, w] = M_shared[:, w]

            block_degrees_in[w] = block_degrees_in_shared[w]
            block_degrees_out[w] = block_degrees_out_shared[w]
            block_degrees[w] = block_degrees_shared[w]
            partition[:] = partition_shared[:]

        update_id = update_id_shared.value

    lock.release()

    if args.verbose > 3:
        print("Rank %d pid %d start %d stop %d step %d" % (rank,mypid,start,stop,step))

    for current_node in range(start, stop, step):

        t0 = timeit.default_timer()

        res = propose_node_movement(current_node, partition, out_neighbors, in_neighbors,
                                    M, num_blocks, block_degrees, block_degrees_out, block_degrees_in,
                                    vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, args)

        t1 = timeit.default_timer()
        t_elapsed_ms = (t1 - t0) * 1e3
        (ni, current_block, proposal, delta_entropy, p_accept) = res
        accept = (np.random.uniform() <= p_accept)

        results_proposal[ni] = proposal
        results_delta_entropy[ni] = delta_entropy
        results_accept[ni] = accept
        results_propose_time_worker_ms[ni] = t_elapsed_ms

    if args.pipe:
        os.write(pipe[1],
                 rank.to_bytes(4, byteorder='little')
                 + update_id.to_bytes(4, byteorder='little')
                 + start.to_bytes(4, byteorder='little')
                 + stop.to_bytes(4, byteorder='little')
                 + step.to_bytes(4, byteorder='little'))
        return
    else:
        return rank,mypid,update_id,start,stop,step

def propose_node_movement_sparse_wrapper(tup):
    global update_id, partition, M, block_degrees, block_degrees_out, block_degrees_in, mypid, mypid_idx, worker_pids

    myrank,start,stop,step = tup

    args = syms['args']
    lock = syms['lock']
    n_thread = syms['n_thread']
    (results_proposal, results_delta_entropy, results_accept, results_propose_time_worker_ms) = syms['results']

    worker_pids = syms['worker_pids']
    pid_box = syms['pid_box']

    if args.pipe:
        pipe = syms['pipe']

    (num_blocks, out_neighbors, in_neighbors, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors) = syms['static_state']

    (update_id_shared, M_shared, partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared, block_modified_time_shared) = syms['nodal_move_state']


    if update_id == -1:
        mypid = current_process().pid    

        (partition, block_degrees, block_degrees_out, block_degrees_in) \
            = (shared_memory_to_private(i) for i in (partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared))

        mypid_idx = next(i for i,e in enumerate(worker_pids) if e == mypid)

        if 1:
            M = M_shared
        else:
            M = fast_sparse_array(M_shared.shape)
            for i in range(M.shape[0]):
                M_row = nonzero_dict(M_shared.take_dict(i, 0))
                M.set_axis_dict(i, 0, M_row)
                M_col = nonzero_dict(M_shared.take_dict(i, 1))
                M.set_axis_dict(i, 1, M_col)

        # Ensure every worker has a different random seed.
        numpy.random.seed((mypid + int(timeit.default_timer() * 1e6)) % 4294967295)
        update_id = 0
    else:
        q = pid_box[mypid_idx]

        while True:
            try:
                update = q.get(block=False)
                (rank,worker_pid,worker_update_id,ni,r,s) = update

                if args.verbose > 3:
                    print("Worker %d pid %d got an update %s" % (myrank,mypid,update))

                r = partition[ni]

                if args.verbose > 2:
                    print("Worker pid %d rank %d move remote %d from block %d to block %d." % (worker_pid,myrank, ni, r, s))

                blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[ni][:, 0]], return_inverse=True)
                count_out = np.bincount(inverse_idx_out, weights=out_neighbors[ni][:, 1]).astype(int)
                blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[ni][:, 0]], return_inverse=True)
                count_in = np.bincount(inverse_idx_in, weights=in_neighbors[ni][:, 1]).astype(int)
                self_edge_weight = np.sum(out_neighbors[ni][np.where(out_neighbors[ni][:, 0] == ni), 1])

                (new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col) = \
                                    compute_new_rows_cols_interblock_edge_count_matrix(
                                        M, r, s,
                                        blocks_out, count_out, blocks_in, count_in,
                                        self_edge_weight, agg_move = 0,
                                        use_sparse_alg = args.sparse_algorithm)

                block_degrees_out[r] = np.sum(new_M_r_row.values())
                block_degrees_out[s] = np.sum(new_M_s_row.values())
                block_degrees_in[r] = np.sum(new_M_r_col.values())
                block_degrees_in[s] = np.sum(new_M_s_col.values())

                block_degrees[s] = block_degrees_out[s] + block_degrees_in[s]
                block_degrees[r] = block_degrees_out[r] + block_degrees_in[r]

                partition, M = update_partition_single(partition, ni, s, M,
                                                       new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, args)

            except queue.Empty:
                break

    if args.verbose > 3:
        print("Propose node movement at worker %d" % (myrank,))

    for current_node in range(start, stop, step):
        t0 = timeit.default_timer()

        res = propose_node_movement(current_node, partition, out_neighbors, in_neighbors,
                                    M, num_blocks, block_degrees, block_degrees_out, block_degrees_in,
                                    vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, args)

        t1 = timeit.default_timer()
        t_elapsed_ms = (t1 - t0) * 1e3
        (ni, current_block, proposal, delta_entropy, p_accept) = res
        accept = (np.random.uniform() <= p_accept)

        results_proposal[ni] = proposal
        results_delta_entropy[ni] = delta_entropy
        results_accept[ni] = accept
        results_propose_time_worker_ms[ni] = t_elapsed_ms

        if accept:
            ni = current_node
            r = partition[current_node]
            s = proposal

            if args.verbose > 2:
                print("Worker %d move self %d from block %d to block %d." % (myrank, ni, r, s))

            blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[ni][:, 0]], return_inverse=True)
            count_out = np.bincount(inverse_idx_out, weights=out_neighbors[ni][:, 1]).astype(int)
            blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[ni][:, 0]], return_inverse=True)
            count_in = np.bincount(inverse_idx_in, weights=in_neighbors[ni][:, 1]).astype(int)
            self_edge_weight = np.sum(out_neighbors[ni][np.where(out_neighbors[ni][:, 0] == ni), 1])

            (new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col) = \
                                compute_new_rows_cols_interblock_edge_count_matrix(
                                    M, r, s,
                                    blocks_out, count_out, blocks_in, count_in,
                                    self_edge_weight, agg_move = 0,
                                    use_sparse_alg = args.sparse_algorithm)

            block_degrees_out[r] = np.sum(new_M_r_row.values())
            block_degrees_out[s] = np.sum(new_M_s_row.values())
            block_degrees_in[r] = np.sum(new_M_r_col.values())
            block_degrees_in[s] = np.sum(new_M_s_col.values())

            block_degrees[s] = block_degrees_out[s] + block_degrees_in[s]
            block_degrees[r] = block_degrees_out[r] + block_degrees_in[r]

            partition, M = update_partition_single(partition, ni, s, M,
                                                   new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, args)

            # Also send result to every worker.
            for i,q in enumerate(pid_box):
                if mypid != worker_pids[i] and worker_pids[i] != -1:
                    update = (myrank,mypid,update_id,ni,r,s)
                    if args.verbose > 3:
                        print("Put update from worker %d to queue %d: %s" % (myrank,mypid,update))
                    q.put(update)

    if args.pipe:
        os.write(pipe[1],
                 myrank.to_bytes(4, byteorder='little')
                 + update_id.to_bytes(4, byteorder='little')
                 + start.to_bytes(4, byteorder='little')
                 + stop.to_bytes(4, byteorder='little')
                 + step.to_bytes(4, byteorder='little'))
        return
    else:
        return myrank,mypid,update_id,start,stop,step


def propose_node_movement(current_node, partition, out_neighbors, in_neighbors, interblock_edge_count, num_blocks,
                          block_degrees, block_degrees_out, block_degrees_in,
                          vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, args):

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
                                                               self_edge_weight, agg_move = 0,
                                                               use_sparse_alg = args.sparse_algorithm)

        # compute new block degrees
        block_degrees_out_new, block_degrees_in_new, block_degrees_new = compute_new_block_degrees(
            current_block, proposal, block_degrees_out, block_degrees_in, block_degrees,
            num_out_neighbor_edges,
            num_in_neighbor_edges,
            num_neighbor_edges)

        # XXX
        use_sparse_data = is_compressed(interblock_edge_count)
        use_sparse_alg = use_sparse_data

        # compute the Hastings correction
        Hastings_correction = compute_Hastings_correction(blocks_out, count_out, blocks_in, count_in,
                                                          current_block,
                                                          proposal,
                                                          interblock_edge_count,
                                                          new_M_r_row,
                                                          new_M_r_col,
                                                          num_blocks, block_degrees,
                                                          block_degrees_new,
                                                          use_sparse_alg,
                                                          use_sparse_data)

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

    return current_node, current_block, proposal, delta_entropy, p_accept

def coo_to_flat(x, size):
    x_i, x_v = x
    f = np.zeros(size)
    f[x_i] = x_v
    return f

def update_partition_single(b, ni, s, M, M_r_row, M_s_row, M_r_col, M_s_col, args):
    r = b[ni]
    b[ni] = s

    if is_compressed(M):
        M.set_axis_dict(r, 0, M_r_row)
        M.set_axis_dict(s, 0, M_s_row)
        M.set_axis_dict(r, 1, M_r_col)
        M.set_axis_dict(s, 1, M_s_col)
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

def shared_memory_to_private(z):
    x = np.empty(z.shape, dtype=z.dtype)
    x[:] = z
    return x

def nodal_moves_sequential(batch_size, max_num_nodal_itr, delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy_cur, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose, args):
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

        t_merge_start = timeit.default_timer()

        for i in L:
            t_propose_start = timeit.default_timer()
            movement = propose_node_movement(i, partition, out_neighbors, in_neighbors, M, num_blocks,
                                             block_degrees, block_degrees_out, block_degrees_in,
                                             vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, args)

            t_propose_end = timeit.default_timer()
            propose_time = (t_propose_end - t_propose_start)
            propose_time_cum += propose_time

            proposal_cnt += 1

            (ni, r, s, delta_entropy, p_accept) = movement
            accept = (np.random.uniform() <= p_accept)

            if not accept:
                continue

            t_update_partition_beg = timeit.default_timer()

            total_num_nodal_moves_itr += 1
            num_nodal_moves += 1
            itr_delta_entropy[itr] += delta_entropy

            r = partition[ni]

            if verbose > 2:
                print("Move %5d from block %5d to block %5d." % (ni, r, s))

            blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[ni][:, 0]], return_inverse=True)
            count_out = np.bincount(inverse_idx_out, weights=out_neighbors[ni][:, 1]).astype(int)
            blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[ni][:, 0]], return_inverse=True)
            count_in = np.bincount(inverse_idx_in, weights=in_neighbors[ni][:, 1]).astype(int)
            self_edge_weight = np.sum(out_neighbors[ni][np.where(out_neighbors[ni][:, 0] == ni), 1])

            (new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col) = \
                                compute_new_rows_cols_interblock_edge_count_matrix(M, r, s,
                                                                                   blocks_out, count_out, blocks_in, count_in,
                                                                                   self_edge_weight, agg_move = 0,
                                                                                   use_sparse_alg = args.sparse_algorithm)

            partition, M = update_partition_single(partition, ni, s, M,
                                                   new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, args)

            t_update_partition_end = timeit.default_timer()

            update_partition_time_ms_cum += (t_update_partition_end - t_update_partition_beg) * 1e3

            btime = timeit.default_timer()

            if not is_compressed(M):
                block_degrees_out[r] = np.sum(new_M_r_row)
                block_degrees_out[s] = np.sum(new_M_s_row)
                block_degrees_in[r] = np.sum(new_M_r_col)
                block_degrees_in[s] = np.sum(new_M_s_col)
            else:
                block_degrees_out[r] = np.sum(new_M_r_row.values())
                block_degrees_out[s] = np.sum(new_M_s_row.values())
                block_degrees_in[r] = np.sum(new_M_r_col.values())
                block_degrees_in[s] = np.sum(new_M_s_col.values())

            block_degrees[s] = block_degrees_out[s] + block_degrees_in[s]
            block_degrees[r] = block_degrees_out[r] + block_degrees_in[r]

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

    return total_num_nodal_moves_itr,M


def nodal_moves_parallel(n_thread, batch_size, max_num_nodal_itr, delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy_cur, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose, args):
    global syms, block_sum_time_cum

    total_num_nodal_moves_itr = 0
    itr_delta_entropy = np.zeros(max_num_nodal_itr)

    lock = mp.Lock()

    modified = np.zeros(M.shape[0], dtype=bool)
    block_modified_time_shared = shared_memory_empty(modified.shape)
    block_modified_time_shared[:] = 0

    update_id_shared = Value('i', 0)

    last_purge = -1
    worker_progress = np.empty(n_thread, dtype=int)
    worker_progress[:] = last_purge

    (partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared) \
        = (shared_memory_copy(i) for i in (partition, block_degrees, block_degrees_out, block_degrees_in, ))

    if is_compressed(M):
        # Do not do a shared memory copy because nodal updates will arrive via message-passing instead of a shared array.
        M_shared = M
        # Mailboxes for messages from parent to each worker.
        pid_box = [Queue() for i in range(2 * n_thread)]
    else:
        M_shared = shared_memory_copy(M)

    static_state = (num_blocks, out_neighbors, in_neighbors, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors)

    shape = partition.shape
    results_proposal = shared_memory_empty(shape)
    results_delta_entropy = shared_memory_empty(shape, dtype='float')
    results_accept = shared_memory_empty(shape, dtype='bool')
    results_propose_time_worker_ms = shared_memory_empty(shape, dtype='float')
    # Sometimes a worker is mistakenly "active"
    worker_pids = shared_memory_empty(shape=(2 * n_thread,))
    worker_pids[:] = -1

    syms = {}
    syms['results'] = (results_proposal, results_delta_entropy, results_accept, results_propose_time_worker_ms)
    syms['lock'] = lock
    syms['static_state'] = static_state
    syms['n_thread'] = n_thread
    syms['nodal_move_state'] = (update_id_shared, M_shared, partition_shared, block_degrees_shared, block_degrees_out_shared, block_degrees_in_shared, block_modified_time_shared)
    syms['args'] = args

    if is_compressed(M):
        syms['pid_box'] = pid_box
        syms['worker_pids'] = worker_pids

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

    if is_compressed(M):
        worker_pids[:] = -1
        active_children = multiprocessing.active_children()
        for i,e in enumerate(active_children):
            worker_pids[i] = e.pid

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

        group_size = args.node_propose_batch_size
        chunks = [((i // group_size) % n_thread, i, min(i+group_size, N), 1) for i in range(0,N,group_size)]

        if is_compressed(M):
            movements = pool.imap_unordered(propose_node_movement_sparse_wrapper, chunks)
        else:
            movements = pool.imap_unordered(propose_node_movement_wrapper, chunks)

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
                buf = os.read(pipe[0], 4*5)
                rank,worker_pid,worker_update_id,start,stop,step = struct.unpack('<iiiii', buf)
            else:
                rank,worker_pid,worker_update_id,start,stop,step = movements.next()
    
            worker_progress[rank] = worker_update_id

            for ni in range(start,stop,step):
                useless_time_beg = timeit.default_timer()

                s = results_proposal[ni]
                delta_entropy = results_delta_entropy[ni]
                accept = results_accept[ni]
                propose_time_worker_ms = results_propose_time_worker_ms[ni]

                if verbose > 3 and accept:
                    print("Parent accepted %d result from worker %d to move index %d from block %d to block %d" % (accept,rank,ni,partition[ni],s))

                proposal_cnt += 1
                propose_time_workers_ms_cum += propose_time_worker_ms

                useless_time_end = timeit.default_timer()
                t_useless += useless_time_end - useless_time_beg

                if accept:
                    t_update_partition_beg = timeit.default_timer()

                    total_num_nodal_moves_itr += 1
                    num_nodal_moves += 1
                    itr_delta_entropy[itr] += delta_entropy

                    r = partition[ni]
                    modified[r] = True
                    modified[s] = True

                    if verbose > 2:
                        print("Parent move %d from block %d to block %d." % (ni, r, s))

                    if 1:
                        blocks_out, inverse_idx_out = np.unique(partition[out_neighbors[ni][:, 0]], return_inverse=True)
                        count_out = np.bincount(inverse_idx_out, weights=out_neighbors[ni][:, 1]).astype(int)
                        blocks_in, inverse_idx_in = np.unique(partition[in_neighbors[ni][:, 0]], return_inverse=True)
                        count_in = np.bincount(inverse_idx_in, weights=in_neighbors[ni][:, 1]).astype(int)
                        self_edge_weight = np.sum(out_neighbors[ni][np.where(out_neighbors[ni][:, 0] == ni), 1])


                        (new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col) = \
                                            compute_new_rows_cols_interblock_edge_count_matrix(
                                                M, r, s,
                                                blocks_out, count_out, blocks_in, count_in,
                                                self_edge_weight, agg_move = 0,
                                                use_sparse_alg = args.sparse_algorithm)

                        if not is_compressed(M):
                            block_degrees_out[r] = np.sum(new_M_r_row)
                            block_degrees_out[s] = np.sum(new_M_s_row)
                            block_degrees_in[r] = np.sum(new_M_r_col)
                            block_degrees_in[s] = np.sum(new_M_s_col)
                        else:
                            block_degrees_out[r] = np.sum(new_M_r_row.values())
                            block_degrees_out[s] = np.sum(new_M_s_row.values())
                            block_degrees_in[r] = np.sum(new_M_r_col.values())
                            block_degrees_in[s] = np.sum(new_M_s_col.values())

                        block_degrees[s] = block_degrees_out[s] + block_degrees_in[s]
                        block_degrees[r] = block_degrees_out[r] + block_degrees_in[r]


                        partition, M = update_partition_single(partition, ni, s, M,
                                                               new_M_r_row, new_M_s_row, new_M_r_col, new_M_s_col, args)

                        t_update_partition_end = timeit.default_timer()

                        update_partition_time_ms_cum += (t_update_partition_end - t_update_partition_beg) * 1e3

                if 0:
                    print("num_nodal_moves, next_batch_cnt, proposal_cnt, N",num_nodal_moves,next_batch_cnt,proposal_cnt, N)

                if num_nodal_moves >= next_batch_cnt or proposal_cnt == N:
                    btime = timeit.default_timer()
                    where_modified = np.where(modified)[0]
                    next_batch_cnt = num_nodal_moves + batch_size

                    btime_end = timeit.default_timer()
                    block_sum_time += btime_end - btime
                    block_sum_time_cum += btime_end - btime

                    update_beg = timeit.default_timer()
                    update_id_cnt += 1

                    block = (proposal_cnt == N)

                    if lock.acquire(block=block):
                        if not is_compressed(M):
                            M_shared[where_modified, :] = M[where_modified, :]
                            M_shared[:, where_modified] = M[:, where_modified]
                        else:
                            pass

                        if args.verbose > 2:
                            print("Modified fraction of id %d is %s" % (update_id_cnt, len(where_modified) / float(M.shape[0])))
                            print("Worker progress is %s" % (worker_progress))

                        block_degrees_in_shared[where_modified] = block_degrees_in[where_modified]
                        block_degrees_out_shared[where_modified] = block_degrees_out[where_modified]
                        block_degrees_shared[where_modified] = block_degrees[where_modified]
                        block_modified_time_shared[where_modified] = update_id_cnt

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


def entropy_for_block_count(num_blocks, num_target_blocks, delta_entropy_threshold, M, block_degrees, block_degrees_out, block_degrees_in, out_neighbors, in_neighbors, N, E, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, partition, args, verbose = False):
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
        syms['args'] = args

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
                                       block_partition, block_degrees, args.n_proposal, block_degrees_out, block_degrees_in, args)

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

        if args.sparse == 2:
            if num_blocks >= compressed_threshold:
                use_compressed = 1
            else:
                use_compressed = 0
            print("Use num_blocks is %d compressed is %d" % (num_blocks,use_compressed))
        else:
            use_compressed = args.sparse_data

        M_t, block_degrees_out_t, block_degrees_in_t, block_degrees_t = \
                initialize_edge_counts(out_neighbors,
                                       num_blocks_t,
                                       partition_t,
                                       use_compressed)

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
        # Can be zero when there are some nodes with no edges and so reducing the block count
        # does not change the entropy.
        if (S[2] - S[1] - (S[1] - S[0])) != 0:
            extrapolated_newton = num_target_blocks[1] - 0.5 * (S[2] - S[0]) / (S[2] - S[1] - (S[1] - S[0]))
        else:
            extrapolated_newton = 0.0

        if verbose:
            print("Stopping criterion not found at %s blocks extrapolate to %s blocks derivative %s at time %4.4f." % (num_target_blocks[1], extrapolated_newton, dS_dn, timeit.default_timer() - t_prog_start))

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
        total_num_nodal_moves_itr = nodal_moves_parallel(n_thread, batch_size, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose, args)
    else:
        total_num_nodal_moves_itr,M = nodal_moves_sequential(batch_size, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose, args)

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


def load_graph_parts(input_filename, args):
    true_partition_available = True
    if not os.path.isfile(input_filename + '.tsv') and not os.path.isfile(input_filename + '_1.tsv'):
            print("File doesn't exist: '{}'!".format(input_filename))
            sys.exit(1)

    if args.parts >= 1:
            print('\nLoading partition 1 of {} ({}) ...'.format(args.parts, input_filename + "_1.tsv"))
            out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available, strm_piece_num=1)
            for part in range(2, args.parts + 1):
                    print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
                    out_neighbors, in_neighbors, N, E = load_graph(input_filename, load_true_partition=False, strm_piece_num=part, out_neighbors=out_neighbors, in_neighbors=in_neighbors)
    else:
            out_neighbors, in_neighbors, N, E, true_partition = load_graph(input_filename, load_true_partition=true_partition_available)
    return out_neighbors, in_neighbors, N, E, true_partition


def find_optimal_partition(out_neighbors, in_neighbors, N, E, args, stop_at_bracket = False, verbose = 0, alg_state = None, num_block_reduction_rate = 0.50, min_number_blocks = 0):

    if verbose > -1:
        print('Number of nodes %d edges %d ' % (N,E))

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

    if not alg_state:
        # initialize by putting each node in its own block (N blocks)
        num_blocks = N
        partition = np.arange(num_blocks, dtype=int)

        if args.sparse == 2:
            if num_blocks >= compressed_threshold:
                use_compressed = 1
            else:
                use_compressed = 0
            print("Use num_blocks is %d compressed is %d" % (num_blocks,use_compressed))
        else:
            use_compressed = args.sparse_data

        # initialize edge counts and block degrees
        interblock_edge_count, block_degrees_out, block_degrees_in, block_degrees \
            = initialize_edge_counts(out_neighbors,
                                     num_blocks,
                                     partition,
                                     use_compressed)
        # initialize items before iterations to find the partition with the optimal number of blocks
        hist, graph_object = initialize_partition_variables()

        initial_num_block_reduction_rate = args.initial_block_reduction_rate

        num_blocks_to_merge = int(num_blocks * initial_num_block_reduction_rate)
        golden_ratio_bracket_established = False
        delta_entropy_threshold = delta_entropy_threshold1
        n_proposals_evaluated = 0
        total_num_nodal_moves = 0
    else:
        # resume search from a previous partition state
        (hist, num_blocks, overall_entropy, partition, interblock_edge_count,block_degrees_out,block_degrees_in,block_degrees,golden_ratio_bracket_established,delta_entropy_threshold,num_blocks_to_merge,optimal_num_blocks_found,n_proposals_evaluated,total_num_nodal_moves) = alg_state

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
                                   partition, args, verbose)

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


        if verbose:
            print('Overall entropy: {}'.format(old_overall_entropy))
            print('Number of blocks: {}'.format(old_num_blocks))
            if optimal_num_blocks_found:
                print('\nOptimal partition found with {} blocks'.format(num_blocks))
            print('Proposals evaluated: {}'.format(n_proposals_evaluated))
            print('Overall nodal moves: {}'.format(total_num_nodal_moves))

        if np.all(np.isfinite(old_overall_entropy)):
            delta_entropy_threshold = delta_entropy_threshold2
            if not golden_ratio_bracket_established:
                golden_ratio_bracket_established = True
                print("Golden ratio found at blocks %s at time %4.4f entropy %s" % (old_num_blocks, timeit.default_timer() - t_prog_start, old_overall_entropy))

            if stop_at_bracket:
                break

        if num_blocks <= min_number_blocks:
            break


    alg_state = (hist,num_blocks,overall_entropy,partition,interblock_edge_count,block_degrees_out,block_degrees_in,block_degrees,golden_ratio_bracket_established,delta_entropy_threshold,num_blocks_to_merge,optimal_num_blocks_found,n_proposals_evaluated,total_num_nodal_moves)

    return alg_state, partition

def find_optimal_partition_wrapper(tup):
    args = syms['args']
    args.threads = max(1, args.threads // args.decimation)
    out_neighbors, in_neighbors, N, E, true_partition = tup
    return find_optimal_partition(out_neighbors, in_neighbors, N, E, args, stop_at_bracket = True, verbose = min(0, args.verbose - 1))


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


def merge_partitions(partitions, stop_pieces, out_neighbors, verbose, use_sparse_alg, use_sparse_data):
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
            = initialize_edge_counts(out_neighbors, B, partition, use_sparse_data)

        if verbose > 2:
            print("M.shape = %s, M = \n%s" % (str(M.shape),M))

        next_partitions = []
        for i in range(0, pieces, 2):
            print("Merge piece %d and %d into %d" % (i, i + 1, i // 2))
            partitions[i],_ = merge_two_partitions(M, block_degrees_out, block_degrees_out, block_degrees_out,
                                                   partitions[i], partitions[i + 1],
                                                   partition_offsets[i], partition_offsets[i + 1],
                                                   Bs[i], Bs[i + 1],
                                                   verbose,
                                                   use_sparse_alg,
                                                   use_sparse_data)
            next_partitions.append(np.concatenate((partitions[i], partitions[i+1])))

        partitions = next_partitions
        pieces //= 2

    return partitions



def merge_two_partitions(M, block_degrees_out, block_degrees_in, block_degrees, partition0, partition1, partition_offset_0, partition_offset_1, B0, B1, verbose, use_sparse_alg, use_sparse_data):
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
        in_idx, in_weight = take_nonzero(M, current_block, 1, sort = False)
        out_idx, out_weight = take_nonzero(M, current_block, 0, sort = False)

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
                                                                     M[current_block, current_block], agg_move = 1,
                                                                     use_sparse_alg = use_sparse_alg)

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

    if verbose > 2:
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
                                                    num_blocks_to_merge, verbose=(verbose > 2))

    return partition, num_blocks

def naive_streaming(args):
    input_filename = args.input_filename
    # Emerging edge piece by piece streaming.
    # The assumption is that unlike parallel decimation, where a static graph is cut into
    # multiple subgraphs which do not have the same nodes, the same node set is potentially
    # present in each piece.
    #
    out_neighbors,in_neighbors = None,None
    t_all_parts = 0.0

    for part in range(1, args.parts + 1):
        print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))
        t_part = 0.0

        if part == 1:
            out_neighbors, in_neighbors, N, E, true_partition = \
                    load_graph(input_filename,
                               load_true_partition=1,
                               strm_piece_num=part,
                               out_neighbors=None,
                               in_neighbors=None)
        else:
            out_neighbors, in_neighbors, N, E = \
                    load_graph(input_filename,
                               load_true_partition=0,
                               strm_piece_num=part,
                               out_neighbors=out_neighbors,
                               in_neighbors=in_neighbors)

        # Run to ground.
        print('Running partition for part %d N %d E %d' % (part,N,E))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 0, alg_state = None)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        t_all_parts += t_part

        if part == args.parts:
            print('Evaluate final partition.')
        else:
            print('Evaluate part %d' % part)

        precision,recall = evaluate_partition(true_partition, partition)
        print('Elapsed compute time for part %d is %f cumulative %f precision %f recall %f' % (part,t_part,t_all_parts,precision,recall))

    return t_all_parts


def copy_alg_state(alg_state):
    # Create a deep copy of algorithmic state.
    (hist, num_blocks, overall_entropy, partition, interblock_edge_count,block_degrees_out,block_degrees_in,block_degrees,golden_ratio_bracket_established,delta_entropy_threshold,num_blocks_to_merge,optimal_num_blocks_found,n_proposals_evaluated,total_num_nodal_moves) = alg_state

    (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) = hist

    hist_copy = tuple((i.copy() for i in hist))
    try:
        num_blocks_copy = num_blocks.copy()
    except AttributeError:
        num_blocks_copy = num_blocks
    overall_entropy_copy = overall_entropy.copy()
    partition_copy = partition.copy()
    interblock_edge_count_copy = interblock_edge_count.copy()
    block_degrees_out_copy = block_degrees_out.copy()
    block_degrees_in_copy = block_degrees_in.copy()
    block_degrees_copy = block_degrees.copy()
    golden_ratio_bracket_established_copy = golden_ratio_bracket_established # bool
    delta_entropy_threshold_copy = delta_entropy_threshold # float
    num_blocks_to_merge_copy = num_blocks_to_merge # int
    optimal_num_blocks_found_copy = optimal_num_blocks_found # bool
    n_proposals_evaluated_copy = n_proposals_evaluated # int
    total_num_nodal_moves_copy = total_num_nodal_moves # int


    alg_state_copy = (hist_copy, num_blocks_copy, overall_entropy_copy, partition_copy, interblock_edge_count_copy, block_degrees_out_copy, block_degrees_in_copy, block_degrees_copy, golden_ratio_bracket_established_copy, delta_entropy_threshold_copy, num_blocks_to_merge_copy, optimal_num_blocks_found_copy, n_proposals_evaluated_copy, total_num_nodal_moves_copy)

    return alg_state_copy



def incremental_streaming(args):
    input_filename = args.input_filename
    # Emerging edge piece by piece streaming.
    # The assumption is that unlike parallel decimation, where a static graph is cut into
    # multiple subgraphs which do not have the same nodes, the same node set is potentially
    # present in each piece.
    #
    out_neighbors,in_neighbors,alg_state = None,None,None
    t_all_parts = 0.0

    for part in range(1, args.parts + 1):
        t_part = 0.0

        if part == 1:
            print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))

            out_neighbors, in_neighbors, N, E, true_partition = \
                    load_graph(input_filename,
                               load_true_partition=1,
                               strm_piece_num=part,
                               out_neighbors=None,
                               in_neighbors=None)
            min_number_blocks = N / 2
        else:
            # Load true_partition here so the sizes of the arrays all equal N.
            if alg_state:
                print('Loading partition {} of {} ({}) ...'.format(part, args.parts, input_filename + "_" + str(part) + ".tsv"))

                out_neighbors, in_neighbors, N, E, alg_state,t_compute = \
                                                load_graph(input_filename,
                                                           load_true_partition=1,
                                                           strm_piece_num=part,
                                                           out_neighbors=out_neighbors,
                                                           in_neighbors=in_neighbors,
                                                           alg_state = alg_state)
                t_part += t_compute
                print("Intermediate load_graph compute time for part %d is %f" % (part,t_compute))
                t0 = timeit.default_timer()
                hist = alg_state[0]
                (old_partition, old_interblock_edge_count, old_block_degrees, old_block_degrees_out, old_block_degrees_in, old_overall_entropy, old_num_blocks) = hist

                print("Incrementally updated alg_state for part %d" %(part))
                print('New Overall entropy: {}'.format(old_overall_entropy))
                print('New Number of blocks: {}'.format(old_num_blocks))
                print("")

                verbose = 1
                n_thread = args.threads
                batch_size = args.node_move_update_batch_size
                vertex_num_in_neighbor_edges = np.empty(N, dtype=int)
                vertex_num_out_neighbor_edges = np.empty(N, dtype=int)
                vertex_num_neighbor_edges = np.empty(N, dtype=int)
                vertex_neighbors = [np.concatenate((out_neighbors[i], in_neighbors[i])) for i in range(N)]

                for i in range(N):
                    vertex_num_out_neighbor_edges[i] = sum(out_neighbors[i][:,1])
                    vertex_num_in_neighbor_edges[i] = sum(in_neighbors[i][:,1])
                    vertex_num_neighbor_edges[i] = vertex_num_out_neighbor_edges[i] + vertex_num_in_neighbor_edges[i]
                #delta_entropy_threshold = delta_entropy_threshold1 = 5e-4
                delta_entropy_threshold = 1e-4

                for j in [0,2,1]:
                    if old_interblock_edge_count[j] == []:
                        continue

                    print("Updating previous state in bracket history.")

                    M_old = old_interblock_edge_count[j].copy()
                    M = old_interblock_edge_count[j]
                    partition = old_partition[j]
                    block_degrees_out = old_block_degrees_out[j]
                    block_degrees_in = old_block_degrees_in[j]
                    block_degrees = old_block_degrees[j]
                    num_blocks = old_num_blocks[j]
                    overall_entropy = old_overall_entropy[j]

                    total_num_nodal_moves_itr = nodal_moves_parallel(n_thread, batch_size, args.max_num_nodal_itr, args.delta_entropy_moving_avg_window, delta_entropy_threshold, overall_entropy, partition, M, block_degrees_out, block_degrees_in, block_degrees, num_blocks, out_neighbors, in_neighbors, N, vertex_num_out_neighbor_edges, vertex_num_in_neighbor_edges, vertex_num_neighbor_edges, vertex_neighbors, verbose, args)

                t1 = timeit.default_timer()
                print("Intermediate nodal move time for part %d is %f" % (part,(t1-t0)))
                t_part += (t1 - t0)
            else:
                # We are not doing partitioning yet. Just wait.
                out_neighbors, in_neighbors, N, E, true_partition = \
                                                load_graph(input_filename,
                                                           load_true_partition=1,
                                                           strm_piece_num=part,
                                                           out_neighbors=out_neighbors,
                                                           in_neighbors=in_neighbors,
                                                           alg_state = None)

            print("Loaded piece %d N %d E %d" % (part,N,E))
            min_number_blocks = int(min_number_blocks / 2)

        print('Running partition for part %d N %d E %d and min_number_blocks %d' % (part,N,E,min_number_blocks))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition,alg_state = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 1, alg_state = alg_state, min_number_blocks = min_number_blocks)
        min_number_blocks /= 2

        alg_state_copy = copy_alg_state(alg_state)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        print("Intermediate partition until save point for part %d is %f" % (part,(t1-t0)))

        t0 = timeit.default_timer()
        t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 0, alg_state = alg_state_copy, min_number_blocks = 5)
        t1 = timeit.default_timer()
        t_part += (t1 - t0)
        print("Intermediate partition until completion for part %d is %f" % (part,(t1-t0)))

        print('Evaluate part %d' % (part))
        precision,recall = evaluate_partition(true_partition, partition)

        t_all_parts += t_part
        print('Elapsed compute time for part %d is %f cumulative %f precision %f recall %f' % (part,t_part,t_all_parts,precision,recall))

    return t_all_parts

def do_main(args):
    global syms, t_prog_start

    if args.sparse:
        args.sparse_algorithm = 1
        args.sparse_data = 1
    elif args.sparse_data:
        args.sparse_algorithm = 1

    t_prog_start = timeit.default_timer()

    if args.verbose > 0:
        print("Program start at %s sec." % (t_prog_start))
        print("Started: " + time.strftime("%a %b %d %Y %H:%M:%S %Z"))
        print("Python version: " + sys.version)
        d = vars(args)
        args_sorted = sorted([i for i in d.items()])
        print("Arguments: {" + "".join(("%s : %s, " % (k,v) for k,v in args_sorted)) + "}\n")

    np.seterr(all='raise')

    if args.seed != 0:
        numpy.random.seed(args.seed % 4294967295)
    else:
        numpy.random.seed((os.getpid() + int(timeit.default_timer() * 1e6)) % 4294967295)

    input_filename = args.input_filename
    args.visualize_graph = False  # whether to plot the graph layout colored with intermediate partitions

    try:
        cols,lines = shutil.get_terminal_size()
        np.set_printoptions(linewidth=cols)
    except AttributeError:
        pass

    if args.parts <= 1:
        out_neighbors, in_neighbors, N, E, true_partition = load_graph_parts(input_filename, args)


        if not args.test_resume:
            t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args)
        else:
            print("")
            print("Test stop functionality.")
            print("")
            t_elapsed_partition,partition,alg_state = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 1, min_number_blocks = 0)

            print("")
            print("Resume bracket search.")
            print("")

            t_elapsed_partition,partition = partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket = 0, alg_state = alg_state)


        precision,recall = evaluate_partition(true_partition, partition)
        return t_elapsed_partition,precision,recall
    else:
        if args.naive_streaming:
            t_compute = naive_streaming(args)
        else:
            t_compute = incremental_streaming(args)
        return t_compute
    return


def partition_static_graph(out_neighbors, in_neighbors, N, E, true_partition, args, stop_at_bracket=0, alg_state=None, min_number_blocks=0):
    global syms, t_prog_start

    if args.verbose > 1:
        from collections import Counter
        print("Overall true partition statistics:")
        print("[" + "".join(("%5d : %3d, " % (i,e,) for i,e in sorted([(e,i) for (i,e) in Counter(true_partition).items()]))) + "]\n")


    if args.predecimation > 1:
        out_neighbors, in_neighbors, N, E, true_partition = \
                                decimate_graph(out_neighbors, in_neighbors, true_partition,
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
        alg_state, M = find_optimal_partition(out_neighbors_piece, in_neighbors_piece, N_piece, E_piece, args, stop_at_bracket = False, verbose = args.verbose)
        t_prog_end = timeit.default_timer()

        partition = alg_state[0][0][1]

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


        syms = {}
        syms['args'] = args
        pool = NonDaemonicPool(decimation)

        results = pool.map(find_optimal_partition_wrapper, pieces)
        partition_brackets,Ms = (list(i) for i in zip(*results))

        pool.close()
        partitions = [partition_brackets[i][0] for i in range(decimation)]
    else:
        decimation = 1
        t_prog_start = timeit.default_timer()

        alg_state, M_bracket = find_optimal_partition(out_neighbors, in_neighbors, N, E, \
                                    args, stop_at_bracket = stop_at_bracket, verbose = args.verbose, \
                                    alg_state = alg_state, num_block_reduction_rate = 0.50, min_number_blocks=min_number_blocks)

        partition = alg_state[0][0][1]
        t_prog_end = timeit.default_timer()

        if args.test_decimation > 0:
            decimation = args.test_decimation
            true_partitions = [true_partition[i::decimation] for i in range(decimation)]
            partitions = [partition[i::decimation] for i in range(decimation)]



    # Either multiprocess pool or MPI results need final merging.
    if decimation > 1:
        if args.verbose > 1:
            for i in range(decimation):
                print("")
                print("Evaluate decimated subgraph %d:" % i)
                evaluate_partition(true_partitions[i], partitions[i])

        t_decimation_merge_start = timeit.default_timer()


        # Merge all pieces into a smaller number.
        partitions = merge_partitions(partitions,
                                      4, out_neighbors, args.verbose, args.sparse_data, args.sparse_algorithm)

        # Merge piece into  big partition and then merge down.
        Bs = [max(i) + 1 for i in partitions]
        partition = np.zeros(N, dtype=int)
        partition_offsets = np.zeros(len(partitions), dtype=int)
        partition_offsets[1:] = np.cumsum(Bs)[:-1]

        partition = np.concatenate([partitions[i] + partition_offsets[i] for i in range(len(partitions))])

        t_decimation_merge_end = timeit.default_timer()
        print("Decimation merge time is %3.5f" % (t_decimation_merge_end - t_decimation_merge_start))

        t_final_partition_search_start = timeit.default_timer()

        partition_bracket, M_bracket = find_optimal_partition(out_neighbors, in_neighbors, N, E, args,
                                                              stop_at_bracket = False, verbose = args.verbose,
                                                              partition_bracket = [partition],
                                                              num_block_reduction_rate = 0.35)

        t_final_partition_search_end = timeit.default_timer()

        partition = partition_bracket[1]
        t_prog_end = timeit.default_timer()
        print("Final partition search took %3.5f" % (t_final_partition_search_end - t_final_partition_search_start))

    t_elapsed_partition = t_prog_end - t_prog_start
    print('\nGraph partition took %.4f seconds' % (t_elapsed_partition))

    if stop_at_bracket:
        return t_elapsed_partition,partition,alg_state
    else:
        return t_elapsed_partition,partition

# See: https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)

block_sum_time_cum = 0

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
    parser.add_argument("--sparse-algorithm", type=int, required=False, default=0)
    parser.add_argument("--sparse-data", type=int, required=False, default=0)
    parser.add_argument("-s", "--sort", type=int, required=False, default=0)
    parser.add_argument("-S", "--seed", type=int, required=False, default=-1)
    parser.add_argument("-m", "--merge-method", type=int, required=False, default=0)
    parser.add_argument("--mpi", action="store_true", default=False)
    parser.add_argument("input_filename", nargs="?", type=str, default="../../data/static/simulated_blockmodel_graph_500_nodes")

    # Debugging options
    parser.add_argument("--initial-block-reduction-rate", type=float, required=False, default=0.50)
    parser.add_argument("--profile", type=str, required=False, default="")
    parser.add_argument("--pipe", type=int, required=False, default=0)
    parser.add_argument("--test-decimation", type=int, required=False, default=0)
    parser.add_argument("--predecimation", type=int, required=False, default=0)
    parser.add_argument("--debug", type=int, required=False, default=0)
    parser.add_argument("--test-resume", type=int, required=False, default=0)
    parser.add_argument("--naive-streaming", type=int, required=False, default=0)

    args = parser.parse_args()

    if args.debug:
        sys.excepthook = info

    if args.profile:
        import cProfile
        cProfile.run('do_main(args)', filename=args.profile)
    else:
        do_main(args)

    print("Block sum time = %s" % block_sum_time_cum)
