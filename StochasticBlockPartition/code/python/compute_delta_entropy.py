import numpy as np
import pickle

def entropy_row_calc(x, y, c, ignore=[]):
    if 1:
        mask = (x != 0)
        if ignore:
            mask[ignore[0]] = 0
            mask[ignore[1]] = 0
    else:
        mask = x.nonzero()[0]
        mask = [i for i in (set(x.nonzero()[0]) - set(ignore))]

    xm = x[mask]
    ym = y[mask]
    return np.sum(xm * (np.log(xm) - np.log(ym * c)))

def compute_delta_entropy_alt(r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out, d_in, d_out_new, d_in_new):
    """Compute change in entropy under the proposal with a faster method."""
    M_r_t1 = M[r, :]
    M_s_t1 = M[s, :]
    M_t2_r = M[:, r]
    M_t2_s = M[:, s]

    # remove r and s from the cols to avoid double counting
    ignore = (int(r), int(s))

    # only keep non-zero entries to avoid unnecessary computation
    d0 = entropy_row_calc(M_r_row.ravel(), d_in_new, d_out_new[r])
    d1 = entropy_row_calc(M_s_row.ravel(), d_in_new, d_out_new[s])
    d2 = entropy_row_calc(M_r_col.ravel(), d_out_new, d_in_new[r], ignore)
    d3 = entropy_row_calc(M_s_col.ravel(), d_out_new, d_in_new[s], ignore)
    d4 = entropy_row_calc(M_r_t1.ravel(),  d_in, d_out[r])
    d5 = entropy_row_calc(M_s_t1.ravel(),  d_in, d_out[s])
    d6 = entropy_row_calc(M_t2_r.ravel(),  d_out, d_in[r], ignore)
    d7 = entropy_row_calc(M_t2_s.ravel(),  d_out, d_in[s], ignore)
    return -d0 - d1 - d2 - d3 + d4 + d5 + d6 + d7

def compute_delta_entropy_orig(r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out, d_in, d_out_new, d_in_new):
    """Compute change in entropy under the proposal. Reduced entropy means the proposed block is better than the current block.

        Parameters
        ----------
        r : int
                    current block assignment for the node under consideration
        s : int
                    proposed block assignment for the node under consideration
        M : ndarray (int), shape = (#blocks, #blocks)
                    edge count matrix between all the blocks.
        M_r_row : ndarray (int)
                    the current block row of the new edge count matrix under proposal
        M_s_row : ndarray (int)
                    the proposed block row of the new edge count matrix under proposal
        M_r_col : ndarray (int)
                    the current block col of the new edge count matrix under proposal
        M_s_col : ndarray  (int)
                    the proposed block col of the new edge count matrix under proposal
        d_out : ndarray (int)
                    the current out degree of each block
        d_in : ndarray (int)
                    the current in degree of each block
        d_out_new : ndarray (int)
                    the new out degree of each block under proposal
        d_in_new : ndarray (int)
                    the new in degree of each block under proposal

        Returns
        -------
        delta_entropy : float
                    entropy under the proposal minus the current entropy

        Notes
        -----
        - M^-: current edge count matrix between the blocks
        - M^+: new edge count matrix under the proposal
        - d^-_{t, in}: current in degree of block t
        - d^-_{t, out}: current out degree of block t
        - d^+_{t, in}: new in degree of block t under the proposal
        - d^+_{t, out}: new out degree of block t under the proposal
        
        The difference in entropy is computed as:
        
        \dot{S} = \sum_{t_1, t_2} {\left[ -M_{t_1 t_2}^+ \text{ln}\left(\frac{M_{t_1 t_2}^+}{d_{t_1, out}^+ d_{t_2, in}^+}\right) + M_{t_1 t_2}^- \text{ln}\left(\frac{M_{t_1 t_2}^-}{d_{t_1, out}^- d_{t_2, in}^-}\right)\right]}
        
        where the sum runs over all entries $(t_1, t_2)$ in rows and cols $r$ and $s$ of the edge count matrix"""

    M_r_t1 = M[r, :]
    M_s_t1 = M[s, :]
    M_t2_r = M[:, r]
    M_t2_s = M[:, s]

    # remove r and s from the cols to avoid double counting
    # (Double counting correction needed on partial sums s2 s3 s6 s7

    idx = np.ones((M.shape[0],), dtype=bool)
    idx[r] = 0
    idx[s] = 0

    M_r_col = M_r_col[idx]
    M_s_col = M_s_col[idx]
    M_t2_r = M_t2_r[idx]
    M_t2_s = M_t2_s[idx]
    d_out_new_ = d_out_new[idx]
    d_out_ = d_out[idx]

    # only keep non-zero entries to avoid unnecessary computation
    d_in_new_r_row = d_in_new[M_r_row.ravel() != 0]
    d_in_new_s_row = d_in_new[M_s_row.ravel() != 0]

    M_r_row = M_r_row[M_r_row != 0]
    M_s_row = M_s_row[M_s_row != 0]
    d_out_new_r_col = d_out_new_[M_r_col.ravel() != 0]
    d_out_new_s_col = d_out_new_[M_s_col.ravel() != 0]
    M_r_col = M_r_col[M_r_col != 0]
    M_s_col = M_s_col[M_s_col != 0]
    d_in_r_t1 = d_in[M_r_t1.ravel() != 0]
    d_in_s_t1 = d_in[M_s_t1.ravel() != 0]
    M_r_t1= M_r_t1[M_r_t1 != 0]
    M_s_t1 = M_s_t1[M_s_t1 != 0]
    d_out_r_col = d_out_[M_t2_r.ravel() != 0]
    d_out_s_col = d_out_[M_t2_s.ravel() != 0]
    M_t2_r = M_t2_r[M_t2_r != 0]
    M_t2_s = M_t2_s[M_t2_s != 0]

    # sum over the two changed rows and cols
    delta_entropy = 0.0
    delta_entropy -= np.sum(M_r_row * np.log(M_r_row.astype(float) / (d_in_new_r_row * d_out_new[r])))
    delta_entropy -= np.sum(M_s_row * np.log(M_s_row.astype(float) / (d_in_new_s_row * d_out_new[s])))
    delta_entropy -= np.sum(M_r_col * np.log(M_r_col.astype(float) / (d_out_new_r_col * d_in_new[r])))
    delta_entropy -= np.sum(M_s_col * np.log(M_s_col.astype(float) / (d_out_new_s_col * d_in_new[s])))
    delta_entropy += np.sum(M_r_t1 * np.log(M_r_t1.astype(float) / (d_in_r_t1 * d_out[r])))
    delta_entropy += np.sum(M_s_t1 * np.log(M_s_t1.astype(float) / (d_in_s_t1 * d_out[s])))
    delta_entropy += np.sum(M_t2_r * np.log(M_t2_r.astype(float) / (d_out_r_col * d_in[r])))
    delta_entropy += np.sum(M_t2_s * np.log(M_t2_s.astype(float) / (d_out_s_col * d_in[s])))

    return delta_entropy

def compute_delta_entropy_verify(r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out, d_in, d_out_new, d_in_new):
    delta_entropy1 = compute_delta_entropy_orig(r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out, d_in, d_out_new, d_in_new)
    delta_entropy2 = compute_delta_entropy_alt(r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out, d_in, d_out_new, d_in_new)
    assert(np.abs(delta_entropy1 - delta_entropy2) < 1e-9)
    return delta_entropy1

compute_delta_entropy = compute_delta_entropy_alt

if __name__ == '__main__':
    import sys
    for fname in sys.argv[1:]:
        with open(fname, "rb") as f:
            (r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out, d_in, d_out_new, d_in_new) = pickle.load(f)
            delta_entropy = compute_delta_entropy(r, s, M, M_r_row, M_s_row, M_r_col, M_s_col, d_out, d_in, d_out_new, d_in_new)
            print("%s: delta_entropy = %s" % (fname, delta_entropy))
