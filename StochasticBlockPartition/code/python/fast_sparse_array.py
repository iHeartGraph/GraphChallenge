import numpy as np


if hasattr(dict, "viewkeys"):
    dict_keys_func = dict.viewkeys
else:
    dict_keys_func = dict.keys

class nonzero_dict(dict):
    def __setitem__(self, idx, val):
        if val == 0:
            try:
                del self[idx]
            except KeyError:
                pass
        else:
            dict.__setitem__(self, idx, val)
    def __getitem__(self, idx):
        try:
            return dict.__getitem__(self, idx)
        except KeyError:
            return 0
    def copy(self):
        d = nonzero_dict(self)
        return d
    def keys(self):
        return np.fromiter(dict.keys(self), dtype=int)
    def values(self):
        return np.fromiter(dict.values(self), dtype=int)
    def dict_keys(self):
        return dict_keys_func(self)

class nonzero_key_value_list(object):
    def __init__(self):
        self.k = []
        self.v = []
    def __setitem__(self, idx, val):
        #print("xxx Inside __setitem__ %s %s" % (str(idx), str(val)))
        try:
            loc = self.k.index(idx)
        except ValueError:
            loc = -1

        if loc == -1:
            if val != 0:
                self.k.append(idx)
                self.v.append(val)
        else:
            if val != 0:
                self.k[loc] = idx
                self.v[loc] = val
            else:
                self.k = self.k[:loc] + self.k[loc+1:]
                self.v = self.v[:loc] + self.v[loc+1:]

    def __getitem__(self, idx):
        #print("xxx Inside __getitem__ %s" % (str(idx)))
        try:
            loc = self.k.index(idx)
            return self.v[loc]
        except ValueError:
            return 0
    def __contains__(self, idx):
        return idx in self.k
    def __delitem__(self, idx):
        try:
            loc = self.k.index(idx)
        except ValueError:
            raise KeyError()
        self.k = self.k[:loc] + self.k[loc+1:]
        self.v = self.v[:loc] + self.v[loc+1:]
    def items(self):
        return zip(self.k,self.v)
    def __iter__(self):
        raise NotImplementedError("__iter__ not implemented")
    def keys(self):
        return np.array(self.k, dtype=int)
    def values(self):
        return np.array(self.v, dtype=int)
    def copy(self):
        d = nonzero_key_value_list()
        d.k = self.k.copy()
        d.v = self.v.copy()
        return d

class nonzero_key_value_sorted_array(object):
    def __init__(self):
        self.k = np.array([], dtype=int)
        self.v = np.array([], dtype=int)
    def __setitem__(self, idx, val):
        # print("xxx Inside __setitem__ %s %s" % (str(idx), str(val)))
        loc = np.searchsorted(self.k, idx)
        if loc == len(self.k) or self.k[loc] != idx:
            if val != 0:
                self.k = np.insert(self.k, loc, idx)
                self.v = np.insert(self.v, loc, val)
        else:
            if val != 0:
                self.k[loc] = idx
                self.v[loc] = val
            else:
                self.k = np.delete(self.k, loc)
                self.v = np.delete(self.v, loc)
    def __getitem__(self, idx):
        # print("xxx Inside __getitem__ %s" % (str(idx)))
        loc = np.searchsorted(self.k, idx)
        if loc == len(self.k) or self.k[loc] != idx:
            return 0
        else:
            return self.v[loc]
    def __contains__(self, idx):
        loc = np.searchsorted(self.k, idx)
        if loc == len(self.k) or self.k[loc] != idx:
            return False
        else:
            return True
    def __delitem__(self, idx):
        loc = np.searchsorted(self.k, idx)
        if loc == len(self.k) or self.k[loc] != idx:
            raise KeyError()
        else:
            self.k = np.delete(self.k, loc)
            self.v = np.delete(self.v, loc)
    def items(self):
        return zip(self.k,self.v)
    def __iter__(self):
        raise NotImplementedError("__iter__ not implemented")
    def keys(self):
        return self.k
    def values(self):
        return self.v
    def copy(self):
        d = nonzero_key_value_sorted_array()
        d.k = self.k.copy()
        d.v = self.v.copy()
        return d

nonzero_data = nonzero_dict
#nonzero_data = nonzero_key_value_sorted_array
#nonzero_data = nonzero_key_value_list

star = slice(None, None, None)
class fast_sparse_array(object):
    def __init__(self, tup):
        self.rows = [nonzero_data() for i in range(tup[0])]
        self.cols = [nonzero_data() for i in range(tup[1])]
        self.shape = tup
        self.debug = 0
        if self.debug:
            self.M_ver = np.zeros(self.shape, dtype=int)
        return
    def __getitem__(self, idx):
        # print("Enter __getitem__ %s" % (str(idx)))
        if 0: #self.debug:
            return self.M_ver.__getitem__(idx)

        i,j = idx
        if type(i) is slice and i == star:
            L = [(k,v) for (k,v) in self.cols[j].items()]
        elif type(j) is slice and j == star:
            L = [(k,v) for (k,v) in self.rows[i].items()]
        else:
            if j in self.rows[i]:
                L = self.rows[i][j]
            else:
                L = 0

        if self.debug:
            L0 = self.M_ver.__getitem__(idx)
            if isinstance(L, Iterable):
                nz = L0.nonzero()[0]
                L_i = np.array([k for (k,v) in L])
                L_v = np.array([v for (k,v) in L])
                s = np.argsort(L_i)
                L_i = L_i[s]
                L_v = L_v[s]
                assert(len(nz) == len(L))
                assert((nz == L_i).all())
                assert((L0[nz] == L_v).all())
            else:
                assert(L0 == L)

        return L
    def __setitem__(self, idx, val):
        #print("Inside __setitem__ %s %s" % (str(idx), str(val)))
        i,j = idx
        self.rows[i][j] = val
        self.cols[j][i] = val
        if self.debug:
            self.M_ver.__setitem__(idx, val)
            self.verify()
    def set_axis_dict(self, idx, axis, d_new):
        if axis == 0:
            # for k in self.rows[idx].keys():
            #     del self.cols[k][idx]
            # for k,v in d.items():
            #     self.cols[k][idx] = v

            for k in self.rows[idx].dict_keys() - d_new.dict_keys():
                del self.cols[k][idx]

            for k,v in d_new.items():
                self.cols[k][idx] = v

            self.rows[idx] = d_new

        elif axis == 1:
            # for k in self.cols[idx].keys():
            #     del self.rows[k][idx]
            # for k,v in d_new.items():
            #     self.rows[k][idx] = v
            # self.cols[idx] = d_new

            for k in self.cols[idx].dict_keys() - d_new.dict_keys():
                del self.rows[k][idx]

            for k,v in d_new.items():
                self.rows[k][idx] = v

            self.cols[idx] = d_new

    def __str__(self):
        return str(self.M_ver)
    def count_nonzero(self):
        return sum(len(d) for d in self.rows)
    def verify(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                a = self.__getitem__((i,j))
                b = self.M_ver.__getitem__((i,j))
                if a != b:
                    raise Exception("Mismatch at element (%d %d) and values (%d %d)" % (i,j,a,b))
    def take(self, idx, axis):
        if axis == 0:
            return (self.rows[idx].keys(),self.rows[idx].values())
        elif axis == 1:
            return (self.cols[idx].keys(),self.cols[idx].values())
        else:
            raise Exception("Invalid axis %s" % (axis))
    def take_dict(self, idx, axis):
        if axis == 0:
            return self.rows[idx]
        elif axis == 1:
            return self.cols[idx]
        else:
            raise Exception("Invalid axis %s" % (axis))
    def copy(self):
        c = fast_sparse_array(self.shape)
        if self.debug:
            c.M_ver = self.M_ver.copy()
        for i in range(c.shape[0]):
            c.rows[i] = self.rows[i].copy()
        for i in range(c.shape[1]):
            c.cols[i] = self.cols[i].copy()
        return c


def is_sorted(x):
    return len(x) == 1 or (x[1:] >= x[0:-1]).all()

def take_nonzero(A, idx, axis, sort):
    if type(A) is np.ndarray:
        a = np.take(A, idx, axis)
        idx = a.nonzero()[0]
        val = a[idx]
        return idx, val
    elif type(A) is fast_sparse_array:
        idx,val = A.take(idx, axis)
        if sort:
            s = np.argsort(idx)
            idx = idx[s]
            val = val[s]
        return idx,val
    else:
        raise Exception("Unknown array type for A (type %s) = %s" % (type(A), str(A)))


def nonzero_slice(A, sort=True):
    if type(A) is np.ndarray:
        idx = A.nonzero()[0]
        val = A[idx]
    elif type(A) is list:
        idx = np.array([k for (k,v) in A], dtype=int)
        val = np.array([v for (k,v) in A], dtype=int)
        if sort:
            s = np.argsort(idx)
            idx = idx[s]
            val = val[s]
    else:
        raise Exception("Unknown array type for A (type %s) = %s" % (type(A), str(A)))
    return idx,val
