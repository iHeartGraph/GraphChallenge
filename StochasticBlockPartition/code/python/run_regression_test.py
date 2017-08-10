import sys, itertools, os
from partition_baseline_main import do_main

try:
    from contextlib import redirect_stdout
except:
    # Back port for Python 2.7x
    # See: https://stackoverflow.com/questions/44226221/contextlib-redirect-stdout-in-python2-7
    import contextlib
    @contextlib.contextmanager
    def redirect_stdout(target):
        original = sys.stdout
        sys.stdout = target
        yield
        sys.stdout = original


base_args = {'debug' : 0, 'decimation' : 0,
             'input_filename' : '../../data/static/simulated_blockmodel_graph_100_nodes',
             'merge_method' : 0, 'mpi' : 0, 'node_move_update_batch_size' : 1, 'node_propose_batch_size' : 4,
             'parallel_phase' : 3, 'parts' : 0, 'pipe' : 0, 'predecimation' : 0, 'profile' : 0, 'seed' : 0, 'sort' : 0,
             'sparse' : 0, 'sparse_algorithm' : 0, 'sparse_data' : 0, 'test_decimation' : 0, 'threads' : 0, 'verbose' : 2}

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


shortname = {'decimation' : 'd',
             'input_filename' : 'F',
             'iteration' : 'itr',
             'merge_method' : 'm',
             'parts' : 'p',
             'predecimation' : 'predec',
             'parallel_phase' : 'P',
             'node_move_update_batch_size' : 'b',
             'node_propose_batch_size' : 'g',
             'sort' : 's',
             'seed' : 'S',
             'threads' : 't',
             'verbose' : 'v'}

def outputname(args_tuple):
    out = 'out'
    for k,v in args_tuple:
        if k == 'input_filename':
            v = os.path.basename(v)
        if k in shortname:
            k = shortname[k]
        if k == 't' or k == 'itr':
            out += ('-%s%02d' % (k,v))
        else:
            out += ('-%s%s' % (k,v))
    return out


def run_test(base_args, input_files, iterations, threads):
    results = {}

    for input_filename,iteration,thread in itertools.product(input_files, iterations, threads):
        args = base_args.copy()
        args['input_filename'] = input_filename
        args['threads'] = thread
        args_sorted = tuple(sorted((i for i in args.items()))) + (('iteration', iteration),)

        outname = outputname(args_sorted)

        with open(outname, 'w') as f:
            with redirect_stdout(f):
                t_elp_part = do_main(Bunch(args))

        results[args_sorted] = outname,t_elp_part
    return results

if __name__ == '__main__':
    input_files = ('../../data/static/simulated_blockmodel_graph_100_nodes',
                   '../../data/static/simulated_blockmodel_graph_500_nodes',
                   '../../data/static/simulated_blockmodel_graph_1000_nodes')
    iterations = range(3)

    results = run_test(base_args, input_files, iterations, threads = (0,))

    print("Single process tests.")
    for k,v in sorted((i for i in results.items())):
        print("%s %s" % (v[0],v[1]))

    results = run_test(base_args, input_files, iterations, threads = (4,8,11))

    print("Multi process tests.")
    for k,v in sorted((i for i in results.items())):
        print("%s %s" % (v[0],v[1]))
