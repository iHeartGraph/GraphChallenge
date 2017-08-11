import multiprocessing as mp
from multiprocessing import Process
import timeit, resource
import sys, itertools, os, time, traceback
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


def child_func(queue, redirected_stdout, func, args):
    rc = 0
    t0 = timeit.default_timer()

    with redirect_stdout(redirected_stdout):
        try:
            func(args)
        except:
            traceback.print_exc()
            traceback.print_exc(file=redirected_stdout)
            rc = 1

    t1 = timeit.default_timer()
    wall_time = t1 - t0
    redirected_stdout.flush()
    rusage_self = resource.getrusage(resource.RUSAGE_SELF)
    rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    queue.put((rc, wall_time, rusage_self, rusage_children))
    sys.exit(rc)

def profile_child(redirected_stdout, func, args):
    queue = mp.Queue()
    p = Process(target=child_func, args=(queue, redirected_stdout, func, args))
    p.start()
    rc,wall_time,rusage_self,rusage_children = queue.get()
    p.join()
    return rc,wall_time,rusage_self,rusage_children

def run_test(out_dir, base_args, input_files, iterations, threads):
    results = {}

    for input_filename,thread,iteration in itertools.product(input_files, threads, iterations):
        args = base_args.copy()
        args['input_filename'] = input_filename
        args['threads'] = thread
        args_sorted = tuple(sorted((i for i in args.items()))) + (('iteration', iteration),)

        outname = out_dir + '/' + outputname(args_sorted)

        with open(outname, 'w') as f:
            print("Running " + str(args))
            rc,t_elp,rusage_self,rusage_children = profile_child(f, do_main, Bunch(args))
            mem_rss = rusage_self.ru_maxrss + rusage_children.ru_maxrss

            if rc == 0:
                print("Took %3.4f seconds and used %d k maxrss" % (t_elp, mem_rss))
                print("")
            else:
                print("Exception occured. Continuing.")
                print("")

        results[args_sorted] = outname,t_elp,mem_rss
    return results


if __name__ == '__main__':
    out_dir = time.strftime("out-%Y-%m-%d")
    try: os.mkdir(out_dir)
    except: pass

    input_files = ('../../data/static/simulated_blockmodel_graph_100_nodes',
                   '../../data/static/simulated_blockmodel_graph_500_nodes',
                   '../../data/static/simulated_blockmodel_graph_1000_nodes',
                   '../../data/static/simulated_blockmodel_graph_5000_nodes')

    iterations = range(3)

    results = run_test(out_dir, base_args, input_files, iterations, threads = (0,))

    print("Single process tests.")
    for k,v in sorted((i for i in results.items())):
        print("%s %s" % (v[0],v[1:]))

    results = run_test(out_dir, base_args, input_files, iterations, threads = (4,8,11))

    print("Multi process tests.")
    for k,v in sorted((i for i in results.items())):
        print("%s %s" % (v[0],v[1]))
