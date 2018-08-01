import multiprocessing as mp
from multiprocessing import Process
import timeit, resource
import sys, itertools, os, time, traceback
from partition_baseline_main import do_main
try:
    import cPickle as pickle
except:
    import pickle

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
             'initial_block_reduction_rate' : 0.50,
             'merge_method' : 0, 'mpi' : 0, 'node_move_update_batch_size' : 1, 'node_propose_batch_size' : 4,
             'parallel_phase' : 3, 'parts' : 0, 'pipe' : 0, 'predecimation' : 0, 'profile' : 0, 'seed' : 0, 'sort' : 0,
             'sparse' : 0, 'sparse_algorithm' : 0, 'sparse_data' : 0, 'test_decimation' : 0, 'threads' : 0, 'verbose' : 2, 'test_resume' : 0, 'min_nodal_moves_ratio' : 0.0}

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


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
class NonDaemonicPool(mp.pool.Pool):
    Process = NoDaemonProcess


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

def outputname(args):
    out = 'out'
    for k,v in args:
        if k == 'input_filename':
            v = os.path.basename(v)
        if k in shortname:
            k = shortname[k]
        if k == 't' or k == 'itr':
            out += ('-%s%02d' % (k,v))
        else:
            out += ('-%s%s' % (k,v))
    return out


def child_func(queue, fout, func, args):
    rc = 0
    wall_time = 0
    rusage_self = None
    rusage_children = None

    try:
        t0 = timeit.default_timer()

        with redirect_stdout(fout):
            func_result = func(args)

        t1 = timeit.default_timer()
        wall_time = t1 - t0
        fout.flush()

        rusage_self = resource.getrusage(resource.RUSAGE_SELF)
        rusage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    except:
        traceback.print_exc()
        traceback.print_exc(file=fout)
        rc = 1

    queue.put((rc, wall_time, rusage_self, rusage_children, func_result))
    sys.exit(rc)

def profile_child(out_dir, func, args):
    outname = out_dir + '/' + outputname(args)
    fout = open(outname, 'w')
    args = Bunch(args)
    queue = mp.Queue()
    p = Process(target=child_func, args=(queue, fout, func, args))
    p.start()
    rc,wall_time,rusage_self,rusage_children,func_result = queue.get()
    p.join()
    return outname,rc,wall_time,rusage_self,rusage_children,func_result

def profile_wrapper(tup):
    (out_dir, args) = tup
    return profile_child(out_dir, do_main, args)

def run_test(out_dir, base_args, input_files, iterations, threads, max_jobs = 1):
    results = {}

    work_list = [i for i in itertools.product(input_files, threads, iterations)]

    arg_list = [base_args.copy() for i in work_list]

    for i,(input_filename,thread,iteration) in enumerate(work_list):
        arg_list[i]['input_filename'] = input_filename
        arg_list[i]['threads'] = thread
        arg_list[i] = tuple(sorted((j for j in arg_list[i].items()))) + (('iteration', iteration),)

    pool = NonDaemonicPool(max_jobs)

    result_list = pool.map(profile_wrapper, [(out_dir, i) for i in arg_list])

    for args,(outname,rc,t_elp,rusage_self,rusage_children,func_result) in zip(arg_list, result_list):
        mem_rss = rusage_self.ru_maxrss + rusage_children.ru_maxrss
        if rc == 0:
            print(args)
            print("Took %3.4f seconds and used %d k maxrss. Function result is %s" % (t_elp, mem_rss, str(func_result)))
            print("")
        else:
            print("Exception occured. Continuing.")
            print("")

        results[args] = outname,t_elp,mem_rss,func_result

    return results


def run_sweep_test(out_dir, base_args, input_files, iterations, threads, reduction_rates, max_jobs = 1):
    results = {}

    work_list = [i for i in itertools.product(input_files, threads, iterations, reduction_rates)]

    arg_list = [base_args.copy() for i in work_list]

    for i,(input_filename,thread,iteration,reduction_rate) in enumerate(work_list):
        arg_list[i]['input_filename'] = input_filename
        arg_list[i]['threads'] = thread
        arg_list[i]['initial_block_reduction_rate'] = reduction_rate
        arg_list[i] = tuple(sorted((j for j in arg_list[i].items()))) + (('iteration', iteration),)

    pool = NonDaemonicPool(max_jobs)

    result_list = pool.map(profile_wrapper, [(out_dir, i) for i in arg_list])

    for args,(outname,rc,t_elp,rusage_self,rusage_children) in zip(arg_list, result_list):
        mem_rss = rusage_self.ru_maxrss + rusage_children.ru_maxrss
        if rc == 0:
            print(args)
            print("Took %3.4f seconds and used %d k maxrss" % (t_elp, mem_rss))
            print("")
        else:
            print("Exception occured. Continuing.")
            print("")

        results[args] = outname,t_elp,mem_rss,func_result

    return results


def update_dict(x, y):
    x.update(y)
    return x

def run_var_test(out_dir, base_args, var_args, max_jobs = 1):
    """
    Run a test suite sweeping across a series of parameter values stored in the var_args dict which override the defaults in base_args.
    """
    results = {}

    # Form the cartesian product of all values for keys arranged in alphabetical order.
    work_list = [i for i in itertools.product(*(i[1] for i in var_args))]

    # Create dicts which will override the default argument dict.
    # The keys are (j[0] for j in var_args), but not used in a generator to avoid exhausting it prematurely.

    var_args_d = ({k : v for k,v in zip((j[0] for j in var_args), i)} for i in work_list)
    arg_list = [update_dict(base_args.copy(), d) for d in var_args_d]

    # Convert to tuples (for later indexing).
    arg_list = [tuple(sorted((j for j in i.items()))) for i in arg_list]

    pool = NonDaemonicPool(max_jobs)

    result_list = pool.map(profile_wrapper, [(out_dir, i) for i in arg_list])

    for args,(outname,rc,t_elp,rusage_self,rusage_children,func_result) in zip(arg_list, result_list):
        mem_rss = rusage_self.ru_maxrss + rusage_children.ru_maxrss
        if rc == 0:
            print(args)
            print("Took %3.4f seconds and used %d k maxrss. Function result is %s" % (t_elp, mem_rss, str(func_result)))
            print("")
        else:
            print("Exception occured. Continuing.")
            print("")

        results[args] = outname,t_elp,mem_rss,func_result

    return results


def print_results(results):
    for k,v in sorted((i for i in results.items())):
        print("%s %s" % (v[0],v[1:]))

if __name__ == '__main__':
    
    out_dir = time.strftime("out-%Y-%m-%d")
    try: os.mkdir(out_dir)
    except: pass

    input_files = ('../../data/static/simulated_blockmodel_graph_100_nodes',
                   '../../data/static/simulated_blockmodel_graph_500_nodes',
                   '../../data/static/simulated_blockmodel_graph_1000_nodes',
                   '../../data/static/simulated_blockmodel_graph_5000_nodes',
                   '../../data/static/simulated_blockmodel_graph_20000_nodes',
                   '../../data/static/simulated_blockmodel_graph_50000_nodes',
                   '../../data/static/simulated_blockmodel_graph_100000_nodes'
    )

    N = {100: '../../data/static/simulated_blockmodel_graph_100_nodes',
         500: '../../data/static/simulated_blockmodel_graph_500_nodes',
         1000: '../../data/static/simulated_blockmodel_graph_1000_nodes',
         5000: '../../data/static/simulated_blockmodel_graph_5000_nodes',
         20000: '../../data/static/simulated_blockmodel_graph_20000_nodes',
         50000: '../../data/static/simulated_blockmodel_graph_50000_nodes',
         100000: '../../data/static/simulated_blockmodel_graph_100000_nodes'
    }

    small_files = input_files[:3]
    big_files = input_files[3:]

    iterations = range(1)

    args = {
        'single-tiny' : 0,
        'single-small' : 0,
        'multi-small'  : 0,
        'single-sparse' : 0,
        'single-big' : 0,
        'reduction-sweep' : 0,
        'reduction-sweep-small' : 0,
        'sparse-sweep' : 0,
        'big' : 1
        }

    results = {}

    results_f = open('regression.pickle', 'wb')

    if args['single-tiny']:
        print("Tiny Single process tests.")
        result = run_test(out_dir, base_args, [N[100]], range(3), threads = (4,), max_jobs = 4)
        print_results(result)
        results.update(result)

    if args['single-small']:
        result = run_test(out_dir, base_args, small_files, iterations, threads = (0,), max_jobs = 6)
        print_results(result)
        results.update(result)

        avg_time = sum([i[1] for i in results.values()]) / len(results)
        print("Mean time is %s" % (avg_time))

    if args['multi-small']:
        result = run_test(out_dir, base_args, small_files, iterations, threads = (2,4,8,16,27,32))
        print("Multi process tests.")
        print_results(result)
        results.update(result)

    if args['single-sparse']:
        print("Sparse tests.")
        base_args['sparse'] = 1
        result = run_test(out_dir, base_args, input_files, iterations, threads = (0,), max_jobs = 6)
        print_results(result)
        results.update(result)

    if args['sparse-sweep']:
        med_files = [N[20000]]

        var_args = (('input_filename', med_files),
                    ('iteration', range(3)),
                    ('initial_block_reduction_rate',(0.50,0.75)),
                    ('sparse',(0,1,2)), ('threads',(8,)))

        result = run_var_test(out_dir, base_args, var_args, max_jobs=2)
        print_results(result)


    if args['big']:
        med_files = [N[100000]]

        var_args = (('input_filename', med_files),
                    ('iteration', range(1)),
                    ('initial_block_reduction_rate',(0.50,0.75,0.95)),
                    ('sparse',(2,)), ('threads',(56,)),
                    ('decimation',(4,))
        )
        result = run_var_test(out_dir, base_args, var_args, max_jobs=1)



    if args['reduction-sweep-small']:
        med_files = [N[20000]]

        var_args = (('input_filename', med_files),
                    ('iteration', range(3)),
                    ('initial_block_reduction_rate',(0.50,0.75,0.90,0.95,0.99)),
                    ('sparse',(0,1,2)), ('threads',(8,)))

        result = run_var_test(out_dir, base_args, var_args, max_jobs=2)
        print_results(result)
        results.update(result)

    if args['reduction-sweep']:
        print("Single var tests.")
        # var_args = { 'initial_block_reduction_rate' : (0.50,0.75,0.90,0.95,0.99), 'threads' : (0, 55) }
        # var_args = (('input_filename', small_files), ('iteration', range(3)))
        var_args = (('input_filename', big_files[3:]), ('initial_block_reduction_rate',(0.50,0.75,0.90,0.95)), ('sparse',(0,)), ('threads',(16,)), ('decimation',(8,)))
        result = run_var_test(out_dir, base_args, var_args)
        print_results(result)
        results.update(result)

        var_args = (('input_filename', big_files[2:]), ('initial_block_reduction_rate',(0.50,0.75,0.90,0.95), ('sparse',(0,)), ('threads',(32,)), ('decimation',(4,))))
        result = run_var_test(out_dir, base_args, var_args)
        print_results(result)
        results.update(result)

    pickle.dump(results, results_f)
