import os, sys, subprocess, argparse
import numpy as np
from partition_baseline_support import load_graph, evaluate_partition

if __name__ == '__main__':
    try:
        import shutil
        cols,lines = shutil.get_terminal_size()
        np.set_printoptions(linewidth=cols)
    except:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", nargs="?", type=str, default="../../data/static/simulated_blockmodel_graph_500_nodes")
    parser.add_argument("-t", "--threads", type=int, required=False, default=0)
    parser.add_argument("-k", "--keep", type=int, required=False, default=0)
    args = parser.parse_args()

    path_to_grappolo = '~/grappolo/grappolo-05-2014/driverForGraphClustering'
    input_filename = args.input_filename

    if not os.path.isfile(input_filename + '.tsv'):
        print("File doesn't exist: '{}'!".format(input_filename))
        sys.exit(1)

    true_partition_available = True
    try:
        true_partition = np.loadtxt(input_filename + '_truePartition.tsv', dtype=np.int64)[:,1] - 1
    except:
        true_partition_available = False

    if args.threads != 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    cmd = path_to_grappolo + ' -f 8 ' + input_filename + '.tsv' + ' -o'
    rc,out = subprocess.getstatusoutput(cmd)
    print(out)

    cluster_output_fname = input_filename + '.tsv_clustInfo'

    if true_partition_available:
        partition = np.loadtxt(cluster_output_fname, delimiter='\t', dtype=np.int64)
        evaluate_partition(true_partition, partition)

    if not args.keep:
        os.remove(cluster_output_fname)
