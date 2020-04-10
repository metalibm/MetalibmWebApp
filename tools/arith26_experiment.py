import subprocess
import re
import sys
import argparse

parser = argparse.ArgumentParser(description='arith26 experimentation script')
parser.add_argument('--output', dest='output', action='store',
                    default="experiment.data",
                    help='set output file')
parser.add_argument('--table', dest='output_table', action='store',
                    default="speedup.tex",
                    help='set output file')
parser.add_argument('--targets', dest='targets', action='store',
					type=(lambda v: list(int(a) for a in v.split(","))),
                    default=range(10, 50, 10),
                    help='set list of target accuracies')
parser.add_argument('--timeout', dest='timeout', action='store',
                    default=30, type=int,
                    help='define test timeout')
parser.add_argument('--problem-defs', dest='problem_defs', action='store',
					type=(lambda v: (v.split(","))),
                    default=["exp", "sin", "sinh", "zeros"],
                    help='set list of target accuracies')


# the following environement variables must be defined
# - METALIBM_LUGDUNUM
#   GENERICIMPLEMENTPOLY_ROOT
#   METALIBM_LUTETIA_BINARY

# EXP_CMD_TEMPLATE = """\
# METALIBM_CFLAGS="-mfma -mavx2 -I${METALIBM_LUGDUNUM}/metalibm_core/" 
# GENERICIMPLEMENTPOLY_ROOT=/home/nicolas/Metalibm/genericimplementpoly/ 
# PYTHONPATH=/home/nicolas/Metalibm/genericimplementpoly/:/home/nicolas/Metalibm/metalibm_py3/:$PWD/
# LD_LIBRARY_PATH=/home/nicolas/Metalibm/local/lib/cgpe/:/usr/local/lib/
# METALIBM_FORCE_LEGACY_IMPLEMENTPOLY={legacy_mode}
# ./metalibm problemdef_arith26.sollya"""

EXP_CMD_TEMPLATE = """\
METALIBM_FORCE_LEGACY_IMPLEMENTPOLY={legacy_mode} ${{METALIBM_LUTETIA_BIN}} problemdef_arith26.sollya"""

EXP_PROBLEM_DEF_TEMPLATE ="""
f = exp(x);
dom = [-1/2,1/2];
target = 2^(-{target});
maxDegree = 17;
minWidth = (sup(dom) - inf(dom)) * 1/4096;
tableIndexWidth = 0;
minimalReductionRatio = 1000;
metaSplitMinWidth = (sup(dom) - inf(dom)) * 1/128;
performExpressionDecomposition = 0;
adaptWorkingPrecision = false;
maxDegreeReconstruction = 5;
"""

SIN_PROBLEM_DEF_TEMPLATE = """
f = sin(x);
dom = [-pi/4,pi/4];
target = 2^(-{target});
maxDegree = 8;
minWidth = (sup(dom) - inf(dom)) * 1/4096;
tableIndexWidth = 0;
minimalReductionRatio = 1000;
metaSplitMinWidth = (sup(dom) - inf(dom)) * 1/128;
performExpressionDecomposition = 0;
adaptWorkingPrecision = false;
maxDegreeReconstruction = 5;

"""
SINH_PROBLEM_DEF_TEMPLATE = """
f = sinh(x);
dom = [-1,1];
target = 2^(-{target});
maxDegree = 17;
minWidth = (sup(dom) - inf(dom)) * 1/4096;
tableIndexWidth = 0;
minimalReductionRatio = 1000;
metaSplitMinWidth = (sup(dom) - inf(dom)) * 1/128;
performExpressionDecomposition = 0;
adaptWorkingPrecision = false;
maxDegreeReconstruction = 5;
"""

ZERO_PROBLEM_DEF_TEMPLATE = """
f = erf(x) - 1/2;
dom = [1/4;3/4];
target = 2^(-{target});
maxDegree = 8;
minWidth = (sup(dom) - inf(dom)) * 1/4096;
tableIndexWidth = 8;
minimalReductionRatio = 1000;
metaSplitMinWidth = (sup(dom) - inf(dom)) * 1/128;
performExpressionDecomposition = 0;
adaptWorkingPrecision = 80;
maxDegreeReconstruction = 5;
"""


args = parser.parse_args()

TIMEOUT = args.timeout
OUTPUT_NAME = args.output
TARGET_LIST = args.targets
print("TARGET_LIST: {}".format(TARGET_LIST))

PROBLEM_DEF_MAP = {
    "exp": ("exp", EXP_PROBLEM_DEF_TEMPLATE),
    "sin": ("sin", SIN_PROBLEM_DEF_TEMPLATE),
    "sinh": ("sinh", SINH_PROBLEM_DEF_TEMPLATE),
    "zeros": ("zeros", ZERO_PROBLEM_DEF_TEMPLATE),
}

PROBLEM_DEF_LIST = [PROBLEM_DEF_MAP[key] for key in args.problem_defs]

def execute_metalibm_test(tag, problemdef_template, target, legacy_mode):
    with open("problemdef_arith26.sollya", "w") as f:
        f.write(problemdef_template.format(target=target))
        f.close()

        exp_cmd = EXP_CMD_TEMPLATE.format(legacy_mode=legacy_mode)

        try:
            cmd_result = subprocess.check_output(
               exp_cmd, shell=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print("timeout for {} target={}, legacy_mode={}".format(tag, target, legacy_mode))

            return -1.0
        else:

            if sys.version_info >= (3, 0):
                metalibm_result = str(cmd_result, 'utf-8')

            match = re.search("(?P<latency>\d+\.\d+) time units", metalibm_result)
            if match is None:
                latency = -1.0
            else:
                latency = float(match.group("latency"))
            print("latency for {} target={}, legacy_mode={} is {}".format(tag, target, legacy_mode, latency))
            return latency
    return -1.0

result = {}
for tag, pdef_template in PROBLEM_DEF_LIST:
    for target in TARGET_LIST:
        for legacy_mode in ["yes", "no"]:
            latency = execute_metalibm_test(tag, pdef_template, target, legacy_mode)
            result[(tag, target, legacy_mode)] = latency

# final summary
for tag, _ in PROBLEM_DEF_LIST:
    print("{}:".format(tag))
    for target in TARGET_LIST:
        print("    {} bits, legacy={:4.4f}, alternative={:4.4f}".format(target, result[(tag, target, "yes")], result[(tag, target, "no")]))

def latency2str(latency):
    if latency < 0:
        return "    ?  "
    else:
        return "{:4.2f}".format(latency)

# generating output experiment.data file for gnuplot
with open(OUTPUT_NAME, "w") as f:
    f.write("t ")
    for tag, _ in PROBLEM_DEF_LIST:
        f.write("{}_yes ".format(tag))
        f.write("{}_no ".format(tag))
    f.write("\n")

    for target in TARGET_LIST:
        f.write("{} ".format(target))
        for tag, _ in PROBLEM_DEF_LIST:
            f.write("{} {} ".format(
                latency2str(result[(tag, target, "yes")]),
                latency2str(result[(tag, target, "no")])
            ))
        f.write("\n")

with open(args.output_table, "w") as f:
    f.write("accuracy ")
    for tag, _ in PROBLEM_DEF_LIST:
        f.write(" & {} ".format(tag))
    f.write(" \\\\\n")
    f.write(" \\hline\n")

    for target in TARGET_LIST:
        f.write("{} ".format(target))
        for tag, _ in PROBLEM_DEF_LIST:
            legacy = result[(tag, target, "yes")]
            new = result[(tag, target, "no")]
            if legacy < 0 or new < 0:
                ratio = "?"
            else:
                ratio = "{:.2f}".format(legacy / new)
            f.write("& {} ".format(ratio))
        f.write("\\\\\n")
