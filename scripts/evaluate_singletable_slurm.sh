# Submit jobs to SLURM for evaluation on singletables.

ENV_NAME="carte_test" # Change the environment name accordingly
JOB_NAME="carte_eval"
TIME_HOUR=20
N_CPUS=16
MAX_PARALLEL_TASKS=1
PARTITION="parietal,normal-best,gpu-best" # parietal normal-best, normal gpu-best
EXCLUDE="marg[042-044],margpu[005-008]"
DATA_NAME=("wine_vivino_price")
METHOD_NAME=("target-encoder_resnet")
NT_VALUES=("512")
RS_VALUES=("10")
DEVICE="cpu"
BAGGING="False"

for i in ${DATA_NAME[@]}; do
    conda run -n $ENV_NAME python -W ignore scripts/evaluate_singletable_slurm.py \
        -jn $JOB_NAME -t $TIME_HOUR --gpu -w $N_CPUS -mpt $MAX_PARALLEL_TASKS -p $PARTITION -ex $EXCLUDE \
        -dn $i -nt ${NT_VALUES[@]} -m ${METHOD_NAME[@]} -rs ${RS_VALUES[@]} -b $BAGGING \
        -dv $DEVICE
done
