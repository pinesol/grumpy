#!/bin/bash

# This assumes your grumpy dir is in /scratch/${USER}/grumpy.

if [ -z "$1" ]
  then
    echo "Usage: ./generate_pbs.sh [user_name] [script_name] [experiment_name] [data_dir] [extra flags]"
    echo "e.g.: ./generate_pbs.sh akp258 experiment_1 data --model=large --use_gru"
    exit
fi

#Generate_pbs.sh
user_name=$1 #eg akp258
experiment_name=$2 #eg experiment_1
data_dir=$3 #eg data_dir

echo "
#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=20:00:00
#PBS -l mem=25GB
#PBS -N ${experiment_name}
#PBS -j oe
#PBS -M ${user_name}@nyu.edu
#PBS -m ae

module purge
module load pillow/intel/2.7.0
module load tensorflow/python2.7/20161029
module load scipy/intel/0.18.0

cd /scratch/${USER}/grumpy
python train.py --data=${data_dir} --save=save_dir/${experiment_name}_$(date +'%m%d%H%M%S') ${@:4}

" > ${experiment_name}.pbs
