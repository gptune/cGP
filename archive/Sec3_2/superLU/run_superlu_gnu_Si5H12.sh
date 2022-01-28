#!/bin/bash
#SBATCH -C haswell
#SBATCH -J test_driver
#SBATCH --qos=regular
#SBATCH -t 40:00:00
#SBATCH --nodes=9
#SBATCH --mail-user=hrluo@lbl.gov
#SBATCH --mail-type=ALL

salloc -N 9 -C haswell -q interactive -t 04:00:00

module load python/3.7-anaconda-2019.10

module unload cray-mpich
module unload cmake
module load cmake/3.14.4

module unload PrgEnv-intel
module load PrgEnv-gnu

export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
BLAS_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"
LAPACK_LIB="${MKLROOT}/lib/intel64/libmkl_gf_lp64.so;${MKLROOT}/lib/intel64/libmkl_gnu_thread.so;${MKLROOT}/lib/intel64/libmkl_core.so;-lgomp"

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
module load openmpi/4.0.1
export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

CCC=mpicc
CCCPP=mpicxx
FTN=mpif90

cd ~/GPTune/examples/minimalcGP/
pip install -r requirements.txt --user

#mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python cGP_constrained.py 123 30 1.0 0 4 3 10 SI2.bin


export PATH="/global/homes/h/hrluo/.local/cori/3.7-anaconda-2019.10/bin:$PATH" 

echo 'cGP superLU experiment environment setup finished...'


sh run_matrix.sh Si5H12.bin


echo 'cGP superLU experiment done!'
