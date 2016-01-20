#!/bin/bash
#PBS -A Monash093
#PBS -N dipole_gas_test_10m
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l pmem=1000MB
#PBS -m abe
#PBS -l walltime=0:10:0
#PBS -o dipole_test_output.log
cd $PBS_O_WORKDIR
python run.py
