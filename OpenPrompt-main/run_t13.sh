#!/bin/bash
#$ -S /bin/bash
#$ -N calibration
#$ -cwd
echo "job start time:`date`"
python tutorial/1.3_calibration.py
echo "job end time:`date`"