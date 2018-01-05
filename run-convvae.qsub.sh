#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e

python test_convvae.py --lr 0.001 --epochs 100
