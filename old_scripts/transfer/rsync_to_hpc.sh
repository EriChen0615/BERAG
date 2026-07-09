#!/bin/bash
rsync -avzP --exclude 'experiments/' --exclude ‘artifacts/‘ KBVQA/ 
jade2.hartree.stfc.ac.uk:/jmain02/home/J2AD017/wga42/jjc28-wga42/ColBERT