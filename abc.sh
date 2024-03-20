#! /bin/bash

tmux new-session -d -s h_index_calc_1\; send-keys "conda activate python_main" Enter "python3.6 translation_support.py 0 1000" Enter
 