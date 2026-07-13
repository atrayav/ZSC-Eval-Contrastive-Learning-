#!/bin/bash
# Wrapper for launching the seed-1 laptop run from Windows (Start-Process wsl
# -e bash <this file>): keeps every wsl.exe argument space-free so no
# PowerShell/WSL quoting can mangle it. All output -> launcher log.
export SEED_BEGIN=1
export SEED_MAX=1
export N_THREADS=12
cd /home/varun/ZSC-Eval/zsceval/scripts/overcooked/shell
exec bash train_fcp_s2_contrastive.sh random3 12 \
    >> /home/varun/ZSC-Eval/logs/launcher_fcp-S2-contrastive-s12.log 2>&1
