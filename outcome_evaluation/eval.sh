#!/bin/bash

EXP_FILE="../baseline_results/expected.jsonl"

# 使用 find 命令遍历 baseline_results 文件夹及其子文件夹下的所有 .jsonl 文件
find ../baseline_results -type f -name "*.jsonl" | while read -r GEN_FILE; do
  echo "Currently processing file: $GEN_FILE"
  
  # 执行第一个 Python 脚本
  python calc.py --gen_file "$GEN_FILE" --exp_file "$EXP_FILE"
  
  # 执行第二个 Python 脚本
  python calc_rel.py --gen_file "$GEN_FILE" --exp_file "$EXP_FILE"

  echo "----------------------------------------------------"
done