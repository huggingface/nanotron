#!/bin/bash

CONFIG_FILE_106K=/capstor/scratch/cscs/lhuang/nanotron_climllama/climllama/configs/config_finetune_forecast_106k.yaml
export CONFIG_FILE=${CONFIG_FILE_106K}
sbatch -J finetune_climllama_forecast_106k run_finetune.sbatch

CONFIG_FILE_52K=/capstor/scratch/cscs/lhuang/nanotron_climllama/climllama/configs/config_finetune_forecast_52k.yaml
export CONFIG_FILE=${CONFIG_FILE_52K}
sbatch -J finetune_climllama_forecast_52k run_finetune.sbatch