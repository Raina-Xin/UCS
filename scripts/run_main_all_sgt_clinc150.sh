#!/bin/bash
# Run comparison: original vs SGT retrievers on 3 models
for model in Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.2-3B-Instruct google/gemma-2-9b-it; do
  python main_openicl_comparison.py \
    --dataset_type clinc150 \
    --model_name $model \
    --embedding_model_name $model \
    --ice_num 10 \
    --test_size 500 \
    --retrievers dpp mdl votek dpp_sgt mdl_sgt votek_sgt \
    --clustering dict_dbscan \
    --dict_n_components 64 \
    --dict_alpha 10.0 \
    --dict_top_k 4 \
    --dict_tau 1e-3 \
    --dict_transform_algorithm omp \
    --dict_regularization_type l2 \
    --dict_max_iter 50 \
    --dict_pca_dim 128 \
    --dict_batch_size 512 \
    --dbscan_k 20 \
    --dbscan_q 0.01 \
    --dbscan_min_samples 1 \
    --sgt_lambda 0.1 \
    --sgt_t 5.0 \
    --sgt_bin_size 20 \
    --sgt_offset 1.0 \
    --sentence_transformers_model all-mpnet-base-v2 \
    --candidate_num 50 \
    --votek_k 3 \
    --mdl_select_time 5 \
    --mdl_ce_model gpt2 \
    --dpp_scale_factor 0.1 \
    --seed 42 \
    --n_runs 3
done