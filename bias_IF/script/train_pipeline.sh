python -m optlm_mid_gender --dataset_type mix
python -m step2.get_info_lora --output_path ./step2/result_1.3b_toxic_mix_val_full --gradient_type sgd --ds step2_val_data/crows_gen_data.csv --md_path ./opt_toxic_mix_1.3b_select_ig
python -m step2.get_info_lora --output_path ./step2/result_1.3b_toxic_mix_full --gradient_type adam --ds val_data/mix_data.csv --md_path ./opt_mix_1.3b_select_ig
python -m step3.cal_IF --model_size 1.3b --train_type mix --dataset_percentage full
python -m step4.write_selected_data --model_size 1.3b --dataset_percentage full --val_type mix --percentage 0.02
python -m compare_csv_overlap --target ./data/350m_toxic_mix --select ./data/val_data/prompt_data.csv --dataset_type toxic_mix --val_type few --percentage 0.02

python -m optlm_mid_retrain --dataset_type mix --val_type full --percentage 0.55

python -m step2.get_info_qwen --output_path ./step2/Qwen/result_500m_gen_mix_val_full --gradient_type sgd --ds step2_val_data/crows_gen_data.csv --md_path ./Qwen/Qwen_gen_mix_full_500m_select_ig --dataset_type gen_mix_full --model_name Qwen/Qwen2.5-0.5B
python -m step2.get_info_qwen_lora --output_path ./step2/Qwen/result_1.3b_gen_mix_val_full --gradient_type sgd --ds step2_val_data/crows_gen_data.csv --md_path ./Qwen/Qwen_gen_mix_full_1.5b_select_ig --dataset_type gen_mix_full --model_name Qwen/Qwen2.5-1.5B-Instruct