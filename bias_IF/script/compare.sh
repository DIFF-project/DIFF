percentages=(0.02 0.05 0.1 0.15 0.35 0.55)
MODEL=$1 # Qwen
DATASET_TYPE=$2 # gen_mix
VAL_TYPE=$3 # full
DATA_SIZE=$4 # 1.3b

for p in "${percentages[@]}"
do
    python -m compare_csv_overlap --target ./data/Qwen/${DATA_SIZE}_${DATASET_TYPE} --select ./data/ag_news/crows_gen2_data.csv --dataset_type ${DATASET_TYPE} --val_type ${VAL_TYPE} --percentage $p
    # python -m compare_csv_overlap --target ./data/$MODEL/1.3b_gen_mix --select ./data/$MODEL/1.3b_gen_mix_full/$p/gen_mix_dataset_select_full.csv --dataset_type $DATASET_TYPE --val_type $VAL_TYPE --percentage $p
done