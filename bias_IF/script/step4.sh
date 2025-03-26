TRAIN_TYPE=$1 # mix
MODEL_SIZE=$2 # 350m
DATASET_PERCENTAGE=$3 # few
MODEL_NAME=$4 #Qwen

python -m step3.cal_IF --model_size $MODEL_SIZE --train_type $TRAIN_TYPE --dataset_percentage $DATASET_PERCENTAGE --model_name $MODEL_NAME
# python -m step4.write_selected_data --model_size $MODEL_SIZE --dataset_percentage $DATASET_PERCENTAGE --val_type $TRAIN_TYPE --percentage 0.02
percentages=(0.02 0.05 0.1 0.15 0.35 0.55)

for p in "${percentages[@]}"
do
  python -m step4.write_selected_data --model_size $MODEL_SIZE --dataset_percentage $DATASET_PERCENTAGE --val_type $TRAIN_TYPE --percentage $p --model_name $MODEL_NAME
done