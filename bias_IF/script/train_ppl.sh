MODEL=$1 # Qwen
DATASET_TYPE=$2 # gen_mix
VAL_TYPE=$3 # full
MODEL_SIZE=$4 # 1.3b

if [[ "${MODEL}" == "Qwen" ]]; then
    if [[ "${MODEL_SIZE}" == "1.5b" ]]; then
        MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
    else
        MODEL_NAME="Qwen/Qwen2.5-0.5B"
    fi
else
    if [[ "${MODEL_SIZE}" == "1.3b" ]]; then
        MODEL_NAME="facebook/opt-1.3b"
    else
        MODEL_NAME="facebook/opt-350m"
    fi
fi

if [[ "${MODEL_SIZE}" == "1.5b" ]]; then
    python -m step2.get_info_qwen_lora --output_path ./step2/${MODEL}/result_${MODEL_SIZE}_${DATASET_TYPE}_val_${VAL_TYPE} --gradient_type sgd --ds step2_val_data/crows_gen_data.csv --md_path ./${MODEL}/${MODEL}_${DATASET_TYPE}_${VAL_TYPE}_${MODEL_SIZE}_select_ig --dataset_type ${DATASET_TYPE}_${VAL_TYPE} --model_name ${MODEL_NAME}
    python -m step2.get_info_qwen_lora --output_path ./step2/${MODEL}/result_${MODEL_SIZE}_${DATASET_TYPE}_${VAL_TYPE} --gradient_type adam --ds val_data/${DATASET_TYPE}_data.csv --md_path ./${MODEL}/${MODEL}_${DATASET_TYPE}_${VAL_TYPE}_${MODEL_SIZE}_select_ig --dataset_type ${DATASET_TYPE}_${VAL_TYPE} --model_name ${MODEL_NAME}
else
    python -m step2.get_info_qwen --output_path ./step2/${MODEL}/result_${MODEL_SIZE}_${DATASET_TYPE}_val_${VAL_TYPE} --gradient_type sgd --ds step2_val_data/crows_gen_data.csv --md_path ./${MODEL}/${MODEL}_${DATASET_TYPE}_${VAL_TYPE}_${MODEL_SIZE}_select_ig --dataset_type ${DATASET_TYPE}_${VAL_TYPE} --model_name ${MODEL_NAME}
    python -m step2.get_info_qwen --output_path ./step2/${MODEL}/result_${MODEL_SIZE}_${DATASET_TYPE}_${VAL_TYPE} --gradient_type adam --ds val_data/${DATASET_TYPE}_data.csv --md_path ./${MODEL}/${MODEL}_${DATASET_TYPE}_${VAL_TYPE}_${MODEL_SIZE}_select_ig --dataset_type ${DATASET_TYPE}_${VAL_TYPE} --model_name ${MODEL_NAME}
fi
python -m step3.cal_IF --model_size ${MODEL_SIZE} --train_type ${DATASET_TYPE} --dataset_percentage ${VAL_TYPE}
# python -m step4.write_selected_data --model_size ${MODEL_SIZE} --dataset_percentage ${VAL_TYPE}--val_type mix --percentage 0.02