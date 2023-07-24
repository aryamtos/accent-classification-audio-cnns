DATASET_TRAIN="dataset/train/"
DATASET_VAL="dataset/validation/"
MAX_WAVE_SIZE=195000
NOISE_VALUE=0.1
PATIENCE=25
N_EPOCHS=50
LR=0.0001
DECAY=0.001
python3 main.py train --dataset_train=$DATASET_TRAIN --dataset_val=$DATASET_VAL --max_wave_size=$MAX_WAVE_SIZE --noise_value=$NOISE_VALUE --patience=$PATIENCE --n_epochs=$N_EPOCHS --learning_rate=$LR --weight_decay_=$DECAY

