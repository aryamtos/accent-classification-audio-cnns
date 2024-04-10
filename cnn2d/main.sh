DATASET_TRAIN=""
DATASET_VAL=""
DATASET_TEST="/test_/"
CHECKPOINT="checkpoint.pt"
BATCH_SIZE=32
LR=0.0001
DECAY=2e-5
N_EPOCHS=50
PATIENCE=25
SEED=42

#python3 run.py train --dataset_train=$DATASET_TRAIN --dataset_val=$DATASET_VAL --batch_size=$BATCH_SIZE --lr=$LR  --decay=$DECAY --n_epochs=$N_EPOCHS --patience=$PATIENCE --seed=$SEED
#python3 run.py eval --dataset_test=$DATASET_TEST --batch_size=$BATCH_SIZE --checkpoint=$CHECKPOINT

if [ -f "$CHECKPOINT" ]; then
    python3 run.py gradcam --dataset_test=$DATASET_TEST --batch_size=$BATCH_SIZE --checkpoint=$CHECKPOINT
else
    echo "Checkpoint não encontrado: $CHECKPOINT"
fi
