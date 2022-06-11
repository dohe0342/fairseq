git pull
task=$1
TASK=${task^^}

fairseq-hydra-train \
    --config-dir /home/work/workspace/fairseq/examples/roberta/config/finetuning \
    --config-name $task \
    task.data=/home/work/workspace/fairseq/examples/roberta/$TASK-bin \
    checkpoint.restore_file=/home/work/workspace/models/data2vec_model/nlp_base.pt \
    criterion._name=sentence_prediction_viewmaker \
    optimization.max_update=6108 \
    optimization.max_epoch=30 \
    #checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/2022-06-11/03-26-09/checkpoints	
    #dataset.batch_size=64
    #+model.branch_ctc_v1=false \
    #+model.branch_ctc_v2=false \
    #+model.branch_ctc_v3=false 

