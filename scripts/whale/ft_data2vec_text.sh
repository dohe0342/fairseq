git pull
task=$1
TASK=${task^^}

for i in {0..4}
do
	fairseq-hydra-train \
		--config-dir /home/work/workspace/fairseq/examples/roberta/config/finetuning \
		--config-name $task \
		task.data=/home/work/workspace/fairseq/examples/roberta/$TASK-bin \
		checkpoint.restore_file=/home/work/workspace/models/data2vec_model/nlp_base.pt \
		criterion._name=sentence_prediction_viewmaker 
done
