git pull
task1=$1
TASK1=${task1^^}

task2=$2
TASK2=${task2^^}

for i in {5..9}
do
	CUDA_VISIBLE_DEVICES=$3 fairseq-hydra-train \
		--config-dir /home/work/workspace/fairseq/examples/roberta/config/finetuning \
		--config-name $task1 \
		task.data=/home/work/workspace/fairseq/examples/roberta/$TASK1-bin \
		checkpoint.restore_file=/home/work/workspace/models/data2vec_model/nlp_base.pt \
		criterion._name=sentence_prediction_viewmaker \
		checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/"$task1"_try$i

	CUDA_VISIBLE_DEVICES=$3 fairseq-hydra-train \
		--config-dir /home/work/workspace/fairseq/examples/roberta/config/finetuning \
		--config-name $task2 \
		task.data=/home/work/workspace/fairseq/examples/roberta/$TASK2-bin \
		checkpoint.restore_file=/home/work/workspace/models/data2vec_model/nlp_base.pt \
		criterion._name=sentence_prediction_viewmaker \
		checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/"$task2"_try$i

done
