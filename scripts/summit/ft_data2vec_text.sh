git pull
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/roberta/config/finetuning \
	--config-name qqp \
	task.data=/workspace/fairseq/examples/roberta/QQP-bin \
	checkpoint.restore_file=/workspace/models/data2vec_model/nlp_base.pt \
	criterion._name=sentence_prediction_viewmaker \
    checkpoint.best_checkpoint_metric=F1	
