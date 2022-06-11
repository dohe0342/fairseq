git pull
fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/roberta/config/finetuning \
	--config-name sst_2 \
	task.data=/workspace/fairseq/examples/roberta/SST-2-bin \
	checkpoint.restore_file=/workspace/models/data2vec_model/nlp_base.pt \
	criterion._name=sentence_prediction_viewmaker 
