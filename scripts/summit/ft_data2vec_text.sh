fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/roberta/config/finetuning \
	--config-name rte \
	task.data=RTE-bin \
	checkpoint.restore_file=/workspace/models/data2vec_model/nlp_base.pt
