fairseq-hydra-train \
	--config-dir /workspace/fairseq/examples/roberta/config/finetuning \
	--config-name rte \
	task.data=RTE-bin \
	checkpoint.restore_file=$ROBERTA_PATH
