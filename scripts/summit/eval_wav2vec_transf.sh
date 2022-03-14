subset=$1
CUDA_VISIBLE_DEVICES=$2 python examples/speech_recognition/infer.py \
  /workspace/LibriSpeech/$subset \
  --task audio_finetuning \
  --nbest 1 \
  --path /workspace/fairseq/examples/wav2vec/wav2vec_model/dohe_wav2vec2_s_100h_v2.pt \
  --gen-subset $subset \
  --results-path /workspace/fairseq/examples/wav2vec/ctc_eval \
  --w2l-decoder fairseqlm \
  --lexicon /workspace/fairseq/examples/wav2vec/lm_model/librispeech_lexicon.lst \
  --lm-model /workspace/fairseq/examples/wav2vec/lm_model/lm_librispeech_word_transformer.pt \
  --lm-weight 0.87 \
  --word-score -1 \
  --sil-weight 0 \
  --criterion ctc \
  --labels ltr \
  --max-tokens 4000000 \
  --post-process letter \
  --beam=500
