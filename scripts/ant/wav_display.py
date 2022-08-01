import matplotlib.pyplot as plt
import librosa
import librosa.display as display

y, sr = librosa.load('/workspace/LibriSpeech/test-clean/3570/5694/3570-5694-0008.wav', sr=16000)
plt.figure()
display.waveshow(y, sr=sr)
plt.savefig('./test_wavform.png', dpi=300)
