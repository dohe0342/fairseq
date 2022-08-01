import matplotlib.pyplot as plt
import librosa
import librosa.display

y, sr = librosa.load('/workspace/LibriSpeech/test-clean/3570/5694/3570-5694-0008.wav', sr=16000)
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.savefig('./test_wavform.png', dpi=300)
