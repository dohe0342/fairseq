import matplotlib.pyplot as plt
import librosa
import librosa.display as display
import glob
#y, sr = librosa.load('/workspace/LibriSpeech/test-clean/3570/5694/3570-5694-0008.wav', sr=16000)
file_list = sorted(glob.glob('/workspace/play/*'))
for wav_file in file_list:
    file_name = wav_file.split('/')[-1]
    y, sr = librosa.load(wav_file, sr=48000)
    plt.figure()
    display.waveshow(y, sr=sr)
    plt.savefig('./.png', dpi=300)
