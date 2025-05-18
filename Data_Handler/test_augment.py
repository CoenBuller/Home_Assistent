from Home_Assistent.Wake_Word_Detector.process_sounddata import AugmentSoundData
import matplotlib.pyplot as plt
import librosa as lb

filepath = 'C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\wake_word_audio\\non_wake_word_map\\common_voice_nl_38255324.wav'

soundfile = AugmentSoundData(filepath)
augmented_mfcc = AugmentSoundData.augment_mfcc(soundfile.mfcc)

plt.figure(figsize=(10, 4))
lb.display.specshow(augmented_mfcc.numpy(), x_axis='time', cmap='coolwarm')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
