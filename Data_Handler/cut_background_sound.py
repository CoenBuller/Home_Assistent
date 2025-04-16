from process_sounddata import AugmentSoundData

filename = "C:\\Users\\coenb\\Coen_bestanden\\home_assistent\\sound_data\\audio_files\\background_noise.wav"
cut_length = 2 #s

background_audio = AugmentSoundData(filename, sr=16000)
background_audio.len_cut_soundfile(cut_length=cut_length) #s
