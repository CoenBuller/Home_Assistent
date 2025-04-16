import os

root = 'C:\Users\coenb\Coen_bestanden\home_assistent\sound_data\wake_word_audio'
path = os.path.join(root, 'non_wake_word')

try: 
    os.makedirs(path)
except FileExistsError:
    print('Directory already exists.')

background_audio = 'C:\Users\coenb\Coen_bestanden\home_assistent\sound_data\background_audio_segments'
for audio in os.listdir(background_audio):
    if audio.endswith('.wav'):
        os.rename(os.path.join(root, audio), os.path.join(path, audio))
        print(f"Moved {audio} to {path}")