from scipy.io.wavfile import write
from tqdm import tqdm
import torch 
import torchaudio
import librosa as lb
import random
import torchaudio.transforms as T
import os


class AugmentSoundData():

    def __init__(self, filepath, sr=16000):
        self._filepath = filepath
        self._y, self._sr = lb.load(filepath, sr=sr)  # Load the audio file
        self._mfcc = self.extract_mfcc()

    def extract_mfcc(self, n_features=13):
        """
        Extract MFCC features from the audio file.
        :param sr: Sample rate
        :param n_features: Number of MFCC features to extract
        :return: MFCC tensor of shape (n_mfcc, time)
        """
        mfcc = lb.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_features)
                                        
        return torch.tensor(mfcc)  
    
    @staticmethod
    def augment_mfcc(mfcc=None):         
        """
        Apply random corruptions to an MFCC tensor.
        :param mfcc: Tensor of shape (n_mfcc, time)
        :return: Augmented MFCC tensor (same shape)
        """
        aug = mfcc.clone()

        # --- 1. Time masking (simulate dropouts)
        if random.random() < 0.5:
            time_mask = T.TimeMasking(time_mask_param=10)
            aug = time_mask(aug)

        # --- 2. Frequency masking (simulate mic response loss)
        if random.random() < 0.5:
            freq_mask = T.FrequencyMasking(freq_mask_param=4)
            aug = freq_mask(aug)

        # --- 3. Random scaling (simulate volume gain)
        if random.random() < 0.4:
            scale = random.uniform(0.7, 1.3)
            aug *= scale

        # --- 4. Random noise injection
        if random.random() < 0.4:
            noise = torch.randn_like(aug) * random.uniform(0.01, 0.05)
            aug += noise

        # --- 5. Time shifting
        if random.random() < 0.5:
            shift = random.randint(-5, 5)
            aug = torch.roll(aug, shifts=shift, dims=1)

        # Clamp to avoid extreme values
        aug = torch.clamp(aug, min=-50.0, max=50.0)
        return aug

    def n_cut_soundfile(self, n_cuts, root=None):
        """
        Cut the sound file into n_cuts segments and save each cut as a seperate sound file
        at the location {filepath}_cut_{i}.
        :param n_cuts: Number of cuts to make
        """

        if root is None:
            raise ValueError("Root directory must be specified.") #User must specify a root directory to save the audio
        elif not os.path.exists(root):
            os.makedirs(root)
        
        cut_length = len(self.y) // n_cuts
        cuts = [self.y[i : i + cut_length] for i in range(0, len(self.y)-cut_length, cut_length)]
        cuts.append(self.y[len(self.y)-cut_length:]) # last cut can be shorter than cut_length

        # Save each cut as a separate file in the created directory
        for i in tqdm(range(len(cuts))):
            filename = os.path.basename(self.filepath).rstrip('.wav')
            file_path = os.path.join(root, f"{filename}_cut_{i}.wav")
            write(file_path, self.sr, cuts[i])  # Ensure data is in int16 format for WAV
            
    def len_cut_soundfile(self, cut_length, root=None):
        """
        cuts the sound file into segments of cut_length and save each cut as a seperate 
        sound file and save it at the location {filepath}_cut_{i} in the map 'background_audio_segments'.
        :param cut_length: Length of each cut in seconds
        """

        if root is None:
            raise ValueError("Root directory must be specified.") #User must specify a root directory to save the audio
        elif not os.path.exists(root): 
            os.makedirs(root)

        cut_length = cut_length * self.sr
        cuts = [self.y[i : i + cut_length] for i in range(0, len(self.y)-cut_length, cut_length)]
        cuts.append(self.y[len(self.y)-cut_length:]) # last cut can be shorter than cut_length
        
        # Save each cut as a separate file in the created directory
        for i in tqdm(range(len(cuts))):
            filename = os.path.basename(self.filepath).rstrip('.wav')
            file_path = os.path.join(root, f"{filename}_cut_{i}.wav")
            write(file_path, self.sr, cuts[i])  # Ensure data is in int16 format for WAV
    
    @property
    def mfcc(self):
        return self._mfcc.clone()
    
    @property
    def filepath(self):
        return self._filepath
    

    