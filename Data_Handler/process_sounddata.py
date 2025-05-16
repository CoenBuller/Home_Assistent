from scipy.io.wavfile import write
from tqdm import tqdm
from torch import Tensor
import torch 
import torchaudio
import librosa as lb
import random
import torchaudio.transforms as T
import os



class AugmentSoundData():

    def __init__(self, filepath: str, sr: int = 16000):
        self._filepath = filepath
        self._y, self._sr = lb.load(filepath, sr=sr)  # Load the audio file
        self._mfcc = self.extract_mfcc()

    def extract_mfcc(self, n_features=13) -> torch.Tensor:
        """
        Extract MFCC features from the audio file.
        :param sr: Sample rate
        :param n_features: Number of MFCC features to extract
        :return: MFCC tensor of shape (n_mfcc, time)
        """
        mfcc = lb.feature.mfcc(y=self._y, sr=self._sr, n_mfcc=n_features)
                                        
        return torch.tensor(mfcc)  
    
    @staticmethod
    def augment_mfcc(mfcc: Tensor) -> Tensor:         
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

        return aug


    def n_cut_soundfile(self, n_cuts, root=None) -> None:
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
        cuts = [self._y[i : i + cut_length] for i in range(0, len(self._y)-cut_length, cut_length)]
        cuts.append(self._y[len(self._y)-cut_length:]) # last cut can be shorter than cut_length

        # Save each cut as a separate file in the created directory
        for i in tqdm(range(len(cuts))):
            filename = os.path.basename(self._filepath).rstrip('.wav')
            file_path = os.path.join(root, f"{filename}_cut_{i}.wav")
            write(file_path, self._sr, cuts[i])  # Ensure data is in int16 format for WAV


    def len_cut_soundfile(self, cut_length: int, root: str|None = None) -> None:
        """
        cuts the sound file into segments of cut_length and save each cut as a seperate 
        sound file and save it at the location {filepath}_cut_{i} in the map 'background_audio_segments'.
        :param cut_length: Length of each cut in seconds
        """

        if root is None:
            raise ValueError("Root directory must be specified.") #User must specify a root directory to save the audio
        elif not os.path.exists(root): 
            os.makedirs(root)

        cut_length = cut_length * self._sr
        cuts = [self._y[i : i + cut_length] for i in range(0, len(self._y)-cut_length, cut_length)]
        cuts.append(self._y[len(self.y)-cut_length:]) # last cut can be shorter than cut_length
        
        # Save each cut as a separate file in the created directory
        for i in tqdm(range(len(cuts))):
            filename = os.path.basename(self.filepath).rstrip('.wav')
            file_path = os.path.join(root, f"{filename}_cut_{i}.wav")
            write(file_path, self._sr, cuts[i])  # Ensure data is in int16 format for WAV
    

    def adjust_duration(self, duration: int) -> Tensor:
        """
        Adjust the duration of the sound file to a specified length by padding or truncating. It cuts the desired lenght of the audio file 
        centered around the max amplitude of the audio file. If the audio file is shorter than the desired length, it pads with zeros.
        :param duration: Desired duration in seconds
        """
        duration = duration * self._sr

        if len(self.y) < duration:
            # Pad with zeros
            padding = torch.zeros(duration - len(self._y))
            adj_y = torch.cat((padding, self.y)) 
        else:
            max_id = torch.argmax(self.y)
            if max_id < duration // 2: # Left of max_id is shorter than duration // 2, so we take from start to duration
                adj_y = self._y[:duration]
            elif max_id > len(self._y) - duration // 2: # Right of max_id is shorter than duration // 2, so we take from max_id - duration // 2 to end
                adj_y = self._y[len(self._y)-duration:] 
            else:
                adj_y = self._y[max_id - duration // 2 : max_id + duration // 2]
        
        return torch.Tensor(adj_y)

    
    @property
    def mfcc(self) -> Tensor:
        return self._mfcc.clone()
    
    @property
    def filepath(self) -> str:
        return self._filepath
    
    @property
    def y(self) -> Tensor:
        return torch.Tensor(self._y).clone()
    
    @property
    def sr(self) -> int:
        return self._sr
    

    