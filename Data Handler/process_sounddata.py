import torch 
import librosa as lb
import random
import torchaudio.transforms as T

class AugmentSoundData():

    def __init__(self, filepath, sr=16000):
        self.filepath = filepath
        self.y, self.sr = lb.load(filepath, sr=sr)
        self.mfcc = self.extract_mfcc()

    def extract_mfcc(self, n_features=13):
        """
        Extract MFCC features from the audio file.
        :param filepath: Path to the audio file
        :param sr: Sample rate
        :param n_features: Number of MFCC features to extract
        :return: MFCC tensor of shape (n_mfcc, time)
        """

        mfcc = lb.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_features)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        return mfcc
    
    def Augment_mfcc(self):         
        """
        Apply random corruptions to an MFCC tensor.
        :param mfcc: Tensor of shape (n_mfcc, time)
        :return: Augmented MFCC tensor (same shape)
        """
        aug = self.mfcc.clone()

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


