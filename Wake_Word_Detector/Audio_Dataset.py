from torch.utils.data import Dataset
from Data_Handler.process_sounddata import AugmentSoundData as asd
import torchaudio
import torch 



class audioDataset(Dataset):

    def __init__(self, filepaths, labels=None, transform=None, training=False):
        """
        Args:
            filepaths (list): List of file paths to the audio files.
            labels (list, optional): List of labels corresponding to the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mfcc = torch.tensor([asd(file).mfcc for file in filepaths]) #We will only keep the MFCC features since we will work with those in our model
        self.labels = labels
        self.transform = transform
        self.training = training
        self.filepaths = filepaths
        self.sr = 16000

    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        return