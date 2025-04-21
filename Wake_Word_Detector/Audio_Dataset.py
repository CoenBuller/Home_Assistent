from torch.utils.data import Dataset 
import os

""""
    This class can be used to load in a custom dataset of audio files. It is important that the root_dir only contains sub directories, where each sub directory
    only contains audio files of the same class. This assures correct labeling of the data. If you choose to give a custom label list, this list must be 
    the same length as the number of sub directories in the root_dir. The labels must be unique and in the same order as the sub directories.
    The transform function is used to apply any transformations to the audio files. This can be used to augment the data, for example by adding noise or 
    changing the pitch.
"""

class audioDataset(Dataset):

    def __init__(self, root_dir, labels=None, transformers=None):
        """
        Args:
            root_dir (Path): Path to the root directory containing all the sub directories.
            labels (list, optional): List of labels corresponding to the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self._labels = labels
        self.transformers = transformers
        self.root_dir = root_dir
        self.sub_dirs= os.listdir(root_dir)
        self.labels_dict = self.label_dirs()
        self.labeled_files = self.label_files()

    
    def label_dirs(self):
        """"
        Label the sub directories in the root directory. The labels are either given by the user or generated automatically.
        """
        
        if not all(os.path.isdir(os.path.join(self.root_dir, sub_dir)) for sub_dir in self.sub_dirs): 
            raise ValueError("All items in the root directory must be sub directories.")

        if self._labels is not None and len(self._labels) != len(self.sub_dirs): 
            raise ValueError("The number of labels must be equal to the number of sub directories.")

        i = 0
        labels_dict = {}
        for dir in self.sub_dirs:
            if self._labels is None:
                labels_dict[dir] = i
            else:
                labels_dict[dir] = self._labels[i]
            i += 1

        return labels_dict
        
    def label_files(self):
        """
        Label the audio files in the sub directories.
        """
        file_dict = {}
        for dir in self.sub_dirs:
            for file in os.listdir(os.path.join(self.root_dir, dir)):
                if os.path.isfile(os.path.join(self.root_dir, dir, file)): #only accepts .mp3 and .wav files for now
                    file_dict[file] = self.labels_dict[dir]
                elif os.path.isdir(file):
                    raise ValueError("The sub directories of the root directory cannot contain new directories.")
                else:
                    raise ValueError("All items in the sub directories must be .mp3 or .wav.")
        
        return file_dict

    def __len__(self):
        return len(self.labeled_files)
    

    def __getitem__(self, idx):

        file_name = list(self.labeled_files.keys())[idx]
        label = self.labeled_files[file_name]
        label_dir = list(self.labels_dict.keys())[list(self.labels_dict.values()) == label]
        audio_path = os.path.join(self.root_dir, label_dir, file_name) #Use full path to assure correct loading of the file

        if self.transformers:
            x = audio_path
            for transform in self.transformers:
                x = transform(x)
            transformed = x
            return audio_path, label, transformed
            
        return audio_path, label
    
    def __str__(self):
        print(f"Audio dataset: {self.__class__.__name__}(")
        print(f"    Number of classes: {len(self.sub_dirs)}")
        print(f"    Number of files: {len(self.labeled_files)}")
        print(f"    Transformer: {self.transformers}\n    )")
        architecture_str = f"Directory architecture: \n{os.path.basename(self.root_dir)}\n"
        for dir in self.sub_dirs:
            architecture_str += f" {' '*int(len(os.path.basename(self.root_dir))/4)}| \n{' '*int(len(os.path.basename(self.root_dir))/4)} |- {dir}\n"
        return architecture_str

    @property
    def files(self):
        return self.labeled_files

    @property
    def labels(self):
        return self.labels_dict
    

    

