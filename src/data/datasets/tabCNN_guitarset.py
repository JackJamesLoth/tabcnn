import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

class TabCNN_GuitarSet(Dataset):
    def __init__(self,
                 file_names, 
                 data_path,
                 spec_repr="c", 
                 con_win_size=9
                 ):
        super().__init__()
        
        self.data_path = data_path
        self.spec_repr = spec_repr
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2

        self.file_names = self.getFilesFrames(file_names)

        # I thought this next line would speed things up but it doesn't :(
        #self.X, self.y = self.getDataNumpy()

    def getFilesFrames(self, file_names):
        log.info("Initializing dataset files")
        fileNames_withframes = []
        for f in tqdm(file_names):
            data_dir = self.data_path + self.spec_repr + "/"
            repr = np.load(data_dir + f)['repr']
            l = repr.shape[0]
            for i in range(l):
                fileNames_withframes.append(str(i) + "=" + f)
        return fileNames_withframes

    def getDataNumpy(self):
        
        data_dir = self.data_path + self.spec_repr + "/"
        if 'X.npy' in os.listdir(data_dir):
            log.info("Loading dataset numpy")
            X_numpy = np.load(data_dir + 'X.npy')
            y_numpy = np.load(data_dir + 'y.npy')
            
        else:
            log.info("Initializing dataset numpy")
            # 192 is number of CQT bins
            # TODO: Chagne this to be no hard coded, though it probably doesnt matter since I don't plan on changing that value
            X_numpy = np.zeros((len(self.file_names), self.con_win_size, 192))
            y_numpy = np.zeros((len(self.file_names), 6, 21))                     # 6 strings, 21 frets

            i = 0
            for id in tqdm(self.file_names):
                frame_idx, filename = id.split("=")
                frame_idx = int(frame_idx)

                # determine filename
                data_dir = self.data_path + self.spec_repr + "/"
                
                # load a context window centered around the frame index
                loaded = np.load(data_dir + filename)
                full_x = np.pad(loaded["repr"], [(self.halfwin,self.halfwin), (0,0)], mode='constant')
                sample_x = full_x[frame_idx : frame_idx + self.con_win_size]

                X_numpy[i] = sample_x
                y_numpy[i] = loaded["labels"][frame_idx]

                i+=1
            np.save(data_dir + 'X.npy', X_numpy)
            np.save(data_dir + 'y.npy', y_numpy)

        return X_numpy, y_numpy

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> torch.Tensor:

        ID = self.file_names[index]
        frame_idx, filename = ID.split("=")
        frame_idx = int(frame_idx)

        # determine filename
        data_dir = self.data_path + self.spec_repr + "/"
        
        
        # load a context window centered around the frame index
        loaded = np.load(data_dir + filename)
        full_x = np.pad(loaded["repr"], [(self.halfwin,self.halfwin), (0,0)], mode='constant')
        sample_x = full_x[frame_idx : frame_idx + self.con_win_size]
        X = np.expand_dims(sample_x, 0)

        # Store label
        y = loaded["labels"][frame_idx]
        #print('y: {}'.format(y.shape))
        is_no_class = np.sum(y, axis=1) == 0  # This will be True for rows that are all zeros
        class_indices = np.argmax(y, axis=1)
        class_indices[is_no_class] = y.shape[1]  # This assumes 0-based indexing, thus setting to 'j'
        #print('class_indices: {}'.format(class_indices.shape))
        #return X, class_indices

        return X, y

       # return np.expand_dims(self.X[index],0), self.y[index]
