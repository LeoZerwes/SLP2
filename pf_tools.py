import os 
from pathlib import Path 
import pickle
import warnings

import torch
from torch import nn, optim
import numpy as np
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_tar

import librosa
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import librosa.display

import datetime
import csv

class CheckThisCell(Exception):
    pass

class ETS(Dataset):

    RELEASE_CONFIGS = {'train100': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf25/lab2/train100.tgz' , 'checksum':'666788bc7858479edcef6f1544cda9da336f6b56b8f25b5bdef1791322a79145'},
                'train': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf25/lab2/train.tgz' , 'checksum':'30b8b0c2afd8e7aec58138148adcd4263bfd287f8a01fd0ee6f02273fd1dbed4'},
                'dev': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf25/lab2/dev.tgz' , 'checksum':'b77ca6e0762971726849fb4a4f1aea6cd31376551cd5b6820816e4b537c7ce52'},
                'evl': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf25/lab2/evl.tgz' , 'checksum':'c86bc79d0cdbd0488b549d72dd31684f65c61d03f08ebad47e96f42b29d108e1'}
                }
                   
    def __init__(self, root : str, dataset_id: str, transform_id: str = "feat", audio_transform : callable = None, chunk_size : int = -1, chunk_hop : int = -1, chunk_transform : callable = None) -> None:
        
        if dataset_id not in ETS.RELEASE_CONFIGS:
            raise ValueError("Not known data set in ETS")
        
        if audio_transform is None:
            raise ValueError("Need to define some tranformation from audiofile to features")

        self.path = Path(root) / dataset_id
        self.url = ETS.RELEASE_CONFIGS[dataset_id]['url']

        self.archive = os.path.basename(self.url)
        self.archive = Path(root) / self.archive

        self.audio_dir = self.path / 'audio'
        self.feat_dir = self.path / transform_id
        self.key_file = self.path / 'key.lst'

        self.audio_transform = audio_transform
        self.chunk_size = chunk_size
        self.chunk_hop = chunk_hop
        self.chunk_transform = chunk_transform

        if self.chunk_size > 0 and self.chunk_hop <= 0:
            self.chunk_hop = self.chunk_size

        self.download_data(ETS.RELEASE_CONFIGS[dataset_id]['checksum'])
        self.data_to_feat()

        if not os.path.isfile(self.key_file):
            raise RuntimeError("Key file does not exist. There was some problem downloading data.")

        if not os.path.isdir(self.feat_dir):
            raise RuntimeError("Features directory does not exist. There was some problem applying feature extraction.")
    
        self._walker = []
        with open(self.key_file) as file:
            for line in file:
                basename, label = line.split()[0].strip(), line.split()[1].strip()
                self._walker.extend([(c, basename, label) for c in os.listdir(self.feat_dir / basename )])
    
    def __getitem__(self, index):
        featIn, basename, label = self._walker[index]

        feats = pickle.load(open(self.feat_dir / basename / featIn, 'rb'))
        
        return feats, label, basename
    
    def __len__(self):
        return len(self._walker)
                   
    def download_data(self,  checksum : str = None) -> None:   
        if not os.path.isdir(self.path):
            if not os.path.isfile(self.archive):
                download_url_to_file(self.url, self.archive, hash_prefix=checksum)
            _extract_tar(self.archive)

    def data_to_feat(self) -> None:
        
        if not os.path.isfile(self.key_file):
            raise RuntimeError("Key file does not exist. Please download it first. There was some problem during data download and extreaction.")

        if os.path.isdir(self.feat_dir):
            warnings.warn("The feature directory already exists, and no new feature extraction will be performed.")
        else:
            if not os.path.isdir(self.audio_dir):
                raise RuntimeError("Audios directory does not exist. There was some problem during data download and extreaction.")
   
            ## Create feature directory
            os.mkdir(self.feat_dir)

            # Extract features
            with open(self.key_file) as file:
                for line in tqdm(file.readlines()):
                    basename, _ = line.split()[0].strip(), line.split()[1].strip()
                    audioin = self.audio_dir / f'{basename}.wav'
                    featOutPath = self.feat_dir / f'{basename}' 
                    
                    if not os.path.isdir(featOutPath):
                        os.mkdir(featOutPath)
                    
                    # print(f'\t{audioin}...', end='')
                    feats = self.audio_transform(str(audioin)) # The output of this is (Ntime x Dim)

                    finish = False
                    if self.chunk_size > 0:
                        for b in range(0, feats.shape[0], self.chunk_hop-1):
                            this_feats = feats[b:b+self.chunk_size]
                            if this_feats.shape[0] < self.chunk_size:
                                this_feats = np.concatenate((this_feats, np.zeros((self.chunk_size-this_feats.shape[0], this_feats.shape[1]), dtype=this_feats.dtype)))
                                finish = True
                            if self.chunk_transform is not None:
                                this_feats = self.chunk_transform(this_feats)

                            pickle.dump(this_feats, open(featOutPath / f'{basename}.{b//self.chunk_hop}.feat' , 'wb'))
                            if finish:
                                break 
                    else:
                        pickle.dump(feats, open(featOutPath / f'{basename}.feat' , 'wb'))
                            
## Auxiliary functions

# Plot audio waveform and spectrogram
def audioplot(filename, sr=16000, mono=True, duration=None):

    x, sr = librosa.load(filename, sr=sr, mono=mono, duration=duration)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
    
    librosa.display.waveshow(x, sr=sr,  ax=ax[0])
    ax[0].set(title='Waveform')
    ax[0].label_outer()
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax[1])
    ax[1].set(title='Linear-frequency power spectrogram')
    ax[1].label_outer()     
    
    return 

def prepare_ets_data(ets_data, collapse_samples=True, expand_labels=True):
    """
    This function permits preparing ETS data for both training and prediction:
    
        - `collapse_samples` permits concanating all the ets_data in a single dictionary with fields 
        'data' and 'label' containing all the data and labels respectively. If False, it returns a 
        dictionary with keys the identifer of esch feature file and with value a 'data' and 'label' 
        dictionary.
        
        - expand_labels permits to expand the labels with the same size of the corresponding data. 
        If False, the labels are kept as they are (one per feature file). Only used if collapse_samples is False.

    """
    
    if collapse_samples:
        train_data = []
        train_labels = []
        train_identifiers = []
        for data, label, basename in ets_data:
                train_data.append(data)
                train_labels.append(np.full(data.shape[0], label)) 
                train_identifiers.append(np.full(data.shape[0], basename)) 
            
                
        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)
        train_identifiers = np.concatenate(train_identifiers)

        return {'data':train_data, 'label': train_labels, 'identifiers': train_identifiers}
    else:
        dev_data = {}
        for data, label, basename in ets_data:
                if basename not in dev_data:
                    if expand_labels:
                        dev_data[basename] = {'data':[], 'label':[]}
                    else:
                        dev_data[basename] = {'data':[], 'label':label}
                        
                dev_data[basename]['data'].append(data)
                if expand_labels:
                    dev_data[basename]['label'].append(np.full(data.shape[0], label)) 
            
        ## We concatenate all the frames belonging to the same filename
        for basename in dev_data:
                dev_data[basename]['data'] = np.concatenate(dev_data[basename]['data'])
                if expand_labels:
                    dev_data[basename]['label'] = np.concatenate(dev_data[basename]['label'])
                    
        return dev_data
    

def save_model(model, model_id, path):
    """
    Save the model to a file.
    """
  
    if not os.path.isdir(path):
        os.mkdir(path)

    now = str(datetime.datetime.now()).replace(' ','_').split('.')[0]
        
    model_name = f'{model_id}_{now}'
    os.mkdir(f'{path}/{model_name}/')

    filename = f'{path}/{model_name}/model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    
    print(f"Model saved to {filename}")
    return model_name

def load_model(filename):
    """
    Load the model from a file.
    """
    if not os.path.isfile(filename):
        raise RuntimeError(f"Model file {filename} does not exist.")
    
    model = pickle.load(open(filename, 'rb'))
    return model


def gmm_predict(models, dev_data, LANGUAGES):
    """
    Predict the language of the input data using the GMM model.
    """

    results_dev = {}
    results_dev['ref'] =  list()
    results_dev['hyp'] =  list()
    results_dev['llhs'] = np.empty((len(dev_data), len(LANGUAGES)), dtype=np.float64)
    results_dev['fileids'] = list()

    for i, fileid in tqdm(enumerate(sorted(dev_data)), total=len(dev_data)):
        data = dev_data[fileid]['data']  # the features
        
        results_dev['fileids'].append(fileid)     #fileid

        # store the reference. Notice that we only have this for the dev set, not for the eval
        results_dev['ref'].append(dev_data[fileid]['label']) #reference

        # obtain the log-likelihood score for each model and store
        results_dev['llhs'][i,:] = np.array([models[lang].score(data) for lang in LANGUAGES])

        
        # Obtain the maximum likelihood nativelanguge estimation
        ix = np.argmax(results_dev['llhs'][i,:])
        results_dev['hyp'].append(LANGUAGES[ix])
        
    return results_dev
    
def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.matshow(cm, cmap=cmap, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center')

    # Set axis labels and ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)


    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title(title, fontsize=12)
    plt.show()
    
    
    
def create_submission_file(dirname, filename, lang2id = None):

    if lang2id is None:
        mapping = lambda x : x 
    else:
        mapping = lambda x : lang2id[x]
        
    filename_dev = f'{dirname}/dev.pkl'
    filename_evl = f'{dirname}/evl.pkl'

    results_dev = pickle.load(open(filename_dev, 'rb'))
    results_evl = pickle.load(open(filename_evl, 'rb'))


    with open(filename, 'w') as file:
        csv_writer = csv.writer(file) # CSV writer
        csv_writer.writerow(('fileId', 'Lang')) # Header of the CSV

        # Save dev results
        for i in range(len(results_dev['fileids'])):
            csv_writer.writerow((results_dev['fileids'][i], mapping(results_dev['hyp'][i])))
        # Save evl results
        for i in range(len(results_evl['fileids'])):
            csv_writer.writerow((results_evl['fileids'][i], mapping(results_evl['hyp'][i])))
            
def predict_nn(model, dataset, class2id=None):
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False
    )
    predictions = []
    references = []
    fileids = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_files in test_loader:
            outputs = model(batch_X.squeeze(dim=1))
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.item())
            references.append(batch_y[0] if class2id is None else class2id[batch_y[0]])
            fileids.append(batch_files[0])
    
    return np.array(predictions), np.array(references), np.array(fileids)

def train_nn(model, traindataset, testdataset, class2id=None, batch_size=16, epochs=200, lr=0.005, momentum=0.9, weight_decay=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    if class2id is None:
        raise ValueError("class2id must be provided")
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
            dataset=traindataset,
            batch_size=batch_size,
            shuffle=True
    )
    for epoch in range(epochs):
        for batch_X, batch_y, _ in train_loader:
            outputs = model(batch_X.squeeze(dim=1))
            loss = criterion(outputs, torch.tensor(list(map(lambda x:class2id[x],batch_y))))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
            if testdataset is not None:
                hyp, ref, _ = predict_nn(model, testdataset, class2id=class2id)
                accuracy = torch.tensor(hyp == ref).float().mean()
                print(f'Dev Accuracy: {accuracy.item() * 100:.2f}%')
                