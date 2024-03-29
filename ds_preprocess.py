from torch_geometric.data import Dataset
import numpy as np 
import pandas as pd
import torch
import torch_geometric
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from tqdm import tqdm
import logging


class MolDataset(Dataset):
    def __init__(self, root, filename, target='Quantum yield', mode = 'train', transform=None, pre_transform=None):

        self.filename = filename
        self.mode = mode
        self.target = target
        self.processed_dir = os.path.join(root, "processed", target.replace(" ", "_"))
        
        super(MolDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_dir(self):
        return self._processed_dir
    
    @processed_dir.setter
    def processed_dir(self, value):
        self._processed_dir = value

    @property
    def raw_file_names(self):
        return self.filename


    @property
    def processed_file_names(self):
        
        dataset = pd.read_csv(self.raw_paths[0])
        dataset = dataset.dropna(subset=[self.target])

        self.data = dataset
        

        if self.mode == 'test':
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        elif self.mode == 'val':
            return [f'data_val_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass


    def process(self):
        
        dataset = pd.read_csv(self.raw_paths[0])
        dataset = dataset.dropna(subset=[self.target])

        self.data = dataset

        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

        for i, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            chromophore = row['Chromophore']
            solvent = row['Solvent']
            target = row[self.target]

            chromophore_mol = Chem.MolFromSmiles(chromophore)

            if chromophore_mol is None:
                logging.warning(f"Failed to parse molecule: {chromophore}")
                continue

            features = featurizer._featurize(chromophore_mol)

            if features.node_features is None or features.edge_features is None:
                logging.warning(f"Failed to featurize molecule: {chromophore}")
                continue

            node_features = torch.tensor(features.node_features, dtype=torch.float)
            edge_features = torch.tensor(features.edge_features, dtype=torch.float)
            edge_index = torch.tensor(features.edge_index, dtype=torch.long).t().contiguous()

            solvent_mol = Chem.MolFromSmiles(solvent)

            if solvent_mol is None:
                logging.warning(f"Failed to parse solvent molecule: {solvent}")
                continue

            solvent_fingerprint = AllChem.GetMorganFingerprintAsBitVect(solvent_mol, radius=2, nBits=128)
            solvent_fingerprint = torch.tensor(solvent_fingerprint, dtype=torch.float)

            data = torch_geometric.data.Data(x=node_features, solvent_fingerprint=solvent_fingerprint, edge_index=edge_index.t().contiguous(), edge_attr=edge_features, y=torch.tensor(target, dtype=torch.float))

            if self.mode == 'test':
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{i}.pt'))
            elif self.mode == 'val':
                torch.save(data, os.path.join(self.processed_dir, f'data_val_{i}.pt'))
            else:
                torch.save(data,  os.path.join(self.processed_dir, f'data_{i}.pt'))


    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    

    def len(self):
        return len(self.processed_file_names)
    

    def get(self, idx):
        if self.mode == 'test':
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        elif self.mode == 'val':
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_val_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   

        # print(f"Graph {idx}: Node features shape: {data.x.shape}, Edge features shape: {data.edge_attr.shape}, Edge index shape: {data.edge_index.shape}")
     
        return data