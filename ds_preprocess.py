from torch_geometric.data import Dataset, InMemoryDataset
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

def preprocess(df, target_column):
    print("Preprocessing data...")

    df = df.dropna(subset=target_column)

    chromophores = df['Chromophore'].to_numpy()
    solvents = df['Solvent'].to_numpy()
    target = df[target_column].to_numpy()

    print("Data preprocessed.")

    return chromophores, solvents, target

def featurize_impute(chromophores, solvents, save_filename):
    
    data = []

    save_dir = 'data/processed/imputed/' + save_filename
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_files_exist = True
    for i in range(len(chromophores)):
        save_path = os.path.join(save_dir, f'graph_{i}.pt')
        if not os.path.isfile(save_path):
            all_files_exist = False
            break
        else:
            graph = torch.load(save_path)
            data.append(graph)

    if all_files_exist:
        print("Loaded preprocessed graphs from disk.")

        for graph in data:
            graph.edge_index = graph.edge_index.t()

        return data


    print("Featurizing data...")
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    features = featurizer.featurize(chromophores)

    print("Data featurized.")

    print("Creating graph data...")

    for i, feat in enumerate(features):
        node_features = torch.tensor(feat.node_features, dtype=torch.float)
        edge_features = torch.tensor(feat.edge_features, dtype=torch.float)
        edge_index = torch.tensor(feat.edge_index, dtype=torch.long)

        chromo_smiles = chromophores[i]
        solvent_smiles = solvents[i]
        solvent_mol = Chem.MolFromSmiles(solvent_smiles)
        solvent_fingerprint = AllChem.GetMorganFingerprintAsBitVect(solvent_mol, radius=2, nBits=128)
        solvent_fingerprint = torch.tensor((solvent_fingerprint), dtype=torch.float)

        # y = torch.tensor([targets[i]], dtype=torch.float)

        graph = torch_geometric.data.Data(x=node_features, solvent_fingerprint = solvent_fingerprint, edge_index=edge_index.t().contiguous(), edge_attr=edge_features, 
                     chromo_smiles=chromo_smiles, solvent_smiles = solvent_smiles)
        data.append(graph)

        # Save each graph object to a file
        save_path = os.path.join(save_dir, f'graph_{i}.pt')
        torch.save(graph, save_path)

    for graph in data:
        graph.edge_index = graph.edge_index.t()

    return data

def featurize(chromophores, solvents, target, save_filename):
    
    data = []

    save_dir = 'data/processed/list/' + save_filename
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_files_exist = True
    for i in range(len(chromophores)):
        save_path = os.path.join(save_dir, f'graph_{i}.pt')
        if not os.path.isfile(save_path):
            all_files_exist = False
            break
        else:
            graph = torch.load(save_path)
            data.append(graph)

    if all_files_exist:
        print("Loaded preprocessed graphs from disk.")

        for graph in data:
            graph.edge_index = graph.edge_index.t()

        return data


    print("Featurizing data...")
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    features = featurizer.featurize(chromophores)

    print("Data featurized.")

    print("Creating graph data...")

    for i, feat in enumerate(features):
        node_features = torch.tensor(feat.node_features, dtype=torch.float)
        edge_features = torch.tensor(feat.edge_features, dtype=torch.float)
        edge_index = torch.tensor(feat.edge_index, dtype=torch.long)

        chromo_smiles = chromophores[i]
        solvent_smiles = solvents[i]
        solvent_mol = Chem.MolFromSmiles(solvent_smiles)
        solvent_fingerprint = AllChem.GetMorganFingerprintAsBitVect(solvent_mol, radius=2, nBits=128)
        solvent_fingerprint = torch.tensor((solvent_fingerprint), dtype=torch.float)

        # y = torch.tensor([targets[i]], dtype=torch.float)

        graph = torch_geometric.data.Data(x=node_features, solvent_fingerprint = solvent_fingerprint, edge_index=edge_index.t().contiguous(), edge_attr=edge_features, 
                     chromo_smiles=chromo_smiles, solvent_smiles = solvent_smiles,  y=torch.tensor(target[i], dtype=torch.float))
        data.append(graph)

        # Save each graph object to a file
        save_path = os.path.join(save_dir, f'graph_{i}.pt')
        torch.save(graph, save_path)

    for graph in data:
        graph.edge_index = graph.edge_index.t()

    return data

def good_old_way_of_doing_things(file_name, target_column):
    path = 'data/raw/' + file_name
    df = pd.read_csv(path)
    # df = df[:500]
    chromophores, solvents, target = preprocess(df, target_column)
    data = featurize(chromophores, solvents, target, save_filename=target_column.replace(" ", "_"))

    return data


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
     
        return data