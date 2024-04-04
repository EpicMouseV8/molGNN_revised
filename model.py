import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

class GNN_QY(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, solvent_feature_dim, output_dim, dropout_rate=0.3):
        super(GNN_QY, self).__init__() 
        self.transformer_conv1 = TransformerConv(node_feature_dim, 128, edge_dim=edge_feature_dim, heads=4, dropout=dropout_rate)
        self.bn1 = BatchNorm1d(128 * 4) 
        self.transformer_conv2 = TransformerConv(128 * 4, 256, edge_dim=edge_feature_dim, heads=4, dropout=dropout_rate)
        self.bn2 = BatchNorm1d(256 * 4)  
        self.fc_solvent = Linear(solvent_feature_dim, 128)
        self.fc1 = Linear((256 * 4) * 2 + 128, 128)  
        self.bn_fc1 = BatchNorm1d(128)
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(128, output_dim)

    def forward(self, data, solvent_feature_dim=128):
        x, edge_index, edge_attr, batch, solvent_fingerprint = data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_fingerprint

        x = self.transformer_conv1(x, edge_index, edge_attr)
        x = F.relu(self.bn1(x))
        x = self.transformer_conv2(x, edge_index, edge_attr)
        x = F.relu(self.bn2(x))

        # combination of global mean pooling and global max pooling
        x_gap = global_mean_pool(x, batch)
        x_gmp = global_max_pool(x, batch)  
        x_combined = torch.cat([x_gap, x_gmp], dim=1) 

        # solvent features
        solvent_fingerprint = solvent_fingerprint.view(-1, solvent_feature_dim)
        solvent_features = F.relu(self.fc_solvent(solvent_fingerprint))

        # gnn features + solvent fingerprint
        x = torch.cat([x_combined, solvent_features], dim=1)

        # batchnorm and dropout
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = torch.sigmoid(self.fc2(x))

        return x