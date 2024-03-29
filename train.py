import torch
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
from ds_preprocess import MolDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from model import GNN_QY
from utilities import plot_losses, plot_predictions_vs_actual
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

def test(model, test_loader, target, loss_fn):

    model.eval()
    labels = []
    preds = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            pred = model(batch, solvent_feature_dim=128)
            loss = loss_fn(pred, batch.y.float())
            total_loss += loss.item()

            preds.append(pred.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    labels = np.concatenate(labels)
    preds = np.concatenate(preds)

    calc_metrics(labels, preds)

    plot_predictions_vs_actual(labels, preds, save_path='visualizations/' + target.replace(" ", "_") + '_predictions_vs_actual.png')

    return total_loss / len(test_loader)

def calc_metrics(labels, preds):
    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)

    print(f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R^2={r2:.4f}")

def train_epoch(model, train_loader, optimizer, loss_fn):

    model.train()
    num_targets = 1
    total_loss = 0.0

    for batch in train_loader:
        #data to GPU
        batch.to(device)
        #resetting gradients
        optimizer.zero_grad()
        #passing data through model
        output = model(batch, solvent_feature_dim=128)
        #reshaping in case of multiple targets
        batch.y = batch.y.view(-1, num_targets) # Reshape targets to match output shape
        #computing loss and backpropagating
        loss = loss_fn(output, batch.y)
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item()

    return total_loss / len(train_loader)

def eval_model(model, loader, loss_fn):
    
        model.eval()
        num_targets = 1
        total_loss = 0.0
    
        with torch.no_grad():
            for batch in loader:
                batch.to(device)
                output = model(batch, solvent_feature_dim=128)
                batch.y = batch.y.view(-1, num_targets)  # Reshape targets to match output shape
                loss = loss_fn(output, batch.y)
                total_loss += loss.item()
    
        return total_loss / len(loader)


def run_training(model_path=None, dataset = 'prep2.csv', target='Quantum yield', n_epochs = 300, bz = 32):

    print("Loading data...")

    data = pd.read_csv('data/raw/'+dataset)

    # data = data[:500]

    data = data.dropna(subset=[target])

    data_train, data_test_val = train_test_split(data, test_size= 1 - train_ratio, random_state=0)
    data_test, data_val = train_test_split(data_test_val, test_size=test_ratio/(test_ratio + validation_ratio), random_state=0)

    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)

    data_train.to_csv('data/raw/train_'+target+'.csv', index=True)
    data_val.to_csv('data/raw/val_'+target+'.csv', index=True)
    data_test.to_csv('data/raw/test_'+target+'.csv', index=True)

    print(data_train.index)

    train_ds = MolDataset(root = "data/", filename="train_"+target+".csv", mode='train')
    val_ds = MolDataset(root = "data/", filename="val_"+target+".csv", mode='val')
    test_ds = MolDataset(root = "data/", filename="test_"+target+".csv", mode='test')

    print("Data loaded.")
    print("Number of training samples: ", len(train_ds))
    print("Number of test samples: ", len(test_ds))

    # print(train_ds[0])

    print("Creating data loaders...")

    train_loader = DataLoader(train_ds, batch_size=bz, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bz, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bz, shuffle=False)

    print("Data loaders created.")

    node_feature_dim = train_ds.num_node_features
    edge_feature_dim = train_ds.num_edge_features


    model = GNN_QY(node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim, solvent_feature_dim=128, output_dim=1, dropout_rate=0.3).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = MSELoss()

    save_path = 'models/'+ target.replace(" ", "_")
    model_name = 'model_'+ target.replace(" ", "_")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Training model...")

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 50
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = eval_model(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path + '/' + 'model_es.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break

        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + '/' + model_name + '_epoch_' + str(epoch) + '.pth')

    # saving loss values as a .csv
    loss_data = {
        'train_loss': train_losses,
        'val_loss': val_losses,
    }

    df_losses = pd.DataFrame(loss_data)

    df_losses.to_csv(save_path + '/' + target.replace(" ", "_") + '_losses.csv', index=False)

    plot_losses(train_losses, val_losses, save_path='visualizations/'+target.replace(" ", "_")+'_losses.png')

    model.load_state_dict(torch.load('model_es.pth'))
    test_loss = test(model, test_loader, target, loss_fn)
    print(f"Test Loss: {test_loss}")

