import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from unimol_tools.data.conformer import ConformerGen, coords2unimol
from unimol_tools.models import UniMolModel


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_dataframe(df, target_col='cryst', test_size=0.2, val_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    train_inner_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df[target_col],
    )
    return train_inner_df, val_df, test_df


def build_unimol_dictionary(remove_hs=True):
    confgen = ConformerGen(data_type='molecule', remove_hs=remove_hs)
    dictionary = confgen.dictionary
    return confgen, dictionary, dictionary.pad()


def parse_xyz_string(xyz_text):
    lines = [line.strip() for line in str(xyz_text).splitlines() if line.strip()]
    n_atoms = int(lines[0])
    atom_lines = lines[2 : 2 + n_atoms]

    atoms = []
    coordinates = []
    for line in atom_lines:
        parts = line.split()
        atoms.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return {
        'atoms': atoms,
        'coordinates': np.array(coordinates, dtype=np.float32),
    }


def to_unimol_input(conf, dictionary, max_atoms=256, remove_hs=True):
    return coords2unimol(
        atoms=conf['atoms'],
        coordinates=np.array(conf['coordinates'], dtype=np.float32),
        dictionary=dictionary,
        max_atoms=max_atoms,
        remove_hs=remove_hs,
    )


class TripleUniMolDataset(Dataset):
    def __init__(
        self,
        df_part,
        dictionary,
        xyz_cols=('xyz_1_3d', 'xyz_2_3d', 'xyz_3_3d'),
        max_atoms=256,
        remove_hs=True,
    ):
        self.df = df_part.dropna(subset=list(xyz_cols)).reset_index(drop=True)
        self.dictionary = dictionary
        self.xyz_cols = xyz_cols
        self.max_atoms = max_atoms
        self.remove_hs = remove_hs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mols = []
        for col in self.xyz_cols:
            mols.append(
                to_unimol_input(
                    parse_xyz_string(row[col]),
                    self.dictionary,
                    self.max_atoms,
                    self.remove_hs,
                )
            )

        return {
            'mol1': mols[0],
            'mol2': mols[1],
            'mol3': mols[2],
            'y': float(row['cryst']),
        }


def pad_1d(arr_list, pad_value, dtype):
    max_len = max(len(x) for x in arr_list)
    batch = np.full((len(arr_list), max_len), pad_value, dtype=dtype)
    for i, x in enumerate(arr_list):
        batch[i, : len(x)] = x
    return batch


def pad_2d_square(arr_list, pad_value, dtype):
    max_len = max(x.shape[0] for x in arr_list)
    batch = np.full((len(arr_list), max_len, max_len), pad_value, dtype=dtype)
    for i, x in enumerate(arr_list):
        l = x.shape[0]
        batch[i, :l, :l] = x
    return batch


def pad_2d_coord(arr_list, pad_value, dtype):
    max_len = max(x.shape[0] for x in arr_list)
    batch = np.full((len(arr_list), max_len, 3), pad_value, dtype=dtype)
    for i, x in enumerate(arr_list):
        l = x.shape[0]
        batch[i, :l, :] = x
    return batch


def collate_single_mol(mol_list, pad_idx):
    src_tokens = [m['src_tokens'] for m in mol_list]
    src_distance = [m['src_distance'] for m in mol_list]
    src_coord = [m['src_coord'] for m in mol_list]
    src_edge_type = [m['src_edge_type'] for m in mol_list]

    return {
        'src_tokens': torch.tensor(pad_1d(src_tokens, pad_idx, np.int64), dtype=torch.long),
        'src_distance': torch.tensor(
            pad_2d_square(src_distance, 0.0, np.float32), dtype=torch.float32
        ),
        'src_coord': torch.tensor(pad_2d_coord(src_coord, 0.0, np.float32), dtype=torch.float32),
        'src_edge_type': torch.tensor(
            pad_2d_square(src_edge_type, 0, np.int64), dtype=torch.long
        ),
    }


def build_triple_collate_fn(pad_idx):
    def triple_collate_fn(batch):
        mol1_batch = collate_single_mol([item['mol1'] for item in batch], pad_idx)
        mol2_batch = collate_single_mol([item['mol2'] for item in batch], pad_idx)
        mol3_batch = collate_single_mol([item['mol3'] for item in batch], pad_idx)
        y_batch = torch.tensor([item['y'] for item in batch], dtype=torch.float32)
        return mol1_batch, mol2_batch, mol3_batch, y_batch

    return triple_collate_fn


def make_loader(dataset, pad_idx, batch_size=16, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=build_triple_collate_fn(pad_idx),
    )


class TripleUniMolClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = UniMolModel(data_type='molecule')
        self.repr_dim = None
        self.head = None
        self.dropout = dropout

    def encode_one(self, batch_dict):
        out = self.backbone(
            src_tokens=batch_dict['src_tokens'],
            src_distance=batch_dict['src_distance'],
            src_coord=batch_dict['src_coord'],
            src_edge_type=batch_dict['src_edge_type'],
            return_repr=True,
        )

        if self.repr_dim is None:
            self.repr_dim = out.shape[1]
            self.head = nn.Sequential(
                nn.Linear(self.repr_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, 1),
            ).to(out.device)

        return out

    def build_system_representation(self, z1, z2, z3):
        mol_pool = torch.stack([z1, z2, z3], dim=1).mean(dim=1)
        d12 = torch.abs(z1 - z2)
        d13 = torch.abs(z1 - z3)
        d23 = torch.abs(z2 - z3)
        pair_pool = torch.stack([d12, d13, d23], dim=1).mean(dim=1)
        x = torch.cat([mol_pool, pair_pool], dim=1)
        return x, mol_pool, pair_pool

    def forward(self, mol1, mol2, mol3):
        z1 = self.encode_one(mol1)
        z2 = self.encode_one(mol2)
        z3 = self.encode_one(mol3)
        x, _, _ = self.build_system_representation(z1, z2, z3)
        return self.head(x).squeeze(1)


def initialize_model(model, loader, device):
    mol1_b, mol2_b, mol3_b, _ = next(iter(loader))
    mol1_b = {k: v.to(device) for k, v in mol1_b.items()}
    mol2_b = {k: v.to(device) for k, v in mol2_b.items()}
    mol3_b = {k: v.to(device) for k, v in mol3_b.items()}
    with torch.no_grad():
        _ = model(mol1_b, mol2_b, mol3_b)
    return model


def save_stage_checkpoint(checkpoint_path, model, stage_name, stage_results, extra=None):
    parent = os.path.dirname(str(checkpoint_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    checkpoint = {
        'stage_name': stage_name,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'best_model_state': copy.deepcopy(stage_results['best_model_state']),
        'best_val_roc_auc': float(stage_results['best_val_roc_auc']),
        'best_val_ap': float(stage_results['best_val_ap']),
        'best_epoch': int(stage_results['best_epoch']),
        'history': copy.deepcopy(stage_results['history']),
        'repr_dim': int(model.repr_dim) if model.repr_dim is not None else None,
        'model_dropout': float(getattr(model, 'dropout', 0.3)),
    }
    if extra is not None:
        checkpoint['extra'] = copy.deepcopy(extra)

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_model_from_checkpoint(
    checkpoint_path,
    loader,
    device,
    map_location=None,
    dropout=None,
    state_key='best_model_state',
):
    if map_location is None:
        map_location = device

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_dropout = checkpoint.get('model_dropout', 0.3) if dropout is None else dropout

    model = TripleUniMolClassifier(dropout=model_dropout).to(device)
    model = initialize_model(model, loader, device)
    state = (
        checkpoint.get(state_key)
        or checkpoint.get('best_model_state')
        or checkpoint.get('model_state_dict')
    )
    model.load_state_dict(state)

    model.eval()
    return model, checkpoint


def compute_pos_weight(df, target_col='cryst'):
    y = df[target_col].values
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return n_neg, n_pos, n_neg / n_pos


def set_trainable_stage(model, stage='head_only'):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    last_layer = model.backbone.encoder.layers[-1]
    if stage == 'head_last_layer':
        for p in last_layer.parameters():
            p.requires_grad = True
    return last_layer


def build_optimizer(model, stage, head_lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4):
    last_layer = set_trainable_stage(model, stage=stage)
    if stage == 'head_only':
        return torch.optim.AdamW(model.head.parameters(), lr=head_lr, weight_decay=weight_decay)
    return torch.optim.AdamW(
        [
            {'params': model.head.parameters(), 'lr': head_lr},
            {'params': last_layer.parameters(), 'lr': backbone_lr},
        ],
        weight_decay=weight_decay,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for mol1, mol2, mol3, y in loader:
        mol1 = {k: v.to(device) for k, v in mol1.items()}
        mol2 = {k: v.to(device) for k, v in mol2.items()}
        mol3 = {k: v.to(device) for k, v in mol3.items()}
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(mol1, mol2, mol3)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for mol1, mol2, mol3, y in loader:
            mol1 = {k: v.to(device) for k, v in mol1.items()}
            mol2 = {k: v.to(device) for k, v in mol2.items()}
            mol3 = {k: v.to(device) for k, v in mol3.items()}
            y = y.to(device)

            logits = model(mol1, mol2, mol3)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    val_loss = total_loss / len(loader.dataset)
    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    probs = 1.0 / (1.0 + np.exp(-all_logits))

    return (
        val_loss,
        average_precision_score(all_targets, probs),
        roc_auc_score(all_targets, probs),
    )


def run_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    stage_name,
    optimizer,
    n_epochs,
    patience,
    scheduler=None,
    global_epoch_offset=0,
    clear_output_fn=None,
):
    stage_history = []
    best_stage_val_roc_auc = -1
    best_stage_val_ap = -1
    best_stage_epoch = 0
    best_stage_model_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_ap, val_roc = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_roc)

        global_epoch = global_epoch_offset + epoch + 1
        stage_history.append(
            {
                'global_epoch': global_epoch,
                'stage_epoch': epoch + 1,
                'stage_name': stage_name,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ap': val_ap,
                'val_roc_auc': val_roc,
            }
        )

        if val_roc > best_stage_val_roc_auc:
            best_stage_val_roc_auc = val_roc
            best_stage_val_ap = val_ap
            best_stage_epoch = epoch + 1
            best_stage_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if clear_output_fn is not None:
            clear_output_fn(wait=True)

        print(f'Stage: {stage_name}')
        print(f'Epoch: {epoch + 1}/{n_epochs} | Global epoch: {global_epoch}')
        print(f'train_loss = {train_loss:.4f}')
        print(f'val_loss   = {val_loss:.4f}')
        print(f'val_AP     = {val_ap:.4f}')
        print(f'val_ROC_AUC= {val_roc:.4f}')
        print(f'best_stage_val_ROC_AUC = {best_stage_val_roc_auc:.4f}')
        print(f'best_stage_epoch       = {best_stage_epoch}')

        if patience_counter >= patience:
            print(f'Early stopping in stage \'{stage_name}\' at epoch {epoch + 1}')
            break

    return {
        'history': stage_history,
        'best_val_roc_auc': best_stage_val_roc_auc,
        'best_val_ap': best_stage_val_ap,
        'best_epoch': best_stage_epoch,
        'best_model_state': best_stage_model_state,
    }


def summarize_two_stage_training(stage1_results, stage2_results):
    full_history = stage1_results['history'] + stage2_results['history']
    history_df = pd.DataFrame(full_history)

    if stage1_results['best_val_roc_auc'] >= stage2_results['best_val_roc_auc']:
        best_model_state = copy.deepcopy(stage1_results['best_model_state'])
        best_val_roc_auc = float(stage1_results['best_val_roc_auc'])
        best_val_ap = float(stage1_results['best_val_ap'])
        best_epoch = int(stage1_results['best_epoch'])
        best_stage_name = 'stage1'
    else:
        best_model_state = copy.deepcopy(stage2_results['best_model_state'])
        best_val_roc_auc = float(stage2_results['best_val_roc_auc'])
        best_val_ap = float(stage2_results['best_val_ap'])
        best_epoch = int(len(stage1_results['history']) + stage2_results['best_epoch'])
        best_stage_name = 'stage2'

    return {
        'history_df': history_df,
        'best_model_state': best_model_state,
        'best_val_roc_auc': best_val_roc_auc,
        'best_val_ap': best_val_ap,
        'best_epoch': best_epoch,
        'best_stage_name': best_stage_name,
    }


def collect_probs_targets(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for mol1, mol2, mol3, y in loader:
            mol1 = {k: v.to(device) for k, v in mol1.items()}
            mol2 = {k: v.to(device) for k, v in mol2.items()}
            mol3 = {k: v.to(device) for k, v in mol3.items()}
            logits = model(mol1, mol2, mol3)
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    probs = np.where(
        all_logits >= 0,
        1.0 / (1.0 + np.exp(-all_logits)),
        np.exp(all_logits) / (1.0 + np.exp(all_logits)),
    )
    return probs, all_targets


def find_best_threshold(y_true, probs, grid=None, beta=2.0):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_fbeta = -1.0
    best_precision = 0.0
    best_recall = 0.0
    rows = []

    for thr in grid:
        preds = (probs >= thr).astype(int)
        fbeta = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        rows.append(
            {
                'threshold': float(thr),
                'f_beta': float(fbeta),
                'precision': float(precision),
                'recall': float(recall),
            }
        )

        if (fbeta > best_fbeta) or (np.isclose(fbeta, best_fbeta) and recall > best_recall):
            best_threshold = float(thr)
            best_fbeta = float(fbeta)
            best_precision = float(precision)
            best_recall = float(recall)

    return best_threshold, best_fbeta, best_precision, best_recall, pd.DataFrame(rows)

def evaluate_loader(model, loader, device, title, threshold=0.5, ax=None):
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for mol1, mol2, mol3, y in loader:
            mol1 = {k: v.to(device) for k, v in mol1.items()}
            mol2 = {k: v.to(device) for k, v in mol2.items()}
            mol3 = {k: v.to(device) for k, v in mol3.items()}
            logits = model(mol1, mol2, mol3)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= threshold).astype(int)

    roc_auc = roc_auc_score(all_targets, probs)
    precision = precision_score(all_targets, preds, zero_division=0)
    recall = recall_score(all_targets, preds, zero_division=0)
    cm = confusion_matrix(all_targets, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', labelsize=14)

    for text in np.ravel(disp.text_):
        if text is not None:
            text.set_fontsize(16)

    print(title)
    print(f'ROC-AUC  : {roc_auc:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'Precision: {precision:.4f}')

    return {
        'title': title,
        'roc_auc': roc_auc,
        'recall': recall,
        'precision': precision,
        'cm': cm,
        'probs': probs,
        'preds': preds,
    }


def extract_embeddings(model, loader, dataset_df, dataset_name, device, threshold=0.5):
    model.eval()

    z1_all, z2_all, z3_all = [], [], []
    mol_pool_all, pair_pool_all, system_repr_all = [], [], []
    logits_all, probs_all, preds_all, y_all = [], [], [], []

    with torch.no_grad():
        for mol1, mol2, mol3, y in loader:
            mol1 = {k: v.to(device) for k, v in mol1.items()}
            mol2 = {k: v.to(device) for k, v in mol2.items()}
            mol3 = {k: v.to(device) for k, v in mol3.items()}

            z1 = model.encode_one(mol1)
            z2 = model.encode_one(mol2)
            z3 = model.encode_one(mol3)
            system_repr, mol_pool, pair_pool = model.build_system_representation(z1, z2, z3)
            logits = model.head(system_repr).squeeze(1)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            z1_all.append(z1.cpu().numpy())
            z2_all.append(z2.cpu().numpy())
            z3_all.append(z3.cpu().numpy())
            mol_pool_all.append(mol_pool.cpu().numpy())
            pair_pool_all.append(pair_pool.cpu().numpy())
            system_repr_all.append(system_repr.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            y_all.append(y.numpy())

    out_df = dataset_df.copy().reset_index(drop=True)
    out_df['pooled_unimol_embedding'] = list(np.vstack(system_repr_all))
    out_df['molecule_pool_embedding'] = list(np.vstack(mol_pool_all))
    out_df['pair_pool_embedding'] = list(np.vstack(pair_pool_all))
    out_df['z1_embedding'] = list(np.vstack(z1_all))
    out_df['z2_embedding'] = list(np.vstack(z2_all))
    out_df['z3_embedding'] = list(np.vstack(z3_all))
    out_df['logit'] = np.concatenate(logits_all)
    out_df['prob'] = np.concatenate(probs_all)
    out_df['pred'] = np.concatenate(preds_all)
    out_df['target'] = np.concatenate(y_all)
    out_df['used_threshold'] = float(threshold)
    out_df['dataset_name'] = dataset_name
    return out_df


def save_experiment_outputs(
    save_dir,
    model,
    best_model_state,
    best_threshold,
):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'best_model_state': best_model_state,
        'best_threshold': float(best_threshold),
        'model_dropout': float(getattr(model, 'dropout', 0.3)),
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pt'))
