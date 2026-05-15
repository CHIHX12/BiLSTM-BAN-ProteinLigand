import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein, encode_sequence_with_features, encode_selfies, encode_selfies_with_features


def collate_selfies_fn(batch):
    """
    自定義 collate 函數用於處理 SELFIES 序列批次
    當藥物使用 SELFIES（序列）表示時使用此函數

    Args:
        batch: [(v_d, v_p, y), ...] 其中：
            v_d = (drug_idx_tensor, drug_len_tensor) 或 (drug_idx_tensor, drug_feat_tensor, drug_len_tensor)

    Returns:
        如果 v_d 是 tuple（SELFIES）：
            v_d: (stacked_drug_idx, stacked_drug_len) 或 (stacked_drug_idx, stacked_drug_feat, stacked_drug_len)
        如果 v_p 是 tuple（BiLSTM with features）：
            v_p: (stacked_p_idx, stacked_p_feat, stacked_p_len)
        y: stacked labels
    """
    v_d_list, v_p_list, y_list = zip(*batch)

    # 處理藥物數據
    if isinstance(v_d_list[0], tuple):
        # SELFIES 模式
        if len(v_d_list[0]) == 3:
            # 有物理特徵：v_d = (drug_idx, drug_feat, drug_len)
            drug_idx_list, drug_feat_list, drug_len_list = zip(*v_d_list)
            v_d = (torch.stack(drug_idx_list), torch.stack(drug_feat_list), torch.stack(drug_len_list))
        else:
            # 無物理特徵：v_d = (drug_idx, drug_len)
            drug_idx_list, drug_len_list = zip(*v_d_list)
            v_d = (torch.stack(drug_idx_list), torch.stack(drug_len_list))
    else:
        # Graph 模式：使用 DGL 的 batch 功能
        import dgl
        v_d = dgl.batch(v_d_list)

    # 處理蛋白質數據
    if isinstance(v_p_list[0], tuple):
        # BiLSTM 模式（with features）：v_p = (p_idx, p_feat, p_len)
        p_idx_list, p_feat_list, p_len_list = zip(*v_p_list)
        v_p = (torch.stack(p_idx_list), torch.stack(p_feat_list), torch.stack(p_len_list))
    else:
        # CNN 模式或簡單序列
        v_p = torch.stack(v_p_list)

    # 標籤 (flatten if needed)
    y = torch.cat(y_list) if y_list[0].dim() > 0 else torch.stack(y_list)

    return v_d, v_p, y


class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, max_drug_nodes=290, max_protein_length=1200, use_features=False,
                 use_selfies=False, selfies_vocab=None, max_drug_length=200, use_drug_features=False):
        """
        Args:
            list_IDs: 數據索引列表
            df: 包含 SMILES, Protein, Y 的 DataFrame
            max_drug_nodes: 圖表示時的最大節點數
            max_protein_length: 蛋白質最大長度
            use_features: 是否使用蛋白質物理化學特徵
            use_selfies: 是否使用 SELFIES（序列表示）而非 Graph
            selfies_vocab: SELFIES vocabulary（use_selfies=True 時必需）
            max_drug_length: SELFIES 最大長度
            use_drug_features: 是否使用藥物物理化學特徵（僅在 use_selfies=True 時生效）
        """
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.max_protein_length = max_protein_length
        self.use_features = use_features  # 是否使用氨基酸物理特徵
        self.use_selfies = use_selfies  # 是否使用 SELFIES（序列）表示
        self.selfies_vocab = selfies_vocab
        self.max_drug_length = max_drug_length
        self.use_drug_features = use_drug_features  # 是否使用藥物物理特徵

        if not use_selfies:
            # Graph-based: 需要 featurizers
            self.atom_featurizer = CanonicalAtomFeaturizer()
            self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
            self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        else:
            # Sequence-based: 需要 vocabulary
            if selfies_vocab is None:
                raise ValueError("selfies_vocab is required when use_selfies=True")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smiles = self.df.iloc[index]['SMILES']

        # 藥物表示：Graph 或 SELFIES 序列
        if self.use_selfies:
            # SELFIES 序列表示
            if self.use_drug_features:
                # 使用帶物理特徵的編碼
                v_d_idx, v_d_feat, v_d_len = encode_selfies_with_features(smiles, self.selfies_vocab, self.max_drug_length)
                v_d = (torch.LongTensor(v_d_idx), torch.FloatTensor(v_d_feat), torch.LongTensor([v_d_len]))
            else:
                # 使用原始編碼（不帶特徵）
                v_d_idx, v_d_len = encode_selfies(smiles, self.selfies_vocab, self.max_drug_length)
                v_d = (torch.LongTensor(v_d_idx), torch.LongTensor([v_d_len]))
        else:
            # Graph 表示
            v_d = self.fc(smiles=smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
            actual_node_feats = v_d.ndata.pop('h')
            num_actual_nodes = actual_node_feats.shape[0]
            num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
            v_d.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
            v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
            v_d = v_d.add_self_loop()

        # 蛋白質表示
        v_p = self.df.iloc[index]['Protein']

        if self.use_features:
            # 使用帶物理特徵的編碼
            v_p_idx, v_p_feat, v_p_len = encode_sequence_with_features(v_p, max_length=self.max_protein_length)
            y = self.df.iloc[index]["Y"]
            return v_d, (torch.LongTensor(v_p_idx), torch.FloatTensor(v_p_feat), torch.LongTensor([v_p_len])), torch.FloatTensor([y])
        else:
            # 使用原始編碼（不帶特徵）
            v_p = integer_label_protein(v_p, max_length=self.max_protein_length)
            y = self.df.iloc[index]["Y"]
            return v_d, torch.LongTensor(v_p), torch.FloatTensor([y])


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches


class SupportSetDTIDataset(data.Dataset):
    """
    Wrapper dataset that adds support set sampling to DTIDataset.
    Implements BatLiNet-style few-shot learning by providing K support samples for each query.
    """
    def __init__(self, base_dataset, support_pool_dataset, K=5, sampling_strategy="random"):
        """
        Args:
            base_dataset: Original DTIDataset for queries
            support_pool_dataset: DTIDataset to sample supports from (typically train set)
            K: Number of support samples per query
            sampling_strategy: "random" or other strategies
        """
        self.base_dataset = base_dataset
        self.support_pool = support_pool_dataset
        self.K = K
        self.sampling_strategy = sampling_strategy
        self.support_pool_size = len(support_pool_dataset.list_IDs)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Get query sample
        query_data = self.base_dataset[index]
        v_d_query, v_p_query, y_query = query_data

        # Sample K support indices
        if self.sampling_strategy == "random":
            support_indices = np.random.choice(
                self.support_pool_size, size=self.K, replace=False)
        else:
            raise NotImplementedError(f"Strategy {self.sampling_strategy} not implemented")

        # Get support samples
        support_data = [self.support_pool[idx] for idx in support_indices]
        support_v_d = [item[0] for item in support_data]
        support_v_p = [item[1] for item in support_data]
        support_y = [item[2] for item in support_data]

        return {
            'query': (v_d_query, v_p_query, y_query),
            'support': (support_v_d, support_v_p, support_y)
        }


class ProteinTaskDataset(data.Dataset):
    """
    Organizes data by protein ID, treating each protein as a separate task for MAML meta-learning.
    Supports support/query split for each task.

    Args:
        base_dataset: Original DTIDataset
        support_size: Number of support samples per task
        query_size: Number of query samples per task
        min_samples: Minimum number of samples for a protein to be considered a valid task
    """
    def __init__(self, base_dataset, support_size=16, query_size=16, min_samples=10):
        self.base_dataset = base_dataset
        self.support_size = support_size
        self.query_size = query_size
        self.min_samples = min_samples

        # Group data indices by protein ID
        self.protein_to_indices = self._group_by_protein()

        # Filter out proteins with insufficient samples
        self.valid_proteins = [
            prot for prot, indices in self.protein_to_indices.items()
            if len(indices) >= min_samples
        ]
        self.protein_list = list(self.valid_proteins)

        print(f"ProteinTaskDataset: {len(self.protein_list)} valid proteins (from {len(self.protein_to_indices)} total)")

    def _group_by_protein(self):
        """Group data indices by protein sequence"""
        protein_groups = {}
        df = self.base_dataset.df

        for idx in self.base_dataset.list_IDs:
            protein_seq = df.iloc[idx]['Protein']
            if protein_seq not in protein_groups:
                protein_groups[protein_seq] = []
            protein_groups[protein_seq].append(idx)

        return protein_groups

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, task_idx):
        """
        Returns a single task (protein) with support and query sets.

        Returns:
            {
                'protein_id': str,
                'support': [(v_d, v_p, y), ...],  # support_size samples
                'query': [(v_d, v_p, y), ...],    # query_size samples
            }
        """
        protein_id = self.protein_list[task_idx]
        indices = self.protein_to_indices[protein_id]

        # Shuffle and split into support/query
        shuffled_indices = np.random.permutation(indices)

        # Handle case where protein has fewer samples than support_size + query_size
        total_needed = self.support_size + self.query_size
        if len(shuffled_indices) < total_needed:
            # Sample with replacement to reach desired size
            shuffled_indices = np.random.choice(indices, size=total_needed, replace=True)

        support_indices = shuffled_indices[:self.support_size]
        query_indices = shuffled_indices[self.support_size:self.support_size + self.query_size]

        # Get actual samples from base dataset
        support_data = [self.base_dataset[idx] for idx in support_indices]
        query_data = [self.base_dataset[idx] for idx in query_indices]

        return {
            'protein_id': protein_id,
            'support': support_data,
            'query': query_data
        }


def collate_maml_fn(batch, use_selfies=False):
    """
    Collate function for MAML batches. Organizes multiple protein tasks into a single batch.

    Args:
        batch: List of dicts from DomainTaskDataset, each containing:
               {
                   'source_task': {'protein_id': str, 'support': [...], 'query': [...]},
                   'target_samples': [(v_d, v_p, y), ...]
               }
        use_selfies: Whether to use SELFIES (sequence) instead of graphs

    Returns:
        {
            'source_task': {
                'protein_ids': [str, ...],  # List of protein IDs
                'support': {
                    'v_d': tensor/graph batch,
                    'v_p': tensor,
                    'labels': tensor
                },
                'query': {
                    'v_d': tensor/graph batch,
                    'v_p': tensor,
                    'labels': tensor
                }
            },
            'target_samples': [(v_d, v_p, y), ...]  # Flattened target samples
        }
    """
    # Extract source tasks and target samples
    source_tasks = [item['source_task'] for item in batch]
    target_samples_nested = [item['target_samples'] for item in batch]

    # Extract protein IDs from source tasks
    protein_ids = [task['protein_id'] for task in source_tasks]

    # Flatten support and query data from all tasks
    support_data_flat = [sample for task in source_tasks for sample in task['support']]
    query_data_flat = [sample for task in source_tasks for sample in task['query']]

    # Flatten target samples from all tasks
    target_samples_flat = [sample for samples in target_samples_nested for sample in samples]

    # For MAML, we DON'T batch the drug data yet (to avoid unbatch/rebatch issues with DGL graphs)
    # Instead, we keep them as lists and batch them in _extract_task_data
    # We only collate protein data and labels here

    # Extract components from support and query data
    support_v_d_list, support_v_p_list, support_y_list = zip(*support_data_flat)
    query_v_d_list, query_v_p_list, query_y_list = zip(*query_data_flat)

    # Handle protein data (batch it now since it doesn't have the same issues as graphs)
    if isinstance(support_v_p_list[0], tuple):
        # BiLSTM mode: v_p is tuple (p_idx, p_feat, p_len)
        p_idx_list, p_feat_list, p_len_list = zip(*support_v_p_list)
        support_v_p = (torch.stack(p_idx_list), torch.stack(p_feat_list), torch.cat(p_len_list))

        p_idx_list, p_feat_list, p_len_list = zip(*query_v_p_list)
        query_v_p = (torch.stack(p_idx_list), torch.stack(p_feat_list), torch.cat(p_len_list))
    else:
        # CNN mode: v_p is single tensor
        support_v_p = torch.stack(support_v_p_list)
        query_v_p = torch.stack(query_v_p_list)

    # Handle labels
    support_y = torch.cat(support_y_list) if support_y_list[0].dim() > 0 else torch.stack(support_y_list)
    query_y = torch.cat(query_y_list) if query_y_list[0].dim() > 0 else torch.stack(query_y_list)

    return {
        'source_task': {
            'protein_ids': protein_ids,
            'support': {
                'v_d': list(support_v_d_list),  # Keep as list - will be batched in _extract_task_data
                'v_p': support_v_p,
                'labels': support_y
            },
            'query': {
                'v_d': list(query_v_d_list),  # Keep as list - will be batched in _extract_task_data
                'v_p': query_v_p,
                'labels': query_y
            }
        },
        'target_samples': target_samples_flat
    }


class DomainTaskDataset(data.Dataset):
    """
    Combines source domain tasks with target domain samples for MAML + MMD.
    Each item contains: source task (support/query) + target samples (for MMD).

    Args:
        source_task_dataset: ProteinTaskDataset for source domain
        target_dataset: DTIDataset for target domain
        target_sample_size: Number of target samples to include per task
    """
    def __init__(self, source_task_dataset, target_dataset, target_sample_size=16):
        self.source_tasks = source_task_dataset
        self.target_dataset = target_dataset
        self.target_sample_size = target_sample_size

    def __len__(self):
        return len(self.source_tasks)

    def __getitem__(self, idx):
        """
        Returns:
            {
                'source_task': {...},  # From ProteinTaskDataset
                'target_samples': [(v_d, v_p, y), ...]  # Target domain samples for MMD
            }
        """
        source_task = self.source_tasks[idx]

        # Randomly sample from target domain
        target_indices = np.random.choice(
            len(self.target_dataset),
            size=self.target_sample_size,
            replace=False if len(self.target_dataset) >= self.target_sample_size else True
        )
        target_samples = [self.target_dataset[i] for i in target_indices]

        return {
            'source_task': source_task,
            'target_samples': target_samples
        }
