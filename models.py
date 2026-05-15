import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]

        # 藥物編碼器：選擇 GCN 或 BiLSTM
        use_drug_bilstm = config["DRUG"].get("USE_BILSTM", False)  # 新增選項

        if use_drug_bilstm:
            # 使用 BiLSTM 編碼器（用於 SELFIES）
            drug_vocab_size = config["DRUG"].get("VOCAB_SIZE", 100)  # SELFIES vocabulary 大小
            drug_lstm_hidden = config["DRUG"].get("LSTM_HIDDEN_DIM", 64)
            drug_lstm_layers = config["DRUG"].get("LSTM_NUM_LAYERS", 2)
            drug_lstm_dropout = config["DRUG"].get("LSTM_DROPOUT", 0.2)
            max_drug_length = config["DRUG"].get("MAX_DRUG_LENGTH", 200)
            use_drug_features = config["DRUG"].get("USE_FEATURES", False)  # 是否使用藥物物理特徵
            drug_feature_dim = config["DRUG"].get("FEATURE_DIM", 8)  # 物理特徵維度

            self.drug_extractor = DrugBiLSTM(
                vocab_size=drug_vocab_size,
                embedding_dim=drug_embedding,
                hidden_dim=drug_lstm_hidden,
                num_layers=drug_lstm_layers,
                dropout=drug_lstm_dropout,
                padding=drug_padding,
                max_length=max_drug_length,
                output_dim=drug_hidden_feats[-1],  # 與 GCN 輸出維度一致
                use_features=use_drug_features,
                feature_dim=drug_feature_dim
            )
            drug_output_dim = drug_hidden_feats[-1]
            self.use_drug_sequence = True  # 使用序列形式（SELFIES）
        else:
            # 使用 GCN 編碼器（Graph-based，用於 SMILES）
            self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                               padding=drug_padding,
                                               hidden_feats=drug_hidden_feats)
            drug_output_dim = drug_hidden_feats[-1]
            self.use_drug_sequence = False

        # 蛋白質編碼器：選擇 CNN 或 BiLSTM
        use_protein_bilstm = config["PROTEIN"].get("USE_BILSTM", False)  # 新增選項

        if use_protein_bilstm:
            # 使用 BiLSTM 編碼器（專家版本）
            lstm_hidden_dim = config["PROTEIN"].get("LSTM_HIDDEN_DIM", 256)
            lstm_num_layers = config["PROTEIN"].get("LSTM_NUM_LAYERS", 2)
            lstm_dropout = config["PROTEIN"].get("LSTM_DROPOUT", 0.2)
            feature_dim = config["PROTEIN"].get("FEATURE_DIM", 4)

            max_protein_length = config["PROTEIN"].get("MAX_PROTEIN_LENGTH", 1200)
            self.protein_extractor = ProteinBiLSTM(
                embedding_dim=protein_emb_dim,
                hidden_dim=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                dropout=lstm_dropout,
                padding=protein_padding,
                feature_dim=feature_dim,
                max_length=max_protein_length
            )
            protein_output_dim = lstm_hidden_dim * 2  # BiLSTM 雙向
            self.use_protein_features = True
        else:
            # 使用 CNN 編碼器（官方版本）
            self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
            protein_output_dim = num_filters[-1]
            self.use_protein_features = False

        # Support Set 配置（BatLiNet-inspired）
        self.use_support_set = config.get("SUPPORT_SET", {}).get("USE", False)
        if self.use_support_set:
            self.support_alpha = config["SUPPORT_SET"]["ALPHA"]
            self.support_aggregation_train = config["SUPPORT_SET"]["AGGREGATION_TRAIN"]
            self.support_aggregation_test = config["SUPPORT_SET"]["AGGREGATION_TEST"]

        # BAN 注意力層
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_output_dim, q_dim=protein_output_dim, h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)

        # MLP 分類器
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, support_bg_d=None, support_v_p=None, support_labels=None, mode="train"):
        """
        Forward pass with optional support set (BatLiNet-style few-shot learning).

        Args:
            bg_d: drug data (graph batch for GCN, or (seq, len) tuple for BiLSTM)
            v_p: protein sequence/features (tensor or tuple)
            support_bg_d: support drug data [B*K] or None
            support_v_p: support protein data [B*K] or None
            support_labels: support labels [B, K] or None
            mode: "train" or "eval"

        Returns:
            In train mode: v_d, v_p_encoded, f, score, score_for_da
                           (score_for_da is the full 2-class score for domain adaptation)
            In eval mode: v_d, v_p_encoded, score, att
        """
        # Extract drug features
        if self.use_drug_sequence and isinstance(bg_d, tuple):
            # BiLSTM 模式：bg_d = (drug_idx, drug_feat, drug_len) 或 (drug_idx, drug_len)
            if len(bg_d) == 3:
                # 有物理特徵
                v_d = self.drug_extractor(bg_d[0], v_feat=bg_d[1], v_len=bg_d[2])  # [B, L, 128]
            else:
                # 無物理特徵
                v_d = self.drug_extractor(bg_d[0], v_len=bg_d[1])  # [B, L, 128]
        else:
            # GCN 模式：bg_d = graph
            v_d = self.drug_extractor(bg_d)  # [B, N, 128]

        # Extract protein features
        if self.use_protein_features and isinstance(v_p, tuple):
            # BiLSTM 模式：v_p = (protein_idx, protein_feat, protein_len)
            v_p_encoded = self.protein_extractor(v_p[0], v_p[1], v_p[2])
        else:
            # CNN 模式或沒有特徵
            v_p_encoded = self.protein_extractor(v_p)

        # Process support set if provided
        support_v_d = None
        support_v_p_encoded = None
        if support_bg_d is not None and support_v_p is not None:
            # Extract support drug features
            if self.use_drug_sequence and isinstance(support_bg_d, tuple):
                # BiLSTM 模式：support_bg_d = (drug_idx[B*K], drug_feat[B*K], drug_len[B*K]) 或 (drug_idx[B*K], drug_len[B*K])
                if len(support_bg_d) == 3:
                    # 有物理特徵
                    support_v_d_flat = self.drug_extractor(support_bg_d[0], v_feat=support_bg_d[1], v_len=support_bg_d[2])  # [B*K, L, 128]
                else:
                    # 無物理特徵
                    support_v_d_flat = self.drug_extractor(support_bg_d[0], v_len=support_bg_d[1])  # [B*K, L, 128]
            else:
                # GCN 模式
                support_v_d_flat = self.drug_extractor(support_bg_d)  # [B*K, N, 128]

            B = v_d.size(0)
            K = support_v_d_flat.size(0) // B
            N = support_v_d_flat.size(1)  # sequence length or num_nodes
            support_v_d = support_v_d_flat.view(B, K, N, -1)  # [B, K, N, 128]

            # Extract support protein features
            if self.use_protein_features and isinstance(support_v_p, tuple):
                # support_v_p = (support_p_idx[B*K], support_p_feat[B*K], support_p_len[B*K])
                support_v_p_encoded_flat = self.protein_extractor(
                    support_v_p[0], support_v_p[1], support_v_p[2])  # [B*K, L, 512]
            else:
                support_v_p_encoded_flat = self.protein_extractor(support_v_p)

            L = support_v_p_encoded_flat.size(1)
            support_v_p_encoded = support_v_p_encoded_flat.view(B, K, L, -1)  # [B, K, L, 512]

        # BAN layer with optional support set
        score_for_da = None  # Store original 2-class score for domain adaptation
        if self.use_support_set and support_v_d is not None:
            # Support set enabled: dual-path prediction
            f, att, support_f = self.bcn(v_d, v_p_encoded, support_v_d, support_v_p_encoded)

            # Original path prediction
            score_ori_full = self.mlp_classifier(f)  # [B, binary]
            score_for_da = score_ori_full  # Keep full score for DA

            # For binary classification (binary=2), extract class 1 logits; for binary=1, use as is
            if score_ori_full.size(1) == 2:
                score_ori = score_ori_full[:, 1:2]  # [B, 1] - extract positive class
            else:
                score_ori = score_ori_full

            # Support path predictions
            B, K, h_dim = support_f.size()
            support_f_flat = support_f.view(B * K, h_dim)
            score_sup_flat = self.mlp_classifier(support_f_flat)  # [B*K, binary]
            # For binary classification (binary=2), extract class 1 logits
            if score_sup_flat.size(1) == 2:
                score_sup_flat = score_sup_flat[:, 1:2]  # [B*K, 1] - extract positive class
            score_sup = score_sup_flat.view(B, K)  # [B, K]

            # Add support labels (relative prediction)
            if support_labels is not None:
                score_sup = score_sup + support_labels  # [B, K]

            # Aggregate support predictions
            training = (mode == "train")
            if training:
                score_sup_agg = score_sup.mean(dim=1, keepdim=True)  # [B, 1]
            else:
                score_sup_agg = score_sup.median(dim=1, keepdim=True)[0]  # [B, 1]

            # Final ensemble
            score = (1 - self.support_alpha) * score_ori + self.support_alpha * score_sup_agg
        else:
            # Standard path (no support set)
            f, att, _ = self.bcn(v_d, v_p_encoded)
            score = self.mlp_classifier(f)
            score_for_da = score  # Use same score for DA when no support set

        if mode == "train":
            return v_d, v_p_encoded, f, score, score_for_da
        elif mode == "eval":
            return v_d, v_p_encoded, score, att


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class ProteinBiLSTM(nn.Module):
    """
    BiLSTM 蛋白質編碼器（來自專家版本）
    使用雙向 LSTM 捕獲序列的雙向依賴關係
    支持物理化學特徵輸入（疏水性、體積、電荷、極性）
    """
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, dropout=0.2, padding=True, feature_dim=4, max_length=1200):
        super(ProteinBiLSTM, self).__init__()
        # 氨基酸 embedding（26 = 20種氨基酸 + 特殊字符）
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)

        self.feature_dim = feature_dim
        self.max_length = max_length  # 最大序列長度，用於固定 padding

        # BiLSTM（輸入 = embedding + 物理特徵）
        self.lstm = nn.LSTM(
            input_size=embedding_dim + feature_dim,  # 128 + 4 = 132
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        # 輸出維度（雙向所以 *2）
        self.output_dim = hidden_dim * 2

    def forward(self, v, v_feat=None, v_len=None):
        """
        Args:
            v: 蛋白質序列 tensor, shape: [batch_size, seq_len]
            v_feat: 物理化學特徵 tensor, shape: [batch_size, seq_len, 4] (可選)
            v_len: 實際序列長度 tensor, shape: [batch_size] (可選)

        Returns:
            outputs: shape: [batch_size, seq_len, hidden_dim*2]
        """
        # Embedding
        emb = self.embedding(v.long())  # [batch, seq_len, emb_dim]

        # 如果提供了物理特徵，串接到 embedding
        if v_feat is not None:
            combined = torch.cat([emb, v_feat.float()], dim=-1)  # [batch, seq_len, emb_dim+4]
        else:
            # 沒有特徵時填充 0
            batch_size, seq_len, _ = emb.shape
            zero_feat = torch.zeros(batch_size, seq_len, self.feature_dim, device=emb.device)
            combined = torch.cat([emb, zero_feat], dim=-1)

        # 計算每個序列的實際長度（非 padding 部分）
        if v_len is not None:
            lengths = v_len.view(-1).cpu().long()  # 確保是 1D tensor [batch] 且為 int64
        else:
            mask = (v != 0).long()
            lengths = mask.sum(dim=1).cpu().long()  # [batch]

        # Pack sequence for efficient processing
        packed = pack_padded_sequence(
            combined, lengths, batch_first=True, enforce_sorted=False
        )

        # BiLSTM
        packed_outputs, (hidden, cell) = self.lstm(packed)

        # Unpack with fixed total_length to ensure consistent dimensions
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=self.max_length)

        # 調整輸出格式以匹配 BAN 的期望格式
        # BAN 期望 [batch, seq_len, channels]
        return outputs


class DrugBiLSTM(nn.Module):
    """
    BiLSTM 藥物編碼器（用於 SELFIES 序列）
    將 SELFIES token 序列轉換為特徵表示
    輸出維度設計為與 MolecularGCN 相同以便比較
    支援加入藥物物理化學特徵（MW, logP, TPSA等）
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, dropout=0.2,
                 padding=True, max_length=200, output_dim=128, use_features=False, feature_dim=8):
        """
        Args:
            vocab_size: SELFIES vocabulary 大小
            embedding_dim: Token embedding 維度
            hidden_dim: LSTM 隱藏層維度
            num_layers: LSTM 層數
            dropout: Dropout 比率
            padding: 是否使用 padding
            max_length: 最大序列長度
            output_dim: 輸出維度（與 GCN 相同，默認 128）
            use_features: 是否使用藥物物理化學特徵
            feature_dim: 物理特徵維度（默認 8：MW, logP, TPSA, HBD, HBA, Ring, RotBonds, Charge）
        """
        super(DrugBiLSTM, self).__init__()

        self.use_features = use_features
        self.feature_dim = feature_dim

        # Token embedding（類似蛋白質的氨基酸 embedding）
        if padding:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.max_length = max_length

        # BiLSTM（雙向）
        # 如果使用特徵，input_size = embedding_dim + feature_dim
        lstm_input_size = embedding_dim + feature_dim if use_features else embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        # 輸出投影層：將 BiLSTM 輸出映射到目標維度
        # BiLSTM 輸出是 hidden_dim * 2（因為雙向）
        self.output_projection = nn.Linear(hidden_dim * 2, output_dim)
        self.output_dim = output_dim

    def forward(self, v, v_feat=None, v_len=None):
        """
        Args:
            v: SELFIES token 序列, shape: [batch_size, seq_len]
            v_feat: 藥物物理特徵, shape: [batch_size, feature_dim] (可選)
            v_len: 實際序列長度, shape: [batch_size] (可選)

        Returns:
            outputs: shape: [batch_size, seq_len, output_dim]
        """
        # Embedding
        emb = self.embedding(v.long())  # [batch, seq_len, emb_dim]

        # 如果使用物理特徵，將特徵擴展到每個序列位置並串接
        if self.use_features:
            if v_feat is not None:
                # 將全局特徵 [batch, 8] 擴展到每個位置 [batch, seq_len, 8]
                batch_size, seq_len, _ = emb.shape
                v_feat_expanded = v_feat.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, 8]
                combined = torch.cat([emb, v_feat_expanded], dim=-1)  # [batch, seq_len, emb_dim+8]
            else:
                # 沒有提供特徵時使用零填充
                batch_size, seq_len, _ = emb.shape
                zero_feat = torch.zeros(batch_size, seq_len, self.feature_dim, device=emb.device)
                combined = torch.cat([emb, zero_feat], dim=-1)
        else:
            combined = emb

        # 計算每個序列的實際長度
        if v_len is not None:
            lengths = v_len.view(-1).cpu().long()  # 確保是 1D tensor [batch] 且為 int64
        else:
            mask = (v != 0).long()
            lengths = mask.sum(dim=1).cpu().long()  # [batch]

        # Pack sequence for efficient processing
        packed = pack_padded_sequence(
            combined, lengths, batch_first=True, enforce_sorted=False
        )

        # BiLSTM
        packed_outputs, (hidden, cell) = self.lstm(packed)

        # Unpack with fixed total_length
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=self.max_length
        )  # [batch, seq_len, hidden_dim*2]

        # 投影到目標維度
        outputs = self.output_projection(outputs)  # [batch, seq_len, output_dim]

        # 輸出格式：[batch, seq_len, output_dim]，與 MolecularGCN 的 [batch, num_nodes, 128] 格式一致
        return outputs


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
