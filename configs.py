from yacs.config import CfgNode as CN

_C = CN()

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75
_C.DRUG.PADDING = True
_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290

# BiLSTM 選項（用於 SELFIES 序列表示）
_C.DRUG.USE_BILSTM = False  # 設為 True 使用 BiLSTM（SELFIES）替代 GCN（Graph）
_C.DRUG.VOCAB_SIZE = 100  # SELFIES vocabulary 大小（實際會根據數據集自動計算）
_C.DRUG.LSTM_HIDDEN_DIM = 64  # BiLSTM 隱藏層維度
_C.DRUG.LSTM_NUM_LAYERS = 2  # LSTM 層數
_C.DRUG.LSTM_DROPOUT = 0.2  # Dropout 率
_C.DRUG.MAX_DRUG_LENGTH = 200  # 最大 SELFIES 序列長度
_C.DRUG.USE_FEATURES = False  # 設為 True 使用藥物物理化學特徵（僅在 USE_BILSTM=True 時生效）
_C.DRUG.FEATURE_DIM = 8  # 藥物物理特徵維度（MW, logP, TPSA, HBD, HBA, Ring, RotBonds, Charge）

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True
# BiLSTM 選項（設為 True 使用 BiLSTM 替代 CNN）
_C.PROTEIN.USE_BILSTM = False
_C.PROTEIN.LSTM_HIDDEN_DIM = 256
_C.PROTEIN.LSTM_NUM_LAYERS = 2
_C.PROTEIN.LSTM_DROPOUT = 0.2
_C.PROTEIN.FEATURE_DIM = 4  # 氨基酸物理化學特徵維度（疏水性、體積、電荷、極性）
_C.PROTEIN.MAX_PROTEIN_LENGTH = 1200  # 最大蛋白質序列長度（官方默認，覆蓋 BindingDB 89%）

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True

# Domain adaptation
_C.DA = CN()
_C.DA.TASK = False
_C.DA.METHOD = "CDAN"
_C.DA.USE = False
_C.DA.INIT_EPOCH = 10
_C.DA.LAMB_DA = 1
_C.DA.RANDOM_LAYER = False
_C.DA.ORIGINAL_RANDOM = False
_C.DA.RANDOM_DIM = None
_C.DA.USE_ENTROPY = True

# MAML Meta-Learning configuration
_C.MAML = CN()
_C.MAML.ENABLE = False              # Enable MAML meta-learning
_C.MAML.INNER_LR = 0.01             # Inner loop learning rate (α)
_C.MAML.INNER_STEPS = 10            # Number of inner adaptation steps (K)
_C.MAML.FIRST_ORDER = False         # Use first-order MAML (saves memory)
_C.MAML.TASK_BATCH_SIZE = 4         # Number of tasks per meta-batch
_C.MAML.SUPPORT_SIZE = 16           # Support set size per task
_C.MAML.QUERY_SIZE = 16             # Query set size per task
_C.MAML.MIN_SAMPLES_PER_PROTEIN = 10  # Minimum samples to be considered a task
_C.MAML.MMD_WEIGHT = 0.1            # Weight for MMD loss (λ)
_C.MAML.MMD_KERNEL_MUL = 2.0        # MMD kernel bandwidth multiplier
_C.MAML.MMD_KERNEL_NUM = 5          # Number of kernels for MMD

# Support Set configuration (BatLiNet-inspired few-shot learning)
_C.SUPPORT_SET = CN()
_C.SUPPORT_SET.USE = False  # Enable/disable support set mechanism
_C.SUPPORT_SET.K = 5  # Number of support samples per query
_C.SUPPORT_SET.ALPHA = 0.5  # Weight for support vs original path (0-1)
_C.SUPPORT_SET.SAMPLING_STRATEGY = "random"  # Support sampling strategy
_C.SUPPORT_SET.AGGREGATION_TRAIN = "mean"  # Training aggregation method
_C.SUPPORT_SET.AGGREGATION_TEST = "median"  # Test aggregation method (more robust)

# Comet config, ignore it If not installed.
_C.COMET = CN()
# Please change to your own workspace name on comet.
_C.COMET.WORKSPACE = "pz-white"
_C.COMET.PROJECT_NAME = "DrugBAN"
_C.COMET.USE = False
_C.COMET.TAG = None


def get_cfg_defaults():
    return _C.clone()
