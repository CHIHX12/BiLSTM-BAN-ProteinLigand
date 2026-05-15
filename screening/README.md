# HTS Drug-Protein Binding Screening System
## 高通量藥物-蛋白質結合預測系統

本系統基於 GCN-BiLSTM-BAN（DrugBAN 改進版）訓練模型，提供
N 個配體 × M 個蛋白質 的批次結合親和力預測，並輸出：
- 結合機率排行榜 (CSV)
- 熱圖矩陣 (CSV，可直接貼入 Excel)
- 預測結合位點殘基 (注意力權重)

---

## 目錄結構

```
screening/
├── input/
│   ├── ligands/
│   │   ├── ligands.txt            ← 【放您的配體 SMILES】
│   │   └── ligands_template.txt   ← 格式範例
│   └── proteins/
│       ├── proteins.txt           ← 【放您的蛋白質序列】
│       └── proteins_template.txt  ← 格式範例
├── output/                        ← 結果自動輸出至此
│   └── YYYYMMDD_HHMMSS/
│       ├── results_full.csv       ← 全部預測結果
│       ├── results_top50.csv      ← Top 50 排行
│       ├── heatmap.csv            ← 熱圖矩陣
│       ├── binding_sites.csv      ← 預測結合位點
│       ├── run_log.txt            ← 詳細執行紀錄
│       └── run_summary.txt        ← 執行摘要報告
├── preprocess/                    ← 輸入驗證模組
├── tests/                         ← 測試套件
│   ├── test_smiles_validator.py   ← SMILES 驗證器單元測試 (21 cases)
│   ├── test_protein_validator.py  ← 蛋白質驗證器單元測試 (26 cases)
│   ├── test_input_loader.py       ← 輸入載入器單元測試 (15 cases)
│   ├── test_model_benchmark.py    ← 模型 AUROC 基準測試 (AUROC ≥ 0.90)
│   └── test_biological_plausibility.py  ← 生物合理性驗證 (已知藥物-靶點對)
├── configs/
│   └── screening.yaml             ← 設定檔
├── validate_inputs.py             ← Step 1: 驗證輸入
├── screen.py                      ← Step 2: 執行篩選
├── run_tests.py                   ← 測試套件執行器
└── run.sh                         ← 一鍵執行腳本
```

---

## 快速開始（一鍵執行）

最簡單的方式：將配體和蛋白質放入 input 資料夾，然後執行：

```bash
cd screening/
chmod +x run.sh   # 第一次才需要
./run.sh          # 標準執行（BiLSTM 模型）
```

腳本會自動完成：環境檢查 → checkpoint 驗證 → 輸入驗證 → 篩選 → 結果預覽。

常用選項：
```bash
./run.sh --model cnn          # 使用 CNN 模型
./run.sh --top 100            # 輸出 Top 100
./run.sh --threshold 0.6      # 更嚴格的結合閾值
./run.sh --test               # 先跑測試，再執行篩選
./run.sh --test-only          # 只跑測試（不執行篩選）
./run.sh --fast-test          # 跳過慢速 benchmark，再執行篩選
./run.sh --help               # 顯示所有選項
```

---

## 手動執行步驟

### Step 1 — 準備輸入檔案

將您的配體 SMILES 放入 `input/ligands/ligands.txt`：
```
# 格式: 名稱 [Tab] SMILES
Aspirin	CC(=O)Oc1ccccc1C(=O)O
Ibuprofen	CC(C)Cc1ccc(cc1)C(C)C(=O)O
```

將您的蛋白質序列放入 `input/proteins/proteins.txt`（支援 FASTA 格式）：
```
# 格式: 名稱 [Tab] 胺基酸序列
EGFR_HUMAN	MRPSGTAGAALLALLAALC...
```

### Step 2 — 驗證輸入（建議先執行）

```bash
cd screening/
python validate_inputs.py
```

輸出範例：
```
[OK]      Aspirin     MW=180.2  atoms=13
[WARNING] Salt_Drug   Multi-fragment SMILES → kept largest fragment
[ERROR]   Invalid_X   RDKit cannot parse SMILES
```
→ 驗證通過後，會生成 `validation_report.txt`

### Step 3 — 執行篩選（或直接用 run.sh）

```bash
python screen.py
```

或加上參數：
```bash
# 使用 BiLSTM 模型 (預設), Top 100, 閾值 0.6
python screen.py --model bilstm --top 100 --threshold 0.6

# 使用 CNN 模型, 小 batch (低記憶體環境)
python screen.py --model cnn --batch-size 8

# 跳過結合位點提取 (加速大規模篩選)
python screen.py --no-binding-sites

# 已驗證過輸入，跳過驗證步驟
python screen.py --skip-validation
```

---

## 輸入格式詳細說明

### 配體檔案 (ligands.txt)

| 欄位 | 必填 | 說明 |
|------|------|------|
| 名稱 | 否   | 任意字串，不含 Tab。省略時自動命名 LIG_0001 |
| SMILES | 是  | 標準 SMILES 格式，支援 Canonical/Isomeric/Kekule |

**自動處理的問題：**
- 多片段 SMILES（鹽類，含 `.`）→ 自動保留最大片段
- Unicode/空白字元 → 自動移除
- 重複條目 → 警告，仍計算（結果相同）

**自動排除的條目：**
- RDKit 無法解析的 SMILES
- 重原子數 > 290（模型上限）

### 蛋白質檔案 (proteins.txt)

| 格式 | 範例 |
|------|------|
| TSV | `PROTEIN_NAME\tMKTAYIAKQR...` |
| 純序列 | `MKTAYIAKQR...` |
| FASTA | `>sp\|P00533\|EGFR_HUMAN ...\nMRPSGTAGAA...` |

**自動處理的問題：**
- FASTA 格式 → 自動剝離 `>` 標頭
- 多鏈 FASTA → 預設取第一條鏈（可用 `--multi-chain concat` 串接）
- 非標準胺基酸 (B, J, O, U, Z) → 替換為 X 並警告
- 小寫 → 轉大寫
- 空格、數字、特殊符號 → 自動移除
- 長度 > 1200 → 截斷並警告

---

## 輸出檔案說明

### results_full.csv

| 欄位 | 說明 |
|------|------|
| rank | 排名 (1=最高結合機率) |
| drug_name | 配體名稱 |
| protein_name | 蛋白質名稱 |
| binding_prob | 結合機率 [0,1] |
| bind_predict | BIND / NO_BIND (依閾值) |
| confidence | HIGH / MEDIUM / LOW / VERY_LOW |

### heatmap.csv

矩陣格式，行 = 配體，列 = 蛋白質，儲存格 = 結合機率。
可直接匯入 Excel → 插入 → 熱圖。

### binding_sites.csv

| 欄位 | 說明 |
|------|------|
| drug_name | 配體名稱 |
| protein_name | 蛋白質名稱 |
| binding_prob | 該對的結合機率 |
| residue_pos | 殘基位置 (1-indexed) |
| residue_aa | 胺基酸單字母碼 |
| attention_score | 注意力分數 [0,1] (已正規化) |
| is_binding_site | True/False (注意力分數 ≥ 80th percentile) |

**注意：** 結合位點資訊只對 binding_prob ≥ threshold 的配對輸出。

---

## 常見問題

### Q: GPU 記憶體不足 (CUDA OOM)
```bash
python screen.py --batch-size 8
```

### Q: 沒有 GPU，CPU 跑很慢
- CPU 速度約每秒 8–15 對 (依伺服器效能)
- 10,000 對 ≈ 15–20 分鐘
- 若時間不允許，先用 `--top 20 --no-binding-sites` 減少計算

### Q: FASTA 多鏈蛋白質 (抗體等)
```bash
python screen.py --multi-chain concat  # 串接所有鏈
python screen.py --multi-chain error   # 遇到多鏈時停止
```

### Q: 如何解讀結合位點？
- `attention_score` 反映 BiLSTM 模型對該殘基的「關注程度」
- `is_binding_site=True` 的殘基是模型預測的活性位點候選
- 建議與 UniProt 已知功能位點比對驗證
- 可用 PyMOL 視覺化（參見主專案 repro_comparison/）

---

## 測試套件

系統包含完整測試套件，確保模型正確載入且預測具生物合理性。

### 快速執行（跳過慢速 benchmark）

```bash
python run_tests.py --fast
# 或透過 run.sh：
./run.sh --fast-test
```

耗時約 5–10 秒，執行 67 個測試（unit tests + biological plausibility）。

### 完整執行（含 AUROC benchmark）

```bash
python run_tests.py
# 或透過 run.sh：
./run.sh --test-only
```

耗時約 2–5 分鐘，額外驗證 BiLSTM/CNN 在 200 樣本 BindingDB 子集的 AUROC ≥ 0.90。

### 測試模組說明

| 模組 | 測試數 | 涵蓋範圍 |
|------|--------|---------|
| test_smiles_validator.py | 21 | SMILES OK/WARNING/ERROR 各情境 |
| test_protein_validator.py | 26 | FASTA/TSV/多鏈/非標準胺基酸 |
| test_input_loader.py | 15 | 檔案載入、BOM、重複命名 |
| test_model_benchmark.py | 5 | AUROC ≥ 0.90，分數分布健康檢查 |
| test_biological_plausibility.py | 5 | 已知藥物-靶點對生物合理性 |

### 生物合理性驗證結果

測試使用文獻確認的藥物-靶點對，驗證模型預測方向正確：

| 藥物 | 靶點 | 模型分數 | 生物事實 |
|------|------|---------|---------|
| Tamoxifen | ESR1 (雌激素受體) | ~0.99 | BIND ✓ |
| Caffeine | A2AR (腺苷受體) | ~0.96 | BIND ✓ |
| Methotrexate | DHFR (二氫葉酸還原酶) | ~0.97 | BIND ✓ |
| Caffeine | HMGCR (膽固醇合成酶) | ~0.14 | NO BIND ✓ |
| Tamoxifen | A2AR (腺苷受體) | ~0.04 | NO BIND ✓ |

---

## 已知模型限制

以下是模型目前的已知假陰性，已在 `test_biological_plausibility.py` 中記錄：

| 藥物 | 靶點 | 模型分數 | 生物事實 | 原因 |
|------|------|---------|---------|------|
| Aspirin | COX2 (PTGS2) | ~0.002 | BIND（假陰性） | Aspirin 共價乙醯化機制在 BindingDB 中代表性不足 |
| Ibuprofen | COX2 (PTGS2) | ~0.021 | BIND（假陰性） | COX 抑制劑在 BindingDB 訓練資料中代表性不足 |
| Atorvastatin | HMGCR | ~0.018 | BIND（假陰性） | HMGCR 抑制劑在 BindingDB 訓練資料中代表性不足 |

**重要提示：** 對於 COX 抑制劑（NSAIDs）和 statin 類藥物，模型可能低估結合機率。
臨床決策請務必結合文獻資料與後續實驗驗證。

---

## 系統需求

```
Python == 3.10
torch == 2.2.1
dgl == 2.1.0+cu121
dgllife == 0.3.2
rdkit >= 2024.3
numpy >= 1.26
```

安裝（使用一鍵腳本，推薦）：
```bash
# Linux / Mac
bash setup.sh

# Windows（Anaconda Prompt）
setup.bat
```

手動安裝：
```bash
conda env create -f environment.yml
conda activate drugban
```

---

## AI Agent 整合說明

本系統設計為可被 LLM Agent 操控。
關鍵接口：

1. **輸入端**：修改 `input/ligands/ligands.txt` 和 `input/proteins/proteins.txt`
2. **執行端**：`python screen.py [參數]`
3. **輸出端**：讀取 `output/最新資料夾/results_full.csv`

典型 Agent 工作流：
```
1. 寫入配體/蛋白質到 input 資料夾
2. 呼叫 validate_inputs.py 確認格式
3. 呼叫 screen.py 執行預測
4. 讀取 results_top50.csv 取得排行結果
5. 讀取 binding_sites.csv 分析結合位點
6. 生成報告或建議下一步實驗
```

---

*系統版本: 1.1.0 | 模型: DrugBAN-BiLSTM (epoch 94) | 測試: 67 passed*
