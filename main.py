comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import DrugBAN
from time import time
from utils import set_seed, graph_collate_func, graph_collate_func_with_support, mkdir, build_selfies_vocab
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader, SupportSetDTIDataset, collate_selfies_fn
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    if not cfg.DA.TASK:
        train_path = os.path.join(dataFolder, 'train.csv')
        val_path = os.path.join(dataFolder, "val.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        use_features = cfg.PROTEIN.get("USE_BILSTM", False)  # BiLSTM 使用物理特徵
        use_drug_features = cfg.DRUG.get("USE_FEATURES", False)  # 藥物物理特徵

        # Build SELFIES vocabulary if using Drug BiLSTM
        use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
        if use_drug_bilstm:
            vocab_path = os.path.join(cfg.RESULT.OUTPUT_DIR, 'selfies_vocab.pkl')
            if os.path.exists(vocab_path):
                # Load existing vocabulary
                print(f"Loading SELFIES vocabulary from {vocab_path}")
                with open(vocab_path, 'rb') as f:
                    selfies_vocab = pickle.load(f)
            else:
                # Build vocabulary from training data
                print("Building SELFIES vocabulary from training data...")
                all_smiles = df_train['SMILES'].tolist()
                selfies_vocab = build_selfies_vocab(all_smiles, max_vocab_size=500)
                cfg.DRUG.VOCAB_SIZE = len(selfies_vocab) + 1  # +1 for padding

                # Save vocabulary
                mkdir(cfg.RESULT.OUTPUT_DIR)
                with open(vocab_path, 'wb') as f:
                    pickle.dump(selfies_vocab, f)
                print(f"SELFIES vocabulary size: {cfg.DRUG.VOCAB_SIZE}")
                print(f"Vocabulary saved to {vocab_path}")
        else:
            selfies_vocab = None

        train_dataset = DTIDataset(
            df_train.index.values, df_train,
            max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
            use_features=use_features,
            use_selfies=use_drug_bilstm,
            selfies_vocab=selfies_vocab,
            max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
            use_drug_features=use_drug_features
        )
        val_dataset = DTIDataset(
            df_val.index.values, df_val,
            max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
            use_features=use_features,
            use_selfies=use_drug_bilstm,
            selfies_vocab=selfies_vocab,
            max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
            use_drug_features=use_drug_features
        )
        test_dataset = DTIDataset(
            df_test.index.values, df_test,
            max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
            use_features=use_features,
            use_selfies=use_drug_bilstm,
            selfies_vocab=selfies_vocab,
            max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
            use_drug_features=use_drug_features
        )
    else:
        train_source_path = os.path.join(dataFolder, 'source_train.csv')
        train_target_path = os.path.join(dataFolder, 'target_train.csv')
        test_target_path = os.path.join(dataFolder, 'target_test.csv')
        df_train_source = pd.read_csv(train_source_path)
        df_train_target = pd.read_csv(train_target_path)
        df_test_target = pd.read_csv(test_target_path)

        use_features = cfg.PROTEIN.get("USE_BILSTM", False)  # BiLSTM 使用物理特徵
        use_drug_features = cfg.DRUG.get("USE_FEATURES", False)  # 藥物物理特徵

        # Build SELFIES vocabulary if using Drug BiLSTM (DA mode)
        use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
        if use_drug_bilstm:
            vocab_path = os.path.join(cfg.RESULT.OUTPUT_DIR, 'selfies_vocab.pkl')
            if os.path.exists(vocab_path):
                print(f"Loading SELFIES vocabulary from {vocab_path}")
                with open(vocab_path, 'rb') as f:
                    selfies_vocab = pickle.load(f)
            else:
                print("Building SELFIES vocabulary from source training data...")
                all_smiles = df_train_source['SMILES'].tolist()
                selfies_vocab = build_selfies_vocab(all_smiles, max_vocab_size=500)
                cfg.DRUG.VOCAB_SIZE = len(selfies_vocab) + 1

                mkdir(cfg.RESULT.OUTPUT_DIR)
                with open(vocab_path, 'wb') as f:
                    pickle.dump(selfies_vocab, f)
                print(f"SELFIES vocabulary size: {cfg.DRUG.VOCAB_SIZE}")
                print(f"Vocabulary saved to {vocab_path}")
        else:
            selfies_vocab = None

        train_dataset = DTIDataset(
            df_train_source.index.values, df_train_source,
            max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
            use_features=use_features,
            use_selfies=use_drug_bilstm,
            selfies_vocab=selfies_vocab,
            max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
            use_drug_features=use_drug_features
        )
        train_target_dataset = DTIDataset(
            df_train_target.index.values, df_train_target,
            max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
            use_features=use_features,
            use_selfies=use_drug_bilstm,
            selfies_vocab=selfies_vocab,
            max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
            use_drug_features=use_drug_features
        )
        test_target_dataset = DTIDataset(
            df_test_target.index.values, df_test_target,
            max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
            use_features=use_features,
            use_selfies=use_drug_bilstm,
            selfies_vocab=selfies_vocab,
            max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
            use_drug_features=use_drug_features
        )

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
            "DA_use": cfg.DA.USE,
            "DA_task": cfg.DA.TASK,
        }
        if cfg.DA.USE:
            da_hyper_params = {
                "DA_init_epoch": cfg.DA.INIT_EPOCH,
                "Use_DA_entropy": cfg.DA.USE_ENTROPY,
                "Random_layer": cfg.DA.RANDOM_LAYER,
                "Original_random": cfg.DA.ORIGINAL_RANDOM,
                "DA_optim_lr": cfg.SOLVER.DA_LR
            }
            hyper_params.update(da_hyper_params)
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{suffix}")

    # Check if using support sets
    use_support_set = cfg.get("SUPPORT_SET", {}).get("USE", False)

    if use_support_set:
        # Wrap datasets with support set sampling
        K = cfg.SUPPORT_SET.K
        sampling_strategy = cfg.SUPPORT_SET.SAMPLING_STRATEGY

        train_dataset_with_support = SupportSetDTIDataset(
            train_dataset, train_dataset,
            K=K, sampling_strategy=sampling_strategy
        )

        # Use support set collate function
        # Choose collate function based on drug representation
        use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
        collate_fn = collate_selfies_fn if use_drug_bilstm else graph_collate_func_with_support
        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': True, 'collate_fn': collate_fn}

        training_generator = DataLoader(train_dataset_with_support, **params)

        # Val/test also need support sets (sampled from training set)
        params['shuffle'] = False
        params['drop_last'] = False

        if not cfg.DA.TASK:
            val_dataset_with_support = SupportSetDTIDataset(
                val_dataset, train_dataset,
                K=K, sampling_strategy=sampling_strategy
            )
            test_dataset_with_support = SupportSetDTIDataset(
                test_dataset, train_dataset,
                K=K, sampling_strategy=sampling_strategy
            )
            val_generator = DataLoader(val_dataset_with_support, **params)
            test_generator = DataLoader(test_dataset_with_support, **params)
        else:
            # For domain adaptation with support sets
            train_target_with_support = SupportSetDTIDataset(
                train_target_dataset, train_dataset,
                K=K, sampling_strategy=sampling_strategy
            )
            test_target_with_support = SupportSetDTIDataset(
                test_target_dataset, train_dataset,
                K=K, sampling_strategy=sampling_strategy
            )

            # Create source and target generators
            source_generator = DataLoader(train_dataset_with_support, **params)
            target_generator = DataLoader(train_target_with_support, **params)
            n_batches = max(len(source_generator), len(target_generator))
            multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)

            val_generator = DataLoader(test_target_with_support, **params)
            test_generator = DataLoader(test_target_with_support, **params)
    else:
        # Original behavior (no support set)
        # Choose collate function based on drug representation
        use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
        collate_fn = collate_selfies_fn if use_drug_bilstm else graph_collate_func
        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': True, 'collate_fn': collate_fn}

        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        if not cfg.DA.TASK:
            val_generator = DataLoader(val_dataset, **params)
            test_generator = DataLoader(test_dataset, **params)
        else:
            # Check if using MAML meta-learning
            use_maml = (cfg.DA.METHOD == "MetaLearning" and cfg.MAML.ENABLE)

            if use_maml:
                print("=" * 70)
                print("Setting up MAML + MMD Meta-Learning...")
                print("=" * 70)

                from dataloader import ProteinTaskDataset, DomainTaskDataset, collate_maml_fn

                # Create source domain task dataset (organized by protein)
                source_task_dataset = ProteinTaskDataset(
                    train_dataset,
                    support_size=cfg.MAML.SUPPORT_SIZE,
                    query_size=cfg.MAML.QUERY_SIZE,
                    min_samples=cfg.MAML.MIN_SAMPLES_PER_PROTEIN
                )
                print(f"Created {len(source_task_dataset)} protein tasks from source domain")

                # Create domain task dataset (combines source tasks + target samples for MMD)
                domain_task_dataset = DomainTaskDataset(
                    source_task_dataset,
                    train_target_dataset,
                    target_sample_size=cfg.MAML.SUPPORT_SIZE
                )

                # MAML collate function
                use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
                maml_collate = lambda batch: collate_maml_fn(batch, use_selfies=use_drug_bilstm)

                params_maml = {
                    'batch_size': cfg.MAML.TASK_BATCH_SIZE,  # Number of tasks per batch
                    'shuffle': True,
                    'num_workers': cfg.SOLVER.NUM_WORKERS,
                    'drop_last': True,
                    'collate_fn': maml_collate
                }

                # MAML uses single dataloader (not MultiDataLoader)
                training_generator = DataLoader(domain_task_dataset, **params_maml)
                print(f"MAML DataLoader created: {len(training_generator)} batches, "
                      f"{cfg.MAML.TASK_BATCH_SIZE} tasks per batch")

                # Validation and test still use standard setup
                params['shuffle'] = False
                params['drop_last'] = False
                val_generator = DataLoader(test_target_dataset, **params)
                test_generator = DataLoader(test_target_dataset, **params)

                print("=" * 70)

            else:
                if cfg.DA.USE:
                    # Standard DA (CDAN, DANN, MMD) - use MultiDataLoader
                    source_generator = DataLoader(train_dataset, **params)
                    target_generator = DataLoader(train_target_dataset, **params)
                    n_batches = max(len(source_generator), len(target_generator))
                    multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
                    training_generator = multi_generator
                else:
                    # DA.TASK=True but DA.USE=False: only use source data for training
                    training_generator = DataLoader(train_dataset, **params)
                params['shuffle'] = False
                params['drop_last'] = False
                val_generator = DataLoader(test_target_dataset, **params)
                test_generator = DataLoader(test_target_dataset, **params)

    model = DrugBAN(**cfg).to(device)

    if cfg.DA.USE:
        # Determine discriminator input size based on DA method
        if cfg["DA"]["METHOD"] == "MMD" or (cfg["DA"]["METHOD"] == "MetaLearning" and cfg.MAML.ENABLE):
            # MMD and MAML don't use a discriminator
            domain_dmm = None
            opt_da = None
        else:
            # For CDAN, DANN, and other methods
            if cfg["DA"]["RANDOM_LAYER"]:
                disc_input_size = cfg["DA"]["RANDOM_DIM"]
            elif cfg["DA"]["METHOD"] == "CDAN":
                # CDAN concatenates features with softmax output
                disc_input_size = cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"]
            elif cfg["DA"]["METHOD"] == "DANN":
                # DANN uses raw features only
                disc_input_size = cfg["DECODER"]["IN_DIM"]
            else:
                # Default: use raw features
                disc_input_size = cfg["DECODER"]["IN_DIM"]

            domain_dmm = Discriminator(input_size=disc_input_size,
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)
            opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)

        # params = list(model.parameters()) + list(domain_dmm.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    if not cfg.DA.USE:
        trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                          discriminator=None,
                          experiment=experiment, **cfg)
    else:
        # Use training_generator (which is either multi_generator for standard DA or MAML dataloader)
        trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=opt_da,
                          discriminator=domain_dmm,
                          experiment=experiment, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
