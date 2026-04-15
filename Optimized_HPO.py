# Imports
import json
import os
import shutil
import sys
from collections import defaultdict
from pprint import PrettyPrinter

import optuna
import torch
import torch.multiprocessing as mp
import transformers
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score

from TCBC_tools import Structure, MachineLearning as ml

# Constants
MODEL_NAME       = "TurkuNLP/bert-base-finnish-cased-v1"
HPO_RESULTS_FILE = "best_hyperparameters.json"
OPTUNA_DB        = "sqlite:///optuna_hpo.db"
STUDY_NAME       = "hpo_study"
N_TOTAL_TRIALS   = 30

pprint = PrettyPrinter(compact=True).pprint
os.environ["WANDB_MODE"] = "disabled"



#  Helper functions

def conllu2RawText(conllu_text):
    """Convert CoNLL-U formatted text to plain whitespace-joined tokens."""
    return " ".join(
        line.split("\t")[1]
        for line in conllu_text.split("\n")
        if len(line) > 0
    )


def assignLabel(book_id):
    """Map a book_id to a 3-class age-group label."""
    age = int(Structure.findAgeFromID(book_id))
    if age < 9:
        return 1
    elif age < 13:
        return 2
    return 0


def map_labels_and_text(batch):
    """Batched map: assign labels and convert CoNLL-U → raw text in one pass."""
    return {
        "label": [assignLabel(bid) for bid in batch["book_id"]],
        "data":  [conllu2RawText(txt) for txt in batch["data"]],
    }



#  Dataset preparation (called once, result cached to disk)

def prepare_dataset(split_id: int, cache_dir: str) -> DatasetDict:
    """Build, tokenize, and cache the dataset.  Idempotent — returns
    instantly if the cache already exists."""
    done_flag = cache_dir + ".done"
    if os.path.exists(done_flag):
        print(f"Dataset cache found at {cache_dir}")
        return DatasetDict.load_from_disk(cache_dir)

    num_workers = max(1, len(os.sched_getaffinity(0)))
    print(f"Preparing dataset with {num_workers} CPU workers …")

    keylists  = ml.getKeylist("Keylists.jsonl")
    train_set = set(keylists[split_id]["train_keys"])
    eval_set  = set(keylists[split_id]["eval_keys"])
    test_set  = set(keylists[split_id]["test_keys"])

    base = Dataset.load_from_disk("TCBC_datasets/sniplen5")

    def make_split(key_set):
        return (
            base
            .filter(lambda x: x["book_id"] in key_set,
                    num_proc=num_workers)
            .shuffle()
            .map(map_labels_and_text, batched=True,
                 num_proc=num_workers)
            .class_encode_column("label")
        )

    dataset = DatasetDict({
        "train": make_split(train_set),
        "eval":  make_split(eval_set),
        "test":  make_split(test_set),
    })

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["data"],
            max_length=512,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

    dataset = (
        dataset
        .map(tokenize, batched=True, batch_size=1024,
             num_proc=num_workers)
        .select_columns(["input_ids", "attention_mask", "label"])
    )

    dataset.save_to_disk(cache_dir)
    open(done_flag, "w").close()
    print(f"Dataset prepared and saved to {cache_dir}")
    return dataset



#  Shared metric function

def compute_metrics(pred):
    """Macro-averaged F1 (stored under the key 'accuracy' for
    backward compatibility with the rest of the pipeline)."""
    y_pred = pred.predictions.argmax(axis=-1)
    return {"accuracy": f1_score(pred.label_ids, y_pred, average="macro")}



#  PHASE 1 — Hyperparameter Optimisation (parallel single-GPU workers)

def _hpo_worker(gpu_id: int, cache_dir: str):
    """One Optuna worker pinned to *gpu_id*.  Launched via mp.Process."""

    # Pin BEFORE any CUDA call
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    dataset   = DatasetDict.load_from_disk(cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    collator  = transformers.DataCollatorWithPadding(tokenizer)

    # Optuna pruning callback
    class _PruneCallback(transformers.TrainerCallback):
        def __init__(self, trial):
            self.trial = trial

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if metrics is None:
                return
            acc = metrics.get("eval_accuracy")
            if acc is not None:
                self.trial.report(acc, step=state.global_step)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

    #Objective function
    def objective(trial: optuna.Trial) -> float:
        lr   = trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True)
        bs   = trial.suggest_categorical(
                   "per_device_train_batch_size", [8, 16, 32])
        wd   = trial.suggest_float("weight_decay", 0.0, 0.15)
        warm = trial.suggest_float("warmup_ratio", 0.0, 0.2)

        model = (
            transformers.AutoModelForSequenceClassification
            .from_pretrained(MODEL_NAME, num_labels=3)
        )

        out_dir = f"_hpo_ckpt_gpu{gpu_id}_t{trial.number}"

        args = transformers.TrainingArguments(
            output_dir                  = out_dir,
            eval_strategy               = "steps",
            logging_strategy            = "steps",
            save_total_limit            = 1,
            eval_steps                  = 100,
            logging_steps               = 100,
            load_best_model_at_end      = True,
            remove_unused_columns       = True,
            max_steps                   = 500,
            per_device_train_batch_size = bs,
            per_device_eval_batch_size  = 64,
            learning_rate               = lr,
            weight_decay                = wd,
            warmup_ratio                = warm,
            fp16                        = True,
            dataloader_num_workers      = 4,
            dataloader_pin_memory       = True,
            report_to                   = "none",
        )

        trainer = transformers.Trainer(
            model            = model,
            args             = args,
            train_dataset    = dataset["train"],
            eval_dataset     = dataset["eval"],
            processing_class = tokenizer,
            data_collator    = collator,
            compute_metrics  = compute_metrics,
            callbacks=[
                transformers.EarlyStoppingCallback(
                    early_stopping_patience=5),
                _PruneCallback(trial),
            ],
        )

        trainer.train()
        result = trainer.evaluate()

        shutil.rmtree(out_dir, ignore_errors=True)
        return result["eval_accuracy"]

    # Then in each worker load study
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=OPTUNA_DB,
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=100,
        ),
    )

    def _stop_when_done(study, trial):
        done = sum(
            1 for t in study.trials
            if t.state in (optuna.trial.TrialState.COMPLETE,
                           optuna.trial.TrialState.PRUNED)
        )
        if done >= N_TOTAL_TRIALS:
            study.stop()

    study.optimize(objective, n_trials=N_TOTAL_TRIALS,
                   callbacks=[_stop_when_done])
    print(f"[GPU {gpu_id}] HPO worker finished.")


def run_hpo(split_id: int) -> dict:
    """Prepare data, spawn parallel workers, return best parameters."""
    n_gpus    = torch.cuda.device_count()
    cache_dir = f"_tmp_processed_split_{split_id}"

    # 1.  Prepare data in the main process (no CUDA yet)
    prepare_dataset(split_id, cache_dir)

    # 2.  Remove stale study so we start fresh
    db_path = OPTUNA_DB.replace("sqlite:///", "")
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create shared study
    study = optuna.create_study(
        study_name     = STUDY_NAME,
        storage        = OPTUNA_DB,
        direction      = "maximize",
        load_if_exists = True,
        pruner         = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=100,
        ),
    )

    # 3.  One worker per GPU
    print(f"Launching {n_gpus} parallel HPO workers …")
    procs = []
    for gid in range(n_gpus):
        p = mp.Process(target=_hpo_worker, args=(gid, cache_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # 4.  Read the best result from the shared study
    study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_DB)
    best  = study.best_trial.params

    with open(HPO_RESULTS_FILE, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best hyperparameters saved to {HPO_RESULTS_FILE}")
    pprint(best)
    return best

def main():
    if len(sys.argv) < 2:
        print("Usage: srun python3 train_finbert.py <split_id>")
        sys.exit(1)

    split_id = int(sys.argv[1].replace(",", ""))
    n_gpus   = torch.cuda.device_count()

    # ── HPO ─────────────────────────────────────────────
    if os.path.exists(HPO_RESULTS_FILE):
        print(f"Found {HPO_RESULTS_FILE} — skipping HPO.")
        with open(HPO_RESULTS_FILE) as f:
            hp = json.load(f)
    else:
        print("=" * 60)
        print(f"PHASE 1 — Hyperparameter search  "
              f"({n_gpus} GPUs × {N_TOTAL_TRIALS} total trials)")
        print("=" * 60)
        study = optuna.create_study(
            study_name     = STUDY_NAME,
            storage        = OPTUNA_DB,
            direction      = "maximize",
            load_if_exists = True,
            pruner         = optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=100,
            ),
        )
        hp = run_hpo(split_id)

    print("Best hyperparameters:")
    pprint(hp)


if __name__ == "__main__":
    # "spawn" is required for safe CUDA use across child processes.
    # "fork" would duplicate the parent's CUDA context → crashes.
    mp.set_start_method("spawn", force=True)
    main()