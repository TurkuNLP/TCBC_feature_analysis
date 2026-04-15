import json
import multiprocessing as mp
import os
import shutil
import sys
from collections import defaultdict
from pprint import PrettyPrinter

pprint = PrettyPrinter(compact=True).pprint

MODEL_NAME = "TurkuNLP/bert-base-finnish-cased-v1"
HP_FILE    = "best_hyperparameters.json"
NUM_SPLITS = 20
GPU_IDS    = [0, 1, 2, 3]

os.environ["WANDB_MODE"] = "disabled"



# Helper functions

def conllu2RawText(conllu_text):
    return " ".join(
        line.split("\t")[1]
        for line in conllu_text.split("\n")
        if len(line) > 0
    )


def map_labels_and_text(batch):
    from TCBC_tools import Structure
    def _label(book_id):
        age = int(Structure.findAgeFromID(book_id))
        if age < 9:   return 1
        if age < 13:  return 2
        return 0
    return {
        "label": [_label(bid) for bid in batch["book_id"]],
        "data":  [conllu2RawText(txt) for txt in batch["data"]],
    }



# Dataset preparation

def prepare_dataset(split_id: int, cache_dir: str):
    import transformers
    from datasets import Dataset, DatasetDict
    from TCBC_tools import MachineLearning as ml

    done_flag = cache_dir + ".done"
    if os.path.exists(done_flag):
        print(f"  [Split {split_id}] Dataset cache found at {cache_dir}")
        return DatasetDict.load_from_disk(cache_dir)

    num_workers = max(1, min(len(os.sched_getaffinity(0)), 40))
    print(f"  [Split {split_id}] Preparing dataset "
          f"with {num_workers} CPU workers …")

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
    print(f"  [Split {split_id}] Dataset saved to {cache_dir}")
    return dataset


def compute_metrics(pred):
    from sklearn.metrics import f1_score
    y_pred = pred.predictions.argmax(axis=-1)
    return {"accuracy": f1_score(pred.label_ids, y_pred, average="macro")}



# Train a single split (called inside a GPU worker)

def _train_one_split(split_id: int, hp: dict, gpu_id: int):
    """Train, evaluate, save, plot for one split.  Returns (split_id, f1)."""
    import transformers

    cache_dir = f"_tmp_processed_split_{split_id}"
    dataset   = prepare_dataset(split_id, cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    collator  = transformers.DataCollatorWithPadding(tokenizer)

    class LogSavingCallback(transformers.TrainerCallback):
        def on_train_begin(self, *args, **kwargs):
            cfg = args[0].to_dict()
            print(f"  [Split {split_id}] "
                  f"lr={cfg['learning_rate']}  "
                  f"bs={cfg['per_device_train_batch_size']}  "
                  f"wd={cfg['weight_decay']}  "
                  f"warm={cfg['warmup_ratio']}")
            self.logs = defaultdict(list)
            self.training = True

        def on_train_end(self, *args, **kwargs):
            self.training = False

        def on_log(self, args, state, control, logs, model=None, **kw):
            if self.training:
                for k, v in logs.items():
                    if k != "epoch" or v not in self.logs[k]:
                        self.logs[k].append(v)

    training_logs = LogSavingCallback()

    model = (
        transformers.AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME, num_labels=3)
    )

    training_args = transformers.TrainingArguments(
        output_dir                  = f"checkpoints_{split_id}",
        eval_strategy               = "steps",
        logging_strategy            = "steps",
        save_total_limit            = 2,
        eval_steps                  = 500,
        logging_steps               = 500,
        load_best_model_at_end      = True,
        remove_unused_columns       = True,
        num_train_epochs            = 3,
        learning_rate               = hp["learning_rate"],
        per_device_train_batch_size = hp["per_device_train_batch_size"],
        weight_decay                = hp.get("weight_decay", 0.0),
        warmup_ratio                = hp.get("warmup_ratio", 0.0),
        per_device_eval_batch_size  = 64,
        fp16                        = True,
        dataloader_num_workers      = 2,
        dataloader_pin_memory       = True,
        report_to                   = "none",
    )

    trainer = transformers.Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = dataset["train"],
        eval_dataset     = dataset["eval"],
        processing_class = tokenizer,
        data_collator    = collator,
        compute_metrics  = compute_metrics,
        callbacks=[
            training_logs,
        ],
    )

    trainer.train()

    test_results = trainer.evaluate(dataset["test"])
    f1 = test_results["eval_accuracy"]
    print(f"[Split {split_id:>2d}] GPU {gpu_id} | "
          f"Test F1 (macro): {f1:.4f}")

    save_dir = f"FinBERT_for_book_snippets_5_new_split_{split_id}"
    trainer.save_model(save_dir)
    print(f"[Split {split_id:>2d}] Model saved to {save_dir}")

    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _plot(logs, keys, labels, filename):
            vals = sum([logs[k] for k in keys], [])
            plt.figure()
            plt.ylim(max(min(vals) - 0.1, 0.0),
                      min(max(vals) + 0.1, 1.0))
            for key, label in zip(keys, labels):
                plt.plot(logs["epoch"], logs[key], label=label)
            plt.legend()
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()

        _plot(training_logs.logs,
              ["loss", "eval_loss"],
              ["Training loss", "Evaluation loss"],
              f"loss_curve_split_{split_id}.png")
        _plot(training_logs.logs,
              ["eval_accuracy"],
              ["Evaluation accuracy"],
              f"accuracy_curve_split_{split_id}.png")
    except Exception as e:
        print(f"[Split {split_id}] Plotting skipped: {e}")

    
    shutil.rmtree(f"checkpoints_{split_id}", ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)
    done_flag = cache_dir + ".done"
    if os.path.exists(done_flag):
        os.remove(done_flag)

    # Free GPU memory before the next split
    del trainer, model
    import torch
    torch.cuda.empty_cache()

    print(f"[Split {split_id:>2d}] ✓ Done")
    return (split_id, f1)



# GPU worker — processes its queue of splits sequentially

def _gpu_worker(gpu_id: int, split_ids: list, hp: dict,
                result_queue: mp.Queue):
    """
    One long-lived process pinned to one GPU.
    Trains each assigned split one after another — no contention.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["WANDB_MODE"]           = "disabled"

    import torch
    assert torch.cuda.device_count() == 1
    print(f"[GPU {gpu_id}] Worker started — "
          f"{torch.cuda.get_device_name(0)} — "
          f"assigned splits: {split_ids}")

    for sid in split_ids:
        try:
            split_id, f1 = _train_one_split(sid, hp, gpu_id)
            result_queue.put((split_id, f1, None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Split {sid:>2d}] ✗ FAILED on GPU {gpu_id}: {e}")
            result_queue.put((sid, None, str(e)))

    print(f"[GPU {gpu_id}] Worker finished all splits.")



# Pre-build dataset caches

def prebuild_caches(split_ids):
    print("=" * 60)
    print("Pre-building dataset caches …")
    print("=" * 60)
    for sid in split_ids:
        cache_dir = f"_tmp_processed_split_{sid}"
        done_flag = cache_dir + ".done"
        if os.path.exists(done_flag):
            print(f"  Split {sid:>2d}: cache exists, skipping")
            continue
        print(f"  Split {sid:>2d}: building …")
        prepare_dataset(sid, cache_dir)
    print("All caches ready.\n")

def main():
    if len(sys.argv) > 1:
        split_ids = [int(x.replace(",", "")) for x in sys.argv[1:]]
    else:
        split_ids = list(range(NUM_SPLITS))

    if not os.path.exists(HP_FILE):
        print(f"ERROR: {HP_FILE} not found. Run HPO first.")
        sys.exit(1)

    with open(HP_FILE) as f:
        hp = json.load(f)
    print("Hyperparameters:")
    pprint(hp)

    prebuild_caches(split_ids)

    
    gpu_assignments = {gid: [] for gid in GPU_IDS}
    for i, sid in enumerate(split_ids):
        gpu = GPU_IDS[i % len(GPU_IDS)]
        gpu_assignments[gpu].append(sid)

    print("=" * 60)
    print(f"Training {len(split_ids)} models across "
          f"{len(GPU_IDS)} GPUs")
    for gid, sids in gpu_assignments.items():
        print(f"  GPU {gid}: splits {sids}")
    print("=" * 60)

    
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for gpu_id, sids in gpu_assignments.items():
        if not sids:
            continue
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, sids, hp, result_queue),
            name=f"gpu-{gpu_id}-worker",
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    
    results  = {}
    failures = {}
    while not result_queue.empty():
        split_id, f1, error = result_queue.get_nowait()
        if error is None:
            results[split_id] = f1
        else:
            failures[split_id] = error

    
    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    for sid in sorted(results):
        print(f"  Split {sid:>2d}:  F1 = {results[sid]:.4f}")
    if results:
        mean_f1 = sum(results.values()) / len(results)
        print(f"\n  Mean F1 across {len(results)} splits: {mean_f1:.4f}")
    if failures:
        print(f"\n  {len(failures)} FAILED splits:")
        for sid in sorted(failures):
            print(f"    Split {sid:>2d}: {failures[sid]}")
    print("=" * 60)

    summary_file = "training_results_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"results": results, "failures": failures}, f, indent=2)
    print(f"Results saved to {summary_file}")


if __name__ == "__main__":
    main()