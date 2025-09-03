# \"\"\"Train script for IndoAddr NER using HuggingFace Trainer.

# Example:
# python scripts/train.py --data_dir data --output_dir models/indoaddr-ner --model_name indobenchmark/indobert-base-p1 --epochs 6
# \"\"\"
import argparse
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import os

LABEL_LIST = [
    "O",
    "B-PROVINCE","I-PROVINCE",
    "B-CITY","I-CITY",
    "B-DISTRICT","I-DISTRICT",
    "B-VILLAGE","I-VILLAGE",
    "B-STREET","I-STREET",
    "B-RT","I-RT",
    "B-RW","I-RW",
    "B-POSTALCODE","I-POSTALCODE"
]

label2id = {l:i for i,l in enumerate(LABEL_LIST)}
id2label = {i:l for l,i in label2id.items()}

def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    return items

def to_dataset(items):
    return Dataset.from_dict({ "tokens": [x["tokens"] for x in items], "ner_tags": [x["labels"] for x in items] })

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",   
        max_length=128
    )

    all_labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None or word_idx >= len(label_seq):
                label_ids.append(-100)
            else:
                if word_idx != previous_word_idx:
                    label = label_seq[word_idx]
                    label_ids.append(label2id[label])
                else:
                    label = label_seq[word_idx]
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    label_ids.append(label2id[label])
                previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def main(args):
    # load data
    train = load_jsonl(os.path.join(args.data_dir, 'train.jsonl'))
    valid = load_jsonl(os.path.join(args.data_dir, 'valid.jsonl'))
    test = load_jsonl(os.path.join(args.data_dir, 'test.jsonl'))
    ds = DatasetDict({
        "train": to_dataset(train),
        "validation": to_dataset(valid),
        "test": to_dataset(test)
    })
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = ds.map(lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id)
    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy='epoch',
        learning_rate=3e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        fp16=False,
        report_to='none'
    )
    seqeval = evaluate.load('seqeval')
    def compute_metrics(p):
        preds, labels = p
        preds = np.argmax(preds, axis=2)
        true_preds, true_labels = [], []
        for pred, lab in zip(preds, labels):
            curp, curl = [], []
            for p_id, l_id in zip(pred, lab):
                if l_id != -100:
                    curp.append(id2label[p_id])
                    curl.append(id2label[l_id])
            true_preds.append(curp)
            true_labels.append(curl)
        res = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            'precision': res['overall_precision'],
            'recall': res['overall_recall'],
            'f1': res['overall_f1']
        }
    trainer = Trainer(model=model, args=args_tr, train_dataset=tokenized['train'], eval_dataset=tokenized['validation'], tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='indobenchmark/indobert-base-p1')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args)
