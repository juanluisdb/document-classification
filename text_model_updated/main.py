from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from datasets import load_metric
import argparse
import json
import numpy as np
from pathlib import Path, PurePath
from data_utils import H5Dataset
import copy
from torch import nn
import math
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()
 
    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep):
        newModuleList.append(oldModuleList[i])
 
    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList
 
    return copyOfModel

def get_optimizer_grouped_parameters(model, model_type, learning_rate, 
    learning_rate_head, weight_decay,layerwise_learning_rate_decay):

    no_decay = ["bias", "LayerNorm.weight"]
    bert_identifiers = ['embedding', 'encoder', 'pooler']
    if layerwise_learning_rate_decay > 0:
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
                "weight_decay": 0.0,
                "lr": learning_rate,
            },
        ]

        # initialize lrs for every layer
        num_layers = model.config.num_hidden_layers
        layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
        layers.reverse()
        lr = learning_rate
        for layer in layers:
            lr *= layerwise_learning_rate_decay
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": lr
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr
                },
            ]
    else:
        optimizer_grouped_parameters = [
                {
                    'params': [param for name, param in model.named_parameters()
                            if any(identifier in name for identifier in bert_identifiers) and
                            not any(identifier_ in name for identifier_ in no_decay)],
                    'lr': learning_rate,
                    'weight_decay': weight_decay
                 },
                {
                    'params': [param for name, param in model.named_parameters()
                            if any(identifier in name for identifier in bert_identifiers) and
                            any(identifier_ in name for identifier_ in no_decay)],
                    'lr': learning_rate,
                    'weight_decay': 0.0
                 },
                {
                    'params': [param for name, param in model.named_parameters()
                            if not any(identifier in name for identifier in bert_identifiers)],
                    'lr': learning_rate_head,
                    'weight_decay': 0.0
                 }
        ]
    return optimizer_grouped_parameters

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='configs/config.json')
    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        config = json.load(f)


    tokenizer = AutoTokenizer.from_pretrained(f"./cache/tokenizer/{config['model_name']}")
    

    metric = load_metric("./cache/accuracy.py")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    bs = config['TrainingArgs']['per_device_train_batch_size'] * config['num_gpus']
    train_steps = math.ceil(800 / bs) * config['TrainingArgs']['num_train_epochs'] 
    print(train_steps)

    results = []
    for file_number in range(1,11):
        print('File number: ', file_number)

        f = 'hdf5_small_tobacco_papers_audebert_' + str(file_number) + '.hdf5'
        hdf5_file = Path(config['path_hdf5']) / f
        config['TrainingArgs']["output_dir"] = config['TrainingArgs']["output_dir"]


        train_dataset = H5Dataset(
            path=hdf5_file,
            tokenizer=tokenizer,
            phase= 'train'
            )
        val_dataset = H5Dataset(
            path=hdf5_file,
            tokenizer=tokenizer,
            phase= 'val'
            )
        test_dataset = H5Dataset(
            path=hdf5_file,
            tokenizer=tokenizer,
            phase= 'test'
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            f"./cache/{config['model_name']}", 
            num_labels=10
            )
        model = deleteEncodingLayers(model, 6)


        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model, 
            'bert', 
            config['TrainingArgs']['learning_rate'], 
            config['learning_rate_head'], 
            config['TrainingArgs']['weight_decay'],
            config['layerwise_learning_rate_decay']
        )

        optimizer = AdamW(
            grouped_optimizer_params,
            lr=config['TrainingArgs']['learning_rate'],
            eps=config['TrainingArgs']['adam_epsilon'],
            correct_bias = not config['use_bertadam']
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = config['TrainingArgs']['warmup_steps'],
            num_training_steps = train_steps
        )

        training_args = TrainingArguments(**config['TrainingArgs'])
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler)
        )


        trainer.train()
        r = trainer.predict(test_dataset)
        print('test acc:', r.metrics['test_accuracy'])
        results.append(r.metrics['test_accuracy'])

    print('Results:', results)
    mean = np.asarray(results).mean()
    std = np.asarray(results).std()
    print(f'Mean: {mean}')
    print(f'Std: {std}')
           