## Updated to use Trainer API


```
python main.py --json_file PATH_TO_CONFIG_JSON
```

The config file 

* "TrainingArgs": [TrainingArguments class](https://huggingface.co/transformers/v4.10.1/main_classes/trainer.html#transformers.TrainingArguments) arguments. Some of them are required to use it in the custom optimizer: 
    * 'learning_rate'
    * 'weight_decay'
    * 'adam_epsilon'
    * 'warmup_steps'
* "num_gpus": Total number of gpus used in training. It is required to calculate actual batch size
* "layerwise_learning_rate_decay": 0 to disable it.
* "learning_rate_head": If "layerwise_learning_rate_decay" is 0, it will be used as learning rate in classification head.
* "use_bertadam": Not correct bias in Adam, like in original BERT repository.
* "model_name": "bert-base-uncased" or "bert-base-cased"
* "path_hdf5": Path to HDF5 files.

Make sure you previously download and store in ```./cache/MODEL_NAME``` the model.