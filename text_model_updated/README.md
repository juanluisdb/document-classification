## Updated to use Trainer API


```
python main.py --json_file PATH_TO_CONFIG_JSON
```

main.py file trains the model in 10 different splits of Small Tobacco Dataset.

Config file:

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


```
python ensemble.py --json_file PATH_TO_CONFIG_JSON
```

ensemble.py file test the full model in 10 different splits of Small Tobacco Dataset.

Config file:
* "num_classes": It should be 10 to used with small tobacco.
* "batch_size"
* "text_model_name": "bert-base-uncased" or "bert-base-cased"
* "eff_model": string with efficienet model, from "b0" to "b7"
* "path_bert_base": Path where bert checkpoints from small tobacco are located. It should be the same as ["TrainingArgs"]["output_dir"] in main.py
* "eff_path": Path where efficientnet checkpoints from small tobacco are located.
* "path_hdf5": Path to HDF5 files.