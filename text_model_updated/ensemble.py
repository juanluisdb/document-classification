from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import argparse
import json
import numpy as np
from pathlib import Path, PurePath
from data_utils import H5Dataset
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
from torchvision import transforms



def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    '''
    Function to remove some encoding layers. 
    It will keep first num_layers_to_keep from encoding
    '''
    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()
 
    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep):
        newModuleList.append(oldModuleList[i])
 
    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList
 
    return copyOfModel


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json_file', 
        type=str, 
        default='configs/config_ensemble.json'
        )
    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        config = json.load(f)

    img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    results = []
    for file_number in range(1,11):
        print('File number: ', file_number)

        f = 'hdf5_small_tobacco_papers_audebert_' + str(file_number) + '.hdf5'
        hdf5_file = Path(config['path_hdf5']) / f

        bert_path = Path(config['path_bert_base']) / str(file_number)

        tokenizer = AutoTokenizer.from_pretrained(f"./cache/tokenizer/{config['text_model_name']}")
        bert_net = AutoModelForSequenceClassification.from_pretrained(bert_path)

        eff_net = EfficientNet.from_pretrained(
            'efficientnet-' + config['eff_model'], num_classes=1000)

        num_ftrs = eff_net._fc.in_features
        eff_net._fc = nn.Linear(num_ftrs, config['num_classes'])
        eff_net = torch.nn.DataParallel(eff_net)
        f = '50.025eff' + config['eff_model'] + '_aud' + str(file_number) + 'ft.pt'
        p = Path(config['eff_path']) / f
        checkpoint = torch.load(p)
        eff_net.load_state_dict(checkpoint['model_state_dict'])

        test_dataset = H5Dataset(
            path = hdf5_file,
            tokenizer = tokenizer,
            phase = 'test',
            use_img = True,
            img_transforms = img_transforms
            )
        test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = config['batch_size'],
            shuffle = False,
            num_workers = 0
            )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_net.to(device)
        eff_net.to(device)
        bert_net.eval()
        eff_net.eval()

        confusion_matrix = torch.zeros(10, 10)
        number_corrected = 0.0

        with torch.no_grad():
            for batch in test_loader:
                img = batch['img'].to(device)
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                label = batch['label'].to(device)
                
                out_bert = bert_net(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask
                )
                
                out_eff = eff_net(img)
                
                smax = nn.Softmax(dim=1)
                prob_out_bert = smax(out_bert.logits)
                prob_out_eff = smax(out_eff)

                final_preds = 0.5*prob_out_bert + 0.5*prob_out_eff
                values, preds = torch.max(final_preds.data, 1)
                for t, p in zip(label.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    if t.long() == p.long():
                        number_corrected += 1
                
                
        acc = number_corrected / len(test_dataset)
        results.append(acc)
        print(f'Acc split number {file_number}: {acc}')
    print('Results:', results)
    mean = np.asarray(results).mean()
    std = np.asarray(results).std()
    print(f'Mean: {mean}')
    print(f'Std: {std}')
        