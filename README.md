# ADL2020 Final Project
## Task Description
Perform information extraction on several japanese documents. Each document has 20 tags, as shown in figures below.
![](https://i.imgur.com/gkAZbBP.png =50%x)![](https://i.imgur.com/Mqljgyk.png =50%x)
The experiment includes 82 training data and 22 development data.

## Execution
### Preprocessing
```
python3.6 preprocessing.py --train_path {path_to_train_data} --dev_path {path_to_dev_data} --test_path {path_to_test_data}
```

### Training
```
python3.6 train.py --backbone {bert, cnn, dualbert}
```

### Predicting
```shell
python3.6 predict.py --model {path_to_model} --mode {dev, test} --backbone {bert, cnn, dualbert}
```

or simply

```shell
python3.6 test.py {path_to_test_set_directory}
```

## Result


| Model | Precision | Recall | F1 score |
| -------- | -------- | -------- | -------- |
| BERT(char only)     | Text     | Text     | 95.82     |
| BERT+CNN     | 96.34     | 96.38     | 96.16     |
| BERT(char+word)   | 96.11     | 96.51     | 96.10     |

## Author
r08922115 吳冠霖  
r08944034 洪商荃  
r08944035 呂翊愷