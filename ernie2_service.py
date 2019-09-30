import os
import csv

os.system('export CUDA_VISIBLE_DEVICES=0')

cls_names = {
    'Hong Kong',
    'London',
    'Toronto',
    'Paris',
    'Rome',
    'Sydney',
    'Dubai',
    'Bangkok',
	'Singapore',
	'Frankfurt'
}

def ERNIE2Client():
    use_cuda = 'true'
    batch_size = '8'
    output_file_name = "/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/ernie/train.{}.{}.ernie2.npy"

    init_pretraining_params = "/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/ERNIE-2.0/ERNIE/params"
    data_set = "/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/ernie2/train.{}.{}.tsv"
    vocab_path = "/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/ERNIE-2.0/ERNIE/vocab.txt"
    max_seq_len = '128'
    ernie_config_path = "/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/ERNIE-2.0/ERNIE/ernie_config.json"
    for cls in cls_names:
        for label in range(2):
            _output_file_name = output_file_name.format(cls, label)
            _data_set = data_set.format(cls, label)
            cmd = "python -u /DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/ERNIE-2.0/ernie_encoder_test.py  \
    --use_cuda {}  \
    --batch_size {}  \
    --output_dir '{}'  \
    --init_pretraining_params '{}'  \
    --data_set '{}'  \
    --vocab_path '{}'  \
    --max_seq_len {}  \
    --ernie_config_path '{}'".format(use_cuda, batch_size, _output_file_name, init_pretraining_params,
                                     _data_set, vocab_path, max_seq_len, ernie_config_path)
            os.system(cmd)


if __name__ == "__main__":
    ERNIE2Client()
