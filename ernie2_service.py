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



ERNIE_PATH = '/home/mlsnrs/data/data/pxd/ERNIE/'
DATA_PATH = '/home/mlsnrs/data/data/pxd/bert_privacy/data/part/'



def prepare_tsv(inpath, outpath):
    out_file = open(outpath,'w+')
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['label', 'text_a'])
    f = open(inpath, 'r')
    sents = [x[:-1] for x in f if x[:-1] != '']
    for sent in sents:
        tsv_writer.writerow(['0', sent])
    return outpath

        
def ERNIE2Client(sents = None):
    use_cuda = 'true'
    batch_size = '1'
    output_file_name = "test.ernie2.npy"
    data_set = DATA_PATH + "hand.1.txt"


    data_set = prepare_tsv(data_set, DATA_PATH + "hand.1.csv")
    
    init_pretraining_params = ERNIE_PATH + "ERNIE-2.0/params"
    vocab_path = ERNIE_PATH + "ERNIE-2.0/vocab.txt"
    max_seq_len = '128'
    ernie_config_path = ERNIE_PATH + "ERNIE-2.0/ernie_config.json"
    
    
    cmd = "python -u {}ernie_encoder.py  \
    --use_cuda {}  \
    --batch_size {}  \
    --output_dir '{}'  \
    --init_pretraining_params '{}'  \
    --data_set '{}'  \
    --vocab_path '{}'  \
    --max_seq_len {}  \
    --ernie_config_path '{}'".format(ERNIE_PATH, use_cuda, batch_size, output_file_name, init_pretraining_params,
                                     data_set, vocab_path, max_seq_len, ernie_config_path)
    os.system(cmd)


if __name__ == "__main__":
    ERNIE2Client()
