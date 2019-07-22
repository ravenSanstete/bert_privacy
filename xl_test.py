import torch
from pytorch_transformers import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary from wikitext 103)
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

# Tokenized input
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)

# Convert token to vocabulary indices
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

# Load pre-trained model (weights)
model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to('cuda')
tokens_tensor_2 = tokens_tensor_2.to('cuda')
model.to('cuda')

with torch.no_grad():
    # Predict hidden states features for each layer
    hidden_states_1, mems_1 = model(tokens_tensor_1)
    print(len(hidden_states_1))
    print(hidden_states_1[-1].size())
    print(len(mems_1))
    print(mems_1[-1].size())
    # We can re-use the memory cells in a subsequent call to attend a longer context
    # hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)
