import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode some inputs
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
indexed_tokens_1 = tokenizer.encode(text_1)
indexed_tokens_2 = tokenizer.encode(text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

# Load pre-trained model (weights)
model = GPT2Model.from_pretrained('gpt2')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to('cuda')
tokens_tensor_2 = tokens_tensor_2.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    hidden_states_1, past = model(tokens_tensor_1)
    print(len(hidden_states_1))
    print(hidden_states_1[-1].size())
    # past can be used to reuse precomputed hidden state in a subsequent predictions
    # (see beam-search examples in the run_gpt2.py example).
    # hidden_states_2, past = model(tokens_tensor_2, past=past)
