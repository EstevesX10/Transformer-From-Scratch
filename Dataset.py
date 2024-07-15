import torch
from torch import (nn)
from torch.utils.data import (Dataset)

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_source, tokenizer_target, source_language, target_language, sequence_lengh) -> None:
        """
        := param: dataset - Dataset from HuggingFace
        := param: tokenizer_source
        := param: tokenizer_target
        := param: source_language
        := param: target_language
        := param: sequence_length
        """
        super().__init__()

        # Saving all given parameters
        self.dataset = dataset
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.source_language = source_language
        self.target_language = target_language

        # Create the Necessary tokens to be later used

        # Start of Sentence Token
        self.sos_token = torch.Tensor([tokenizer_source.token_to_id(['[SOS]'])], dtype=torch.int64)
        # End of Sentence Token
        self.eos_token = torch.Tensor([tokenizer_source.token_to_id(['[EOS]'])], dtype=torch.int64)
        # Padding Token
        self.padding_token = torch.Tensor([tokenizer_source.token_to_id(['[PAD]'])], dtype=torch.int64)
        
    def __len__(self):
        # Tells the length of the Dataset itself
        return len(self.dataset)
    
    def __getitem__(self, index:any) -> any:
        pass