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
        self.sequence_length = sequence_lengh

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
        # Extract the Original pair from the HuggingFace Dataset
        source_target_pair = self.dataset[index]

        # Extract the source and target text
        source_text = source_target_pair['translation'][self.source_language]
        target_text = source_target_pair['translation'][self.target_language]

        # Convert each text into tokens and therefore into input ID's
        # Objectively, the Tokenizer will first split the sentence into single words and then will map each one into the respective number in the Vocabulary
        encoder_input_tokens = self.tokenizer_source.encode(source_text).ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids

        # PAD the sentence to reach the sequence length (Very important, so that the model always performs with fixed length)
        # Let's calculate how many padding tokens are necessary for both the encoder and decoder
        encoder_num_padidng_tokens = self.sequence_length - len(encoder_input_tokens) - 2 # We use -2 due to the SOS and EOS tokens
        decoder_num_padidng_tokens = self.sequence_length - len(decoder_input_tokens) - 1 # We use -1 since we only add the SOS token to the decoder side. Consequently, we only add the EOS token in the target / label (Final Output)

        if encoder_num_padidng_tokens < 0 or decoder_num_padidng_tokens < 0:
            raise ValueError('Sentence is too Long!')
        
        # Let's build both Tensors for the Encoder Input and the Decoder Input
        # Added the Start of Sentence and End of Sentence Tokens to the Source Text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.padding_token] * encoder_num_padidng_tokens, dtype=torch.int64)
            ]
        )
