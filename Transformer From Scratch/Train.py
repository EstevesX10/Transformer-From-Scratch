# A Tokenizer comes before an Input Embedding in order to split a given sentence into multiple segments / single words
# The Tokenizer used was:
# -> Word Level Tokenizer --> each whitespace defines the boundary for each word
# After, it converts each word into a number (Input ID) which will incorporate the Vocabulary

# We can also create special Tokens:
# --> Padding
# --> Start of Sentence
# --> End of Sentence
# Note: All of these Tokens are important to Train the Transformer

import torch
from torch import (nn)
from torch.utils.data import (Dataset, DataLoader, random_split)

from Dataset import (BilingualDataset, causal_mask)

from Model import (Transformer, Build_Transformer)

from datasets import (load_dataset)
from tokenizers import (Tokenizer)
from tokenizers.models import (WordLevel)
from tokenizers.trainers import (WordLevelTrainer)
from tokenizers.pre_tokenizers import (Whitespace)
from pathlib import (Path)

def Get_All_Sentences(dataset, language):
    """
    := param: dataset - Dataset used to Train the Transformer
    := param: language - Language to which we are going to build the Tokenizer
    """
    for item in dataset:
        yield item['translation'][language]

def Get_or_Build_Tokenizer(config, dataset, language):
    """
    := param: config - Configuration of the Tokenizer
    := param: dataset - Dataset used to Train the Transformer
    := param: language - Language to which we are going to build the Tokenizer
    """
    # Path to Save the Tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format(language))

    # If the tokenizer does not exist, we create it
    if not Path.exists(tokenizer_path):
        # Creating and Defining the Tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # Specified what to attribute to unknown words
        tokenizer.pre_tokenizer = Whitespace
        tokenizer_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]"], min_frequency=2) # Specified the Special Tokens and the Minimum of occurences a word need to have to appear in the Vocabulary

        # Train and Save the Tokenizer
        tokenizer.train_from_iterator(Get_All_Sentences(dataset, language), trainer=tokenizer_trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load a previously existing Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Return the Tokenizer
    return tokenizer

def Get_Dataset(config):
    """
    := param: config - Configuration of the Model
    """
    # Extrating dynamically the Dataset from HuggingFace
    raw_dataset = load_dataset('opus_books', f'{config["source_language"]}-{config["target_language"]}', split='train')

    # Build the tokenizers
    tokenizer_source = Get_or_Build_Tokenizer(config, raw_dataset, config['source_language'])
    tokenizer_target = Get_or_Build_Tokenizer(config, raw_dataset, config['target_language'])

    # Split the Data (90% for Trainning and 10% for Testing / Validation)
    train_dataset_size = int(0.9 * len(raw_dataset))
    test_dataset_size = len(raw_dataset) - train_dataset_size
    raw_train_dataset, raw_test_dataset = random_split(raw_dataset, [train_dataset_size, test_dataset_size])

    # Creating the respective datasets from the splitted data
    train_dataset = BilingualDataset(raw_train_dataset, tokenizer_source, tokenizer_target, config['source_language'], config['target_language'], config['sequence_length'])
    test_dataset = BilingualDataset(raw_test_dataset, tokenizer_source, tokenizer_target, config['source_language'], config['target_language'], config['sequence_length'])

    # In order to choose the Max Sequence Length we need to find the Maximum length of the sentence in the source and target
    max_length_source = 0
    max_length_target = 0

    for item in raw_dataset:
        source_ids = tokenizer_source.encode(item['translation'][config['source_language']]).ids
        target_ids = tokenizer_source.encode(item['translation'][config['target_language']]).ids
        max_length_source = max(max_length_source, len(source_ids))
        max_length_target = max(max_length_target, len(target_ids))
    
    print(f"Max Length of Source Sentence: {max_length_source}")
    print(f"Max Length of Target Sentence: {max_length_target}")

    # Create the Data Loaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Return the DataLoaders of the Trainning and Testing / Validation as well as the source and target tokenizers
    return train_dataloader, test_dataloader, tokenizer_source, tokenizer_target

# Build the Model
def Get_Model(config, source_vocabulary_size, target_vocabulary_size):
    """
    := param: config
    := param: source_vocabulary_size
    := param: target_vocabulary_size
    """
    
    # Create a Transformer
    model = Build_Transformer(source_vocabulary_size, target_vocabulary_size, config['sequence_length'], config['sequence_length'], config['dim_model'])

    # Return the new Model
    return model