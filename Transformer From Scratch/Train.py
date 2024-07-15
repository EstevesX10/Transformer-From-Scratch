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
    pass

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