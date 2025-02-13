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
from torch.utils.tensorboard import (SummaryWriter)

from .Dataset import (BilingualDataset)
from .Configuration import (Get_Tokenizer_File_Path, Get_Weights_File_Path)
from .Model import (Transformer, Build_Transformer)
from .Validation import (Run_Validation)

from datasets import (load_dataset)
from tokenizers import (Tokenizer)
from tokenizers.models import (WordLevel)
from tokenizers.trainers import (WordLevelTrainer)
from tokenizers.pre_tokenizers import (Whitespace)

from pathlib import (Path)
from tqdm import (tqdm)

def Get_All_Sentences(dataset, language):
    """
    := param: dataset - Dataset used to Train the Transformer
    := param: language - Language to which we are going to build the Tokenizer
    """
    for item in dataset:
        yield item['translation'][language]

def Get_or_Build_Tokenizer(config:dict, dataset, language):
    """
    := param: config - Configuration of the Tokenizer
    := param: dataset - Dataset used to Train the Transformer
    := param: language - Language to which we are going to build the Tokenizer
    """

    # Making Sure the Transformer Package Folder is created
    Path(config['package_folder']).mkdir(parents=True, exist_ok=True)

    # Making sure the Tokenizers folder is created
    Path(Path('.') / config['package_folder'] / config['tokenizers_folder']).mkdir(parents=True, exist_ok=True)

    # Path to Save the Tokenizer
    tokenizer_path = Get_Tokenizer_File_Path(config, language)

    # If the tokenizer does not exist, we create it
    if not Path.exists(tokenizer_path):
        # Creating and Defining the Tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # Specified what to attribute to unknown words
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer_trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # Specified the Special Tokens and the Minimum of occurences a word need to have to appear in the Vocabulary

        # Train and Save the Tokenizer
        tokenizer.train_from_iterator(Get_All_Sentences(dataset, language), trainer=tokenizer_trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load a previously existing Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Return the Tokenizer
    return tokenizer

def Get_Dataset(config:dict):
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
def Get_Model(config:dict, source_vocabulary_size, target_vocabulary_size):
    """
    := param: config
    := param: source_vocabulary_size
    := param: target_vocabulary_size
    """
    
    # Create a Transformer
    model = Build_Transformer(source_vocabulary_size, target_vocabulary_size, config['sequence_length'], config['sequence_length'], config['dim_model'])

    # Return the new Model
    return model

# Train the Model (given the configuration)
def Train_Model(config:dict, validate:bool=True):
    """
    := param: config
    """
    
    # Define the device on which we are going to place all the tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Making Sure the Transformer Package Folder is created
    Path(config['package_folder']).mkdir(parents=True, exist_ok=True)

    # Making sure the weights folder is created
    Path(Path('.') / config['package_folder'] / config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Load the Dataset
    train_dataloader, test_dataloader, tokenizer_source, tokenizer_target = Get_Dataset(config)

    # Create the Model and transfer it to the device
    model = Get_Model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    # Start TensorBoard -> Allows to visualize the loss through graphics and charts
    writer = SummaryWriter(config['package_folder'] + '/' + config['experiment_name'])

    # Create the Optimizer (Used the Adam Optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Create a way to manage the Model in order to restore the state of the Model and its optimizer in the future
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        # Get the filename 
        model_filename = Get_Weights_File_Path(config, config['preload'])
        print(f'Preloading Model {model_filename}')

        # Load the File and update the initial epoch and global step
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']

        # Load the state of the optimizer
        optimizer.load_state_dict(state['optimizer_state_dict'])
    
    # Define the loss funtion
    loss_funtion = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id('[PAD]'), label_smoothing=0.1).to(device) # label smoothing allows to transfer a small percentage of the occurence with the highest probability and redistribute it through the other occurences / "ouputs"

    # Create the Trainning Loop
    for epoch in range(initial_epoch, config['num_epochs'] + 1):
        # Define a Batch Iterator
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:
            model.train()

            # Get the encoder / decoder inputs
            encoder_input = batch['encoder_input'].to(device) # Shape (batch size, sequence length)
            decoder_input = batch['decoder_input'].to(device) # Shape (batch size, sequence length)

            # Get the encoder / decoder masks
            encoder_mask = batch['encoder_mask'].to(device) # Shape (batch size, 1, 1, sequence length)
            decoder_mask = batch['decoder_mask'].to(device) # Shape (batch size, 1, sequence length, sequence length)

            # -> Run the tensors through the Transformer
            
            # Calculate the output of the encoder
            encoder_output = model.encode(encoder_input, encoder_mask) # Shape (batch size, sequence length, dim model)
            
            # Calculate the output of the decoder
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # Shape (batch size, sequence length, dim model)

            # Map it back to the Vocabulary through Projection
            projection_output = model.project(decoder_output) # Shape (batch size, sequence length, target vocabulary size)

            # -> Now that we have the output of the Model, we want to compare it with the label

            # Extract the Label from the Batch
            label = batch['label'].to(device) # Shape (batch size, sequence length)

            # Compute the Loss [Shape from (batch size, sequence length, target vocabulary size) --> to (batch size * sequence length, target vocabulary size)]
            loss = loss_funtion(projection_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))

            # Update the Progress Bar with the calculated Loss
            batch_iterator.set_postfix({f"Loss" : f"{loss.item():6.3f}"})

            # Log the Loss into Tensorboard
            writer.add_scalar('Train Loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the Loss
            loss.backward()

            # Update the weights of the model
            optimizer.step()
            optimizer.zero_grad()

            # Increment the global step [Mostly used to track the Loss in Tensorboard]
            global_step += 1

        if validate:
            # Run the Validation
            Run_Validation(model, test_dataloader, tokenizer_source, tokenizer_target, config['sequence_length'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the Model at the end of N saving steps / epochs
        if ((epoch  % config['saving_step']) == 0):
            model_filename = Get_Weights_File_Path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)