import torch
from torch.utils.tensorboard import (SummaryWriter)
import torchmetrics

from tokenizers import (Tokenizer)
from .Dataset import (causal_mask)
from .Model import (Transformer)

# Creating a Function to perform greedy decoding - used in the Validation Loop
def Greedy_Decode(model:Transformer, source, source_mask, tokenizer_source:Tokenizer, tokenizer_target:Tokenizer, max_length:int, device:torch.device):
    """
    := param: model
    := param: source
    := param: source_mask
    := param: tokenizer_source
    := param: tokenizer_target
    := param: max_length
    := param: device
    """
    
    # Creating necessary tokens and getting it's indices
    sos_token_idx = tokenizer_target.token_to_id('[SOS]')
    eos_token_idx = tokenizer_target.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_token_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_length:
            break

        # Create a Mask for the Target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)

        # Calculate the Output
        output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the probabilities of each token occuring next
        probabilities_next_token = model.project(output[:, -1])

        # Select the Token with the highest Probability [Greedy Search]
        _, next_word = torch.max(probabilities_next_token, dim=1)

        # Append the next word to the decoder_input
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        # Found the end of the sentence
        if next_word == eos_token_idx:
            break

    return decoder_input.squeeze(0)

# Create the Validation / Testing Loop
def Run_Validation(model:Transformer, validation_dataset, tokenizer_source:Tokenizer, tokenizer_target:Tokenizer, max_length:int, device:torch.device, print_message, global_step, writer:SummaryWriter, num_examples:int=2):
    """
    := param: model
    := param: validation_dataset
    := param: tokenizer_source
    := param: tokenizer_target
    := param: max_length
    := param: device
    := param: print_message - Function from tqdm
    := param: global_state
    := param: writer
    := param: num_examples
    """

    # Place the model into validation mode
    model.eval()

    # Infer 2 sentences and infer the output of the model
    counter = 0

    # Creating list to store the input and output sentences
    source_texts = []
    expected_texts = []
    predicted_texts = []

    # Size of the Control Window
    console_width = 80

    with torch.no_grad(): # Disabling the Gradient Calculation
        for batch in validation_dataset:
            counter += 1

            # Get the encoder input of current batch as well as its mask
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # Verify the Size of the Batch (Should be 1)
            assert encoder_input.size(0) == 1, "Batch Size must be equal to 1 to perform Validation!"

            # Get the Model output using greedy decoding
            model_output = Greedy_Decode(model, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_length, device)

            # -> Compare the Model Output with the expected label / results

            # Get all the texts involved in Validation
            source_text = batch['source_text'][0]
            target_text = batch['target_text'][0]
            predicted_text = tokenizer_target.decode(model_output.detach().cpu().numpy())

            # Save all the texts involved in Validation
            source_texts.append(source_text)
            expected_texts.append(target_text)
            predicted_texts.append(predicted_text)

            # Print it to the Console
            print_message('-'*console_width)
            print_message(f'[SOURCE]: {source_text}')
            print_message(f'[TARGET]: {target_text}')
            print_message(f'[PREDICTED]: {predicted_text}')

            if counter == num_examples:
                break

    # Send all this information into Tensorboard
    if writer: # Evaluate the Character Error Rate
        
        # Compute the Character Error Rate
        metric = torchmetrics.CharErrorRate()
        character_error_rate = metric(predicted_texts, expected_texts)
        writer.add_scalar('Validation Character Error Rate', character_error_rate, global_step)
        writer.flush()

        # Compute the Word Error Rate
        metric = torchmetrics.WordErrorRate()
        word_error_rate = metric(predicted_texts, expected_texts)
        writer.add_scalar('Validation Word Error Rate', word_error_rate, global_step)
        writer.flush()

        # Compute the BLEU Score
        metric = torchmetrics.BLEUScore()
        bleu_score = metric(predicted_texts, expected_texts)
        writer.add_scalar('Validation BLEU', bleu_score, global_step)
        writer.flush()