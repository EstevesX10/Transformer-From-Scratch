from pathlib import (Path)

def Get_Configuration() -> dict:
    return {
        "batch_size" : 8,
        "num_epochs" : 100,
        "saving_step": 10,
        "lr" : 1e-4,
        "sequence_length" : 350,
        "dim_model" : 512,
        "source_language" : "en",
        "target_language" : "pt",
        "package_folder" : "Transformer",
        "model_folder" : "Trained_Models",
        "model_basename" : "tranformer_model_",
        "preload" : None,
        "tokenizers_folder" : "Tokenizers",
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "Runs/transformer_model"
    }

def Get_Tokenizer_File_Path(config:dict, language:str) -> str:
    """
    := param: config
    := param: language
    """

    # Get Package Folder
    package_folder = config['package_folder']

    # Get the Tokenizers Folder
    tokenizers_folder = config['tokenizers_folder']

    # Get the Tokenizer Filename
    tokenizer_filename = config['tokenizer_file'].format(language)

    return (Path('.') / package_folder / tokenizers_folder / tokenizer_filename)

def Get_Weights_File_Path(config:dict, epoch:str) -> str:
    """
    := param: config
    := param: epoch
    """
    
    # Get the Package Folder
    package_folder = config['package_folder']

    # Getting the Model Folder
    model_folder = config['model_folder']

    # Getting the name of the Model
    model_basename = config['model_basename']

    # Defining the Model's filename
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.') / package_folder / model_folder / model_filename)