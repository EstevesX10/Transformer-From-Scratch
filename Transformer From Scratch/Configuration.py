from pathlib import (Path)

def Get_Configuration():
    return {
        "batch_size" : 8,
        "num_epochs" : 20,
        "lr" : 1e-4,
        "sequence_length" : 350,
        "dim_model" : 512,
        "source_language" : "en",
        "target_language" : "pt",
        "model_folder" : "Weights",
        "model_basename" : "tranformer_model_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "Runs/transformer_model"
    }

def Get_Weights_File_Path(config:dict, epoch:str):
    """
    := param: config
    := param: epoch
    """
    
    # Getting the Model Folder
    model_folder = config['model_folder']

    # Getting the name of the Model
    model_basename = config['model_basename']

    # Defining the Model's filename
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.') / model_folder / model_filename)