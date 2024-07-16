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
        "model_filename" : "tmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/tmodel"
    }