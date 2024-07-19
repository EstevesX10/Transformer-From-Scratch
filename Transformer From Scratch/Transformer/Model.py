# This Project focuses mainly on using a Tranformer in a Translation Task [English to Portuguese]
# However, the Transformer can be used outside this spectrum

import math
import torch
from torch import (nn)

# Input Embedding
# -> Takes an input and converts it to an embedding (aka a vector of size 512)
class InputEmbeddings(nn.Module):
    def __init__(self, Dim_Model:int, Vocabulary_Size:int) -> None:
        """
        := param: Dim_Model - Dimension of the Final Vector
        := param: Vocabulary_Size - How many words there are in the Vocabulary
        """
        super().__init__()
        self.dim_model = Dim_Model
        self.vocabulary_size = Vocabulary_Size
        self.embedding = nn.Embedding(Vocabulary_Size, Dim_Model)
    
    def forward(self, x:torch.Tensor):
        # Do the mapping based on the Embedding layer provided by pytorch
        return self.embedding(x) * math.sqrt(self.dim_model)

# Positional Encoding (Conveys the Position of each word in the sentence)

class PositionalEncoding(nn.Module):
    def __init__(self, Dim_Model:int, Sequence_Length:int, Dropout:float) -> None:
        """
        := param: Dim_Model - Size of the Vector of Positional Enconding should be
        := param: Sequence_Length - Maximum Length of the Sentence
        := param: Dropout - Prevents the model from Overfitting
        """
        super().__init__()
        self.dim_model = Dim_Model
        self.sequenece_length = Sequence_Length
        self.dropout = nn.Dropout(Dropout)

        # Create a Matrix of Shape (Sequence_Length, Dim_Model)
        Pos_Encoding = torch.zeros(Sequence_Length, Dim_Model)

        # Create a Vector Position of Shape (Sequence_Length) - Represents the Position of the Word inside the Sentence
        Word_Position = torch.arange(0, Sequence_Length, dtype=torch.float).unsqueeze(1) # Tensor of Shape (Sequence_Length. 1)
        
        # Creating the Denominator of the Formula
        Denom_Term = torch.exp(torch.arange(0, Dim_Model, 2).float() * (-math.log(10000.0) / Dim_Model))

        # Apply the sin to the Even Positions
        Pos_Encoding[:, 0::2] = torch.sin(Word_Position * Denom_Term)
        
        # Apply the cos to the Odd Positions
        Pos_Encoding[:, 1::2] = torch.cos(Word_Position * Denom_Term)

        # Add Batch Dimension - So that we can apply it to full sentence
        Pos_Encoding = Pos_Encoding.unsqueeze(0) # Tensor of shape (1, Sequence_Length, Dim_Model)

        # To save the Tensor along with the State of the Model (in a File), we need to register it as a Buffer
        self.register_buffer('Pos_Encoding', Pos_Encoding)

    def forward(self, x:torch.Tensor):
        # Adding Positional Encoding to every word in the sentence
        x = x + (self.Pos_Encoding[: , :x.shape[1], :]).requires_grad_(False)

        # Apply the Dropout
        return self.dropout(x)

# Layer Normalization
# -> Given a Batch of N items, we calculate the mean and variance of each one and consequently update their values based on their mean and variance
# -> Introduction of alpha (multiplication) and beta/bias (addition) parameters that introduce some fluctuations in the data since having all values in [0, 1] can be too restrictive

class LayerNormalization(nn.Module):
    def __init__(self, features:int, eps:float=1e-6) -> None:
        """
        := param: features
        := param: eps - Error
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # Used in Multiplication
        self.bias = nn.Parameter(torch.zeros(features)) # Used in Addition

    def forward(self, x:torch.Tensor):
        """
        := param: x
        """

        # Calculate the Mean
        mean = x.mean(dim = -1, keepdim=True)

        # Calculate the Standard Deviation
        std = x.std(dim = -1, keepdim=True)

        # Apply the Normalization
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, Dim_Model:int, Dim_FeedForward:int, Dropout:float) -> None:
        """
        := param:  Dim_Model - Dimensionality of the Input ad Output Layers
        := param:  Dim_FeedForward - Dimensionality of the Inner-Layer
        := param:  Dropout - Prevents the model from Overfitting
        """
        super().__init__()

        # Define the first matrix (W1 and b1)
        self.linear_1 = nn.Linear(Dim_Model, Dim_FeedForward)
        
        # Define the Dropout
        self.dropout = nn.Dropout(Dropout)

        # Define the second matrix (W2 and b2)
        self.linear_2 = nn.Linear(Dim_FeedForward, Dim_Model)

    def forward(self, x:torch.Tensor):
        """
        := param: x
        """

        # Input Sentence - Tensor with shape (Batch, Sequence_Length, Dim_Model)
        # Convert it using linear_1 into another tensor of shape (Batch, Sequence_Length, Dim_FeedForward)
        # In the end, we convert it back using linear_2, obtaining the original shape (Batch, Sequence_Length, Dim_Model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Multi-Head Attention Block
# -> Calculates the MultiHead Attention Output given a Query, a Key and Values

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, Dim_Model:int, Num_Heads:int, Dropout:float) -> None:
        """
        := param: Dim_Model - Dimensionality of the Input ad Output Layers
        := param: Num_Heads - Number of Heads
        := param: Dropout

        Note: To Divide the embedding vector into <Num_Heads> Heads, the Dim_Model should be divisible by the Num_Heads
        """
        super().__init__()
        self.dim_model = Dim_Model
        self.num_heads = Num_Heads
        self.dropout = nn.Dropout(Dropout)

        # Making sure that the Dim_Model is divisible by the Num_Heads
        assert Dim_Model % Num_Heads == 0, "Dim_Model is not divisible by Num_Heads"

        # Dk -> Dim_Model // Num_Heads (According to the Source Material)
        self.D_k = Dim_Model // Num_Heads

        # -> Next Step: Getting the Matrices by which we are going to multiply the query, the key and the values as well as the Output Matrix (W_O)
        
        # Query Matrix [Shape (Dim_Model, Dim_Model)]
        self.W_q = nn.Linear(Dim_Model, Dim_Model, bias=False)

        # Key Matrix [Shape (Dim_Model, Dim_Model)]
        self.W_k = nn.Linear(Dim_Model, Dim_Model, bias=False)

        # Values Matrix [Shape (Dim_Model, Dim_Model)]
        self.W_v = nn.Linear(Dim_Model, Dim_Model, bias=False)

        # Output Matrix [Shape (Dim_Model, Dim_Model)]
        self.W_o = nn.Linear(Dim_Model, Dim_Model, bias=False)

    @staticmethod
    def attention(Query:torch.Tensor, Key:torch.Tensor, Values:torch.Tensor, Mask, Dropout:nn.Dropout):
        # Get D_k based on the shape of the Query
        D_k = Query.shape[-1]

        # Apply part of the formula from the Paper
        attention_scores = (Query @ Key.transpose(-2, -1)) / math.sqrt(D_k) # Shape (Batch_Size, Num_Heads, Sequence_Length, D_k) --> (Batch_Size, Num_Heads, Sequence_Length, Sequence_Length) 

        # Apply the Mask
        if Mask is not None:
            attention_scores.masked_fill_(Mask == 0, -1e9)

        # Apply the Softmax Function
        attention_scores = attention_scores.softmax(dim = -1) # Shape (Batch_Size, Num_Heads, Sequence_Length, Sequence_Length)

        # Apply Dropout
        if Dropout is not None:
            attention_scores = Dropout(attention_scores)

        # Return the attention scores which can be used for visualization
        return (attention_scores @ Values, attention_scores)

    def forward(self, Query:torch.Tensor, Key:torch.Tensor, Values:torch.Tensor, Mask):
        """
        := param: Query
        := param: Key
        := param: Values
        := param: Mask - Used in order to prevent some words to interact with other words
        
        """
        # Q' Calculation
        query = self.W_q(Query) # From shape (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Dim_Model)

        # K' Calculation
        key = self.W_k(Key) # From shape (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Dim_Model)
        
        # V' Calculation
        values = self.W_v(Values) # From shape (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Dim_Model)

        # Split the Embedding into small matrices [(Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Num,_Heads, D_k) -- <By Transposition> -> (Batch_Size, Num_Heads, Sequence_Length, D_k)]
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.D_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.D_k).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.num_heads, self.D_k).transpose(1, 2)

        # Calculate the Attention 
        Output, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, values, Mask, self.dropout)

        # (Batch_Size, Num_Heads, Sequence_Length, D_k) --> (Batch_Size, Sequence_Length, Num_Heads, D_k) --> (Batch_Size, Sequence_Length, Dim_Model)
        Output = Output.transpose(1, 2).contiguous().view(Output.shape[0], -1, self.num_heads * self.D_k) 

        # Multiply the Output with W_0 and return its value
        # Shape (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Dim_Model)
        return self.W_o(Output)

# Residual Connection -> Connection that allows to Skip/Redirect the Output towards multiple Layers

class ResidualConnection(nn.Module):
    def __init__(self, Features:int, Dropout:float) -> None:
        """
        := param: Features
        := param: Dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(Dropout)
        self.norm = LayerNormalization(Features)

    def forward(self, x:torch.Tensor, SubLayer):
        """
        := param: x
        := param: SubLayer -> Previous Layer
        """
    
        return x + self.dropout(SubLayer(self.norm(x)))

# Encoder Block
# -> Contains two Main Blocks: MultiHead Attention Block and the FeedForward Block
# -> It also includes 2 steps of Addition and Normalization between the Outputs of each block and the Outputs from External Residual Connections

class EncoderBlock(nn.Module):
    def __init__(self, Features:int, Self_Attention_Block:MultiHeadAttentionBlock, Feed_Forward_Block:FeedForwardBlock, Dropout:float) -> None:
        """
        := param: Features
        := param: Self_Attention_Block
        := param: Feed_Forwaard_Block
        := param: Dropout
        """
        super().__init__()

        # Save the given Blocks
        self.self_attention_block = Self_Attention_Block
        self.feed_forward_block = Feed_Forward_Block
        
        # Define the 2 Residual Connections
        self.residual_connections = nn.ModuleList([ResidualConnection(Features, Dropout) for _ in range(2)])

    def forward(self, x:torch.Tensor, Source_Mask):
        """
        := param: x
        := param: Source_Mask - Mask that we want to apply to the input of the Encoder (To prevent the interaction between the padding word with other words)
        """

        # Making the First Residual Connection (takes into account the query, key, values and Mask)
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, Source_Mask))

        # Making the Second Residual Connection (Skips the MultiHead Attention Block and is redirected to the FeedFoward Block)
        x = self.residual_connections[1](x, self.feed_forward_block)

        # Return the Final Value
        return x

# Encoder
# -> The Encoder contains N Encoder Blocks. Therefore it takes into account each one when forwarding the input message throughut the entire system

class Encoder(nn.Module):
    def __init__(self, Features:int, Layers:nn.ModuleList) -> None:
        """
        := param: Features
        := param: Layers - Layers of the Encoder (Has up to N Layers)
        """
        super().__init__()
        self.layers = Layers
        self.norm = LayerNormalization(Features)

    def forward(self, x:torch.Tensor, Mask):
        """
        := param: x
        := param: Mask
        """
        
        # Iterate through the Layer
        for layer in self.layers:
            x = layer(x, Mask)

        # Apply Layer Normalization and Return the Output
        return self.norm(x)

# Decoder Block
# -> Contains 3 Main Blocks:
#     - Masked Multi-Head Attention Block [Receives the Output Embedding which in pratical terms is equal to the Input Embedding]
#     - Multi-Head Attention Block [Takes into account the Key and Values that were outputed by the Decoder]
#     - Feed Forward Block
# -> It also contains 3 additional steps of addition and Normalization

class DecoderBlock(nn.Module):
    def __init__(self, Features:int, Self_Attention_Block:MultiHeadAttentionBlock, Cross_Attention_Block:MultiHeadAttentionBlock, Feed_Forward_Block:FeedForwardBlock, Dropout:float) -> None:
        """
        := param: Features
        := param: Self_Attention_Block
        := param: Cross_Attention_Block - Allows to merge the output of the Encoder (Key and Values) with the Query that comes from the previous layers of the decoder block
        := param: Feed_Forward_Block
        := param: Dropout
        """
        super().__init__()

        # Saving the Blocks
        self.self_attention_block = Self_Attention_Block
        self.cross_attention_block = Cross_Attention_Block
        self.feed_forward_block = Feed_Forward_Block

        # Defining the Residual Connections (In this Case we have 3 of them)
        self.residual_connections = nn.ModuleList([ResidualConnection(Features, Dropout) for _ in range(3)])

    def forward(self, x:torch.Tensor, Encoder_Output, Source_Mask, Target_Mask):
        """
        := param: x - Input of the Decoder
        := param: Encoder_Output - Output of the Encoder
        := param: Source_Mask - Mask applied to the Encoder
        := param: Target_Mask - Mask applied to the Decoder
        """

        # Calculate the Self Attention [First Part of the Decoder Block]
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, Target_Mask))

        # Calculate the Cross Attention
        x = self.residual_connections[1](x, lambda x : self.cross_attention_block(x, Encoder_Output, Encoder_Output, Source_Mask))
        
        # Finally, we add the Feed Forward Block
        x = self.residual_connections[2](x, self.feed_forward_block)

        # Return the Output / Final Value of the Decoder Block
        return x
    
# Decoder
# -> The Decoder contains N Decoder Blocks. Therefore it takes into account each one when forwarding the input message throughut the entire system

class Decoder(nn.Module):
    def __init__(self, Features:int, Layers:nn.ModuleList) -> None:
        """
        := param: Features
        := param: Layers - Layers of the Decoder (Has up to N Layers)
        """
        super().__init__()
        self.layers = Layers
        self.norm = LayerNormalization(Features)

    def forward(self, x, Encoder_Output, Source_Mask, Target_Mask):
        """
        := param: x - Input of the Decoder
        := param: Encoder_Output - Output of the Encoder
        := param: Source_Mask - Mask applied to the Encoder
        := param: Target_Mask - Mask applied to the Decoder
        """

        # Iterate through each Layer
        for layer in self.layers:
            x = layer(x, Encoder_Output, Source_Mask, Target_Mask)
        
        # Apply Layer Normalization and Return the Output
        return self.norm(x)

# Projection Layer [Called Linear Layer in the Diagram of the Paper]
# -> Projects / Converts the Embedding [Ouput of the Decoder] into the Vocabulary 

class ProjectionLayer(nn.Module):
    # Simply a Linear Layer that converts the Output from Size of Dim_Model to the Vocabulary_Size
    def __init__(self, Dim_Model:int, Vocabulary_Size:int) -> None:
        """
        := param: Dim_Model 
        := param: Vocabulary_Size
        """
        super().__init__()
        self.projection_layer = nn.Linear(Dim_Model, Vocabulary_Size)

    def forward(self, x:torch.Tensor):
        """
        := param: x
        """

        # Shape Conversion: (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Vocabulary_Size)
        # We also apply the log of softmax to maintain numerical stability and return the Value

        # return torch.log_softmax(self.projection_layer(x), dim = -1)
        return self.projection_layer(x)
    
# Transformer
# -> Processes the Input throughout the components of the system

class Transformer(nn.Module):
    def __init__(self, Encoder:Encoder, Decoder:Decoder, Source_Embedding:InputEmbeddings, Target_Embedding:InputEmbeddings, Source_Position:PositionalEncoding, Target_Position:PositionalEncoding, Projection_Layer:ProjectionLayer) -> None:
        """
        := param: Encoder
        := param: Decoder
        := param: Source_Embedding - Input Embedding for the Source Language 
        := param: Target_Embedding - Input Embedding for the Target Language
        := param: Source_Position
        := param: Target_Position
        := param: Projection_Layer
        """
        super().__init__()
        
        # Saving the Components of the Transformer
        self.encoder = Encoder
        self.decoder = Decoder
        self.source_embedding = Source_Embedding
        self.target_embedding = Target_Embedding
        self.source_position = Source_Position
        self.target_position = Target_Position
        self.projection_layer = Projection_Layer

    def encode(self, Source, Source_Mask):
        """
        := param: Source
        := param: Source_Mask
        """
        
        # Apply the Embedding
        Source = self.source_embedding(Source)

        # Apply the Positional Encoding
        Source = self.source_position(Source)

        # Apply the Encoder
        return self.encoder(Source, Source_Mask)

    def decode(self, Encoder_Output:torch.Tensor, Source_Mask:torch.Tensor, Target, Target_Mask):
        """
        := param: Encoder_Output
        := param: Source_Mask
        := param: Target
        := param: Target_Mask
        """

        # Apply the Target Embedding to the Target Sentence
        Target = self.target_embedding(Target)

        # Apply the Positional Encoding to the Target Sentence
        Target = self.target_position(Target)

        # Apply the Decoder
        return self.decoder(Target, Encoder_Output, Source_Mask, Target_Mask)

    def project(self, x:torch.Tensor):
        """
        := param: x
        """

        # Simply apply the Projection (Goes from the Embedding to the Vocabulary Size)
        return self.projection_layer(x)

# Build_Transformer Function 
# -> Merges all the previous Blocks, allowing to create a custom Transformer with given hyperparameters

def Build_Transformer(Source_Vocabulary_Size:int, Target_Vocabulary_Size:int, Source_Sequence_Length:int, Target_Sequence_Length:int, Dim_Model:int=512, N:int=6, Num_Heads:int=8, Dropout:float=0.1, D_ff:int = 2048) -> Transformer:
    """
    := param: Source_Vocabulary_Size
    := param: Target_Vocabulary_Size
    := param: Source_Sequence_Length
    := param: Target_Sequence_Length
    := param: Dim_Model
    := param: N - Number of Layers (Number of Encoder and Decoder Blocks)
    := param: Num_Heads - Number of Heads
    := param: Dropout
    := param: D_ff
    """
    
    # Create the Embedding Layers
    Source_Embedding = InputEmbeddings(Dim_Model, Source_Vocabulary_Size)
    Target_Embedding = InputEmbeddings(Dim_Model, Target_Vocabulary_Size)

    # Create the Positional Encoding Layers
    Source_Position = PositionalEncoding(Dim_Model, Source_Sequence_Length, Dropout)
    Target_Position = PositionalEncoding(Dim_Model, Target_Sequence_Length, Dropout)

    # Create the Encoder Blocks
    Encoder_Blocks = []
    for _ in range(N):
        Encoder_Self_Attention_Block = MultiHeadAttentionBlock(Dim_Model, Num_Heads, Dropout)
        Feed_Forward_Block = FeedForwardBlock(Dim_Model, D_ff, Dropout)
        Encoder_Block = EncoderBlock(Dim_Model, Encoder_Self_Attention_Block, Feed_Forward_Block, Dropout)
        Encoder_Blocks.append(Encoder_Block)

    # Create the Decoder Blocks
    Decoder_Blocks = []
    for _ in range(N):
        Decoder_Self_Attention_Block = MultiHeadAttentionBlock(Dim_Model, Num_Heads, Dropout)
        Decoder_Cross_Attention_Block = MultiHeadAttentionBlock(Dim_Model, Num_Heads, Dropout)
        Feed_Forward_Block = FeedForwardBlock(Dim_Model, D_ff, Dropout)
        Decoder_Block = DecoderBlock(Dim_Model, Decoder_Self_Attention_Block, Decoder_Cross_Attention_Block, Feed_Forward_Block, Dropout)
        Decoder_Blocks.append(Decoder_Block)

    # Create the Encoder and Decoder
    Encoder_ = Encoder(Dim_Model, nn.ModuleList(Encoder_Blocks))
    Decoder_ = Decoder(Dim_Model, nn.ModuleList(Decoder_Blocks))

    # Create the Projection Layer
    Projection_Layer = ProjectionLayer(Dim_Model, Target_Vocabulary_Size)

    # Build the Transformer
    Transformer_ = Transformer(Encoder_, Decoder_, Source_Embedding, Target_Embedding, Source_Position, Target_Position, Projection_Layer)

    # Initialize the Parameters
    for param in Transformer_.parameters():
        if param.dim() > 1 :
            nn.init.xavier_uniform_(param)

    # Return the Transformer
    return Transformer_
