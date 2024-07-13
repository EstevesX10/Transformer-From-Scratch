import math
import torch
from torch import (nn)

# Input Embedding
# -> Takes an input and converts it to an embedding (aka a vector of size 512)
class Input_Embeddings(nn.Module):
    def __init__(self, Dim_Model:int, Vocabulary_Size:int) -> None:
        """
        := param: Dim_Model - Dimension of the Final Vector
        := param: Vocabulary_Size - How many words there are in the Vocabulary
        """
        super.__init__()
        self.Dim_Model = Dim_Model
        self.Vocabulary_Size = Vocabulary_Size
        self.Embedding = nn.Embedding(Vocabulary_Size, Dim_Model)
    
    def forward(self, x:torch.Tensor):
        # DO the mapping based on the Embedding layer provided by pytorch
        return self.Embedding(x) * math.sqrt(self.Dim_Model)
    

# Positional Encoding (Conveys the Position of each word in the sentence)

class Positional_Encoding(nn.Module):
    def __init__(self, Dim_Model:int, Sequence_Length:int, Dropout:float) -> None:
        """
        := param: Dim_Model - Size of the Vector of Positional Enconding should be
        := param: Sequence_Length - Maximum Length of the Sentence
        := param: Dropout - Prevents the model from Overfitting
        """
        super().__init__()
        self.Dim_Model = Dim_Model
        self.Sequenece_Length = Sequence_Length
        self.Dropout = nn.Dropout(Dropout)

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

        # To save the Tensor along with the State of the Model (in a File), we need to register it to the Buffer
        self.register_buffer('pe', Pos_Encoding)

    def forward(self, x:torch.Tensor):
        # Adding Positional Encoding to every word in the sentence
        x = x + (self.Pos_Encoding[: , :x.shape[1], :]).requires_grad(False)

        # Apply the Dropout
        return self.Dropout(x)

# Layer Normalization
# -> Given a Batch of N items, we calculate the mean and variance of each one and consequently update their values based on their mean and variance
# -> Introduction of alpha (multiplication) and beta/bias (addition) parameters that introduce some fluctuations in the data since having all values in [0, 1] can be too restrictive

class Layer_Normalization(nn.Module):
    def __init__(self, eps:float=1e-6) -> None:
        """
        := param: eps - Error
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Used in Multiplication
        self.bias = nn.Parameter(torch.zeros(1)) # Used in Addition

    def forward(self, x:torch.Tensor):
        # Calculate the Mean
        mean = x.mean(dim = -1, keepdim=True)

        # Calculate the Standard Deviation
        std = x.std(dim = -1, keepdim=True)

        # Apply the Normalization
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward_Block(nn.Module):
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
        self.Dropout = nn.Dropout(Dropout)

        # Define the second matrix (W2 and b2)
        self.linear_2 = nn.Linear(Dim_FeedForward, Dim_Model)

    def forward(self, x:torch.Tensor):
        # Input Sentence - Tensor with shape (Batch, Sequence_Length, Dim_Model)
        # Convert it using linear_1 into another tensor of shape (Batch, Sequence_Length, Dim_FeedForward)
        # In the end, we convert it back using linear_2, obtaining the original shape (Batch, Sequence_Length, Dim_Model)
        self.linear_2(self.Dropout(torch.relu(self.linear_1(x))))

# Multi-Head Attention Block
# -> ...

class MultiHead_Attention_Block(nn.Module):
    def __init__(self, Dim_Model:int, Num_Heads:int, Dropout:float) -> None:
        """
        := param: Dim_Model - Dimensionality of the Input ad Output Layers
        := param: Num_Heads - Number of Heads
        := param: Dropout

        Note: To Divide the embedding vector into <Num_Heads> Heads, the Dim_Model should be divisible by the Num_Heads
        """

        super().__init__()
        self.Dim_Model = Dim_Model
        self.Num_Heads = Num_Heads
        self.Dropout = nn.Dropout(Dropout)

        # Making sure that the Dim_Model is divisible by the Num_Heads
        assert Dim_Model % Num_Heads == 0, "Dim_Model is not divisible by Num_Heads"

        # Dk -> Dim_Model // Num_Heads (According to the Source Material)
        self.D_k = Dim_Model // Num_Heads

        # -> Next Step: Getting the Matrices by which we are going to multiply the query, the key and the values as well as the Output Matrix (W_O)
        
        # Query Matrix [Shape (Dim_Model, Dim_Model)]
        self.W_q = nn.Linear(Dim_Model, Dim_Model)

        # Key Matrix [Shape (Dim_Model, Dim_Model)]
        self.W_k = nn.Linear(Dim_Model, Dim_Model)

        # Values Matrix [Shape (Dim_Model, Dim_Model)]
        self.W_v = nn.Linear(Dim_Model, Dim_Model)

        # Output Matrix [Shape (Dim_Model, Dim_Model)]
        self.W_o = nn.Linear(Dim_Model, Dim_Model)

        # Defining the Dropout
        self.Dropout = Dropout(Dropout)

        def forward(self, Query, Key, Values, Mask):
            """
            := param: Query
            := param: Key
            := param: Values
            := param: Mask - Used in order to prevent some words to interact with other words
            
            """
            # Q' Calculation
            query = self.W_q(Query) # From shape (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Dim_Model)

            key = self.W_k(Key) # From shape (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Dim_Model)
            
            values = self.W_v(Values) # From shape (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Dim_Model)

            # Split the Embedding [(Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Num,_Heads, D_k) -- <By Transposition> -> (Batch_Size, Num_Heads, Sequence_Length, D_k)]
            query = query.view(query.shape[0], query.shape[1], self.Num_Heads, self.D_k).transpose(1, 2)

if __name__ == "__main__":
    print("HELLO THERE")