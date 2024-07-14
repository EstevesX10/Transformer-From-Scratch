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
        """
        := param: x
        """

        # Calculate the Mean
        mean = x.mean(dim = -1, keepdim=True)

        # Calculate the Standard Deviation
        std = x.std(dim = -1, keepdim=True)

        # Apply the Normalization
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class Feed_Forward_Block(nn.Module):
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
        """
        := param: x
        """

        # Input Sentence - Tensor with shape (Batch, Sequence_Length, Dim_Model)
        # Convert it using linear_1 into another tensor of shape (Batch, Sequence_Length, Dim_FeedForward)
        # In the end, we convert it back using linear_2, obtaining the original shape (Batch, Sequence_Length, Dim_Model)
        self.linear_2(self.Dropout(torch.relu(self.linear_1(x))))

# Multi-Head Attention Block
# -> Calculates the MultiHead Attention Output given a Query, a Key and Values

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

            # Return the 
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
            query = query.view(query.shape[0], query.shape[1], self.Num_Heads, self.D_k).transpose(1, 2)
            key = key.view(key.shape[0], key.shape[1], self.Num_Heads, self.D_k).transpose(1, 2)
            values = values.view(values.shape[0], values.shape[1], self.Num_Heads, self.D_k).transpose(1, 2)

            # Calculate the Attention 
            Output, self.attention_scores = MultiHead_Attention_Block.attention(Query, Key, Values, Mask, self.Dropout)

            # (Batch_Size, Num_Heads, Sequence_Length, D_k) --> (Batch_Size, Sequence_Length, Num_Heads, D_k) --> (Batch_Size, Sequence_Length, Dim_Model)
            Output = Output.transpose(1, 2).contiguous().view(Output.shape[0], -1, self.Num_Heads * self.D_k) 

            # Multiply the Output with W_0 and return its value
            # Shape (Batch_Size, Sequence_Length, Dim_Model) --> (Batch_Size, Sequence_Length, Dim_Model)
            return self.W_o(Output)

# Residual Connection -> Connection that allows to Skip/Redirect the Output towards multiple Layers

class Residual_Connection(nn.Module):
    def __init__(self, Dropout:float) -> None:
        """
        := param: Dropout
        """
        super().__init__()
        self.Dropout = nn.Dropout(Dropout)
        self.norm = Layer_Normalization()

    def forward(self, x:torch.Tensor, SubLayer):
        """
        := param: x
        := param: SubLayer -> Previous Layer
        """
        return x + self.Dropout(SubLayer(self.norm(x)))

# Encoder Block
# -> Contains two Main Blocks: MultiHead Attention Block and the FeedForward Block
# -> It also includes 2 steps of Addition and Normalization between the Outputs of each block and the Outputs from External Residual Connections

class Encoder_Block(nn.Module):
    def __init__(self, Self_Attention_Block:MultiHead_Attention_Block, Feed_Forward_Block:Feed_Forward_Block, Dropout:float) -> None:
        """
        := param: Self_Attention_Block
        := param: Feed_Forwaard_Block
        := param: Dropout
        """
        super().__init__()

        # Save the given Blocks
        self.Self_Attention_Block = Self_Attention_Block
        self.Feed_Forward_Block = Feed_Forward_Block
        
        # Define the 2 Residual Connections
        self.Residual_Connections = nn.ModuleList([Residual_Connection(Dropout) for _ in range(2)])

        def forward(self, x:torch.Tensor, Source_Mask):
            """
            := param: x
            := param: Source_Mask - Mask that we want to apply to the input of the Encoder (To prevent the interaction between the padding word with other words)
            """

            # Making the First Residual Connection (takes into account the query, key, values and Mask)
            x = self.Residual_Connections[0](x, lambda x : self.Self_Attention_Block(x, x, x, Source_Mask))

            # Making the Second Residual Connection (Skips the MultiHead Attention Block and is redirected to the FeedFoward Block)
            x = self.Residual_Connections[1](x, self.Feed_Forward_Block)

            # Return the Final Value
            return x

# Encoder
# -> The Encoder contains N Encoder Blocks. Therefore it takes into account each one when forwarding the input message throughut the entire system

class Encoder(nn.Module):
    def __init__(self, Layers:nn.ModuleList) -> None:
        """
        := param: Layers - Layers of the Encoder (Has up to N Layers)
        """
        super().__init__()
        self.Layers = Layers
        self.Norm = Layer_Normalization()

    def forward(self, x:torch.Tensor, Mask):
        """
        := param: x
        := param: Mask
        """
        
        # Iterate through the Layer
        for Layer in self.Layers:
            x = Layer(x, Mask)

        # Apply Layer Normalization and Return the Output
        return self.Norm(x)


# Decoder Block
# -> Contains 3 Main Blocks:
#     - Masked Multi-Head Attention Block [Receives the Output Embedding which in pratical terms is equal to the Input Embedding]
#     - Multi-Head Attention Block [Takes into account the Key and Values that were outputed by the Decoder]
#     - Feed Forward Block
# -> It also contains 3 additional steps of addition and Normalization

class Decoder_Block(nn.Module):
    def __init__(self, Self_Attention_Block:MultiHead_Attention_Block, Cross_Attention_Block:MultiHead_Attention_Block, Feed_Forward_Block:Feed_Forward_Block, Dropout:float) -> None:
        """
        := param: Self_Attention_Block
        := param: Cross_Attention_Block - Allows to merge the output of the Encoder (Key and Values) with the Query that comes from the previous layers of the decoder block
        := param: Feed_Forward_Block
        := param: Dropout
        """
        super().__init__()

        # Saving the Blocks
        self.Attention_Block = Self_Attention_Block
        self.Cross_Attention_Block = Cross_Attention_Block
        self.Feed_Forward_Block = Feed_Forward_Block

        # Defining the Residual Connections (In this Case we have 3 of them)
        self.Residual_Connections = nn.Module([Residual_Connection(Dropout) for _ in range(3)])

    def forward(self, x:torch.Tensor, Encoder_Output, Source_Mask, Target_Mask):
        """
        := param: x - Input of the Decoder
        := param: Encoder_Output - Output of the Encoder
        := param: Source_Mask - Mask applied to the Encoder
        := param: Target_Mask - Mask applied to the Decoder
        """

        # Calculate the Self Attention [First Part of the Decoder Block]
        x = self.Residual_Connections[0](x, lambda x : self.Attention_Block(x, x, x, Target_Mask))

        # Calculate the Cross Attention
        x = self.Residual_Connections[1](x, lambda x : self.Cross_Attention_Block(x, Encoder_Output, Encoder_Output, Source_Mask))
        
        # Finally, we add the Feed Forward Block
        x = self.Residual_Connections[2](x, self.Feed_Forward_Block)

        # Return the Output / Final Value of the Decoder Block
        return x
    
# Decoder
# -> The Decoder contains N Decoder Blocks. Therefore it takes into account each one when forwarding the input message throughut the entire system

class Decoder(nn.Module):
    def __init__(self, Layers:nn.ModuleList) -> None:
        """
        := param: Layers - Layers of the Decoder (Has up to N Layers)
        """
        super().__init__()
        self.Layers = Layers
        self.Norm = Layer_Normalization()

    def forward(self, x, Encoder_Output, Source_Mask, Target_Mask):
        """
        := param: x - Input of the Decoder
        := param: Encoder_Output - Output of the Encoder
        := param: Source_Mask - Mask applied to the Encoder
        := param: Target_Mask - Mask applied to the Decoder
        """

        # Iterate through each Layer
        for Layer in self.Layers:
            x = Layer(x, Encoder_Output, Source_Mask, Target_Mask)
        
        # Apply Layer Normalization and Return the Output
        return self.Norm(x)
        
if __name__ == "__main__":
    # Translation Task [English to Italian]
    print("HELLO THERE")