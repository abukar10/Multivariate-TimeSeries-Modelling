"""
@author: Model Architecture update by Abukar Ali. Original author: S.Tahir.H.Rizvi. 
"""

"""

"""
"""

Enhanced version: Residual connections + Dropout
"""
#importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Importing the RevIN normalization layer
from .RevIN import RevIN


# Defining the GLinear model class
# This model is a simple linear model with residual connections and dropout layers
# It is designed for time series forecasting tasks and can handle multiple input features.

class Model(nn.Module):
    """
    Residual Stacked GLinear Model (Depth=4)

        # Initialize the model with the given sequence length, prediction length, and number of input features.
        # The model will consist of several linear blocks with residual connections and dropout layers.
        # it will also include a normalization layer to handle the input data.
        # and it will be used for time series forecasting tasks.
    """
    # Initialize the model with input configurations.
    # Extract Input length, prediction length, and number of input features 
    # from the config object passed during model creation.

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in

    
    

        # Initial normalization layer
        # This layer normalizes the input features across the batch
        # It is used to ensure that the input data is standardized before processing
        # RevIN is a normalization layer that can handle time series data   
        # It normalizes the input data and we will use it to denormalize it later
        
        self.revin_layer = RevIN(self.in_num_features)

        # Depth = 4 residual blocks (each with linear layer + activation + dropout)
        # Each block consists of a linear layer, GELU activation, and dropout
        # These blocks will process the input data sequentially
        # The blocks are designed to learn complex patterns in the data
        # Each block will take the output of the previous block as input    
        # The blocks will be stacked to create a deep model
        # Each block will have the same input and output dimensions


        # Define the first residual block:
        # - Linear layer maps input to same shape
        # - GeLU introduces non-linearity
        # - Dropout helps reduce overfitting

        self.block1 = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Define the second residual block:
        # Second residual block (same structure)
        self.block2 = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Define the third residual block:
        # Third residual block (same structure)
        self.block3 = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Define the fourth residual block:
        # Fourth residual block (same structure)
        # This block is the last one in the sequence

        self.block4 = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Final projection layer to map to prediction length
        # This layer projects the output of the last block to the prediction length
        # Converts the processed sequence of length `Input length` to the desired `pred_len`.

        #This creates a fully connected (linear) layer that transforms a tensor of shape:
        # [batch_size, seq_len] to [batch_size, pred_len]. example: [32, 96] to [32, 24]
        # Note: seq_len is the length of the input sequence, and pred_len is the length of the prediction window.
        

        self.output_proj = nn.Linear(self.seq_len, self.pred_len)
       

        # Residual projection for matching output dimensions if needed
        # If the sequence length and prediction length are different,
        # we need a projection layer to match the dimensions

        # if the number of time steps in the input sequence is not equal to the number of time steps in the prediction window,
        # then the residual input (residual) must be resized to match the output dimensions.

        # This transforms the residual from shape [batch, variables, seq_len] → [batch, variables, pred_len], making the addition possible.

        if self.seq_len != self.pred_len:
            self.residual_proj = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.residual_proj = nn.Identity()    # this is +x skip connection.



    # Forward pass of the model
    # This method defines how the input data flows through the model
    # It takes the input tensor `x` and processes it through the defined layers
    # The input tensor `x` is expected to have the shape [batch, time, variables]   

    def forward(self, x):
        
        # Normalize input: shape [batch, time, variables]
        # Step 1: Normalize the input across each instance
        # This ensures that the input data is standardized before processing


        x = self.revin_layer(x, 'norm')

        # Step 2: Rearrange dimensions for processing over time
        # Permute to [batch, variables, time] for temporal processing

        # this reoders the input tensor to have the shape [batch, variables, time]
        x = x.permute(0, 2, 1)

        # Initialize residual connection
        # Step 3: Create a residual connection
        # This will be used to add the original input back to the output later
        residual = x.clone()   # F(x) + x, see the paper for more details. 

        # Apply 3 stacked linear blocks
        # Step 3: Pass through 4 stacked linear blocks
        # Each block applies linear transformation, activation, and dropout

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        # Step 4: Project the output to the prediction window
        # Final projection to prediction window

        out = self.output_proj(out)  # [batch, variables, pred_len]

        # Applying residual connection 
        # Step 5: Apply skip connection from the original input
        # If the input and output dimensions differ, project the residual
        # to match the output dimensions
        # Residual connection: add original input to the output
        # Ensure residual is in the same shape as output
        # [batch, variables, pred_len]


        residual = self.residual_proj(residual)  # [batch, variables, pred_len]
        # Add the residual to the output
        # This allows the model to learn residuals effectively
        # out shape: [batch, variables, pred_len]
        # Ensure the output and residual are compatible for addition


        out = out + residual
        # Ensure the output is still in the shape [batch, variables, pred_len]
        # out shape: [batch, variables, pred_len]
        # out = out + residual

        # Step 6: Revert to original time-major format
        # Permute back to - new shape:  [batch, pred_len, variables]
        # This is necessary for the final output format
        # Permute to [batch, pred_len, variables] for final output
        out = out.permute(0, 2, 1)

        
        # Denormalize the output
        # Step 7: De-normalize the output

        out = self.revin_layer(out, 'denorm')

        # this is the final outputshape: [batch, pred_len, variables]
        return out  

