#!/bin/bash

# Script to set up and run the XLSTM4Rec project

# Clone the repository
git clone https://github.com/yourusername/XLSTM4Rec.git
cd XLSTM4Rec

# Create and activate the conda environment
conda create --name xlstm4rec_env --file requirements.txt -y
conda activate xlstm4rec_env

# Navigate to the source directory
cd src

# Train the model
python run.py

# Instructions for further steps
echo "To run the Gradio interface for the Movielens1M dataset, execute: python gui.py"
echo "To modify training parameters, edit the config.yaml file."
echo "To use the Google Colab notebook, upload src/collab.ipynb to your Google Drive and open it in Colab."