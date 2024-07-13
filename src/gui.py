import gradio as gr
import torch
import pandas as pd
import glob
import os
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from XLSTM4Rec import xLSTM4Rec

def load_dataset_from_csv(csv_file):
    """
    Load MovieLens dataset from a CSV file.
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        tuple: Lists containing movie titles and corresponding item IDs.
    """
    df = pd.read_csv(csv_file)
    movie_list = df['movie_title'].tolist()
    movie_id_list = df['item_id'].tolist()
    return movie_list, movie_id_list

def load_model(model_path, config, train_data):
    """
    Load a saved model.
    Args:
        model_path (str): Path to the model file.
        config (Config): Configuration object.
        train_data (Dataset): Training dataset.
    Returns:
        Model: The loaded model in evaluation mode.
    """
    model = xLSTM4Rec(config, train_data.dataset).to(config['device'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Configuration
config_path = os.path.relpath('config.yaml')
config = Config(model=xLSTM4Rec, config_file_list=[config_path])

# Load CSV file and extract movie lists and IDs
csv_file_path = 'dataset/ml-1m/ml-1m.csv'
movie_list, movie_id_list = load_dataset_from_csv(csv_file_path)

# List available models
model_files = glob.glob('models/*.pt')


# Load dataset and model
dataset = create_dataset(config)
train_data, _,_ = data_preparation(config, dataset)

def recommend_movies(selected_movies, selected_model_path):
    """
    Recommend movies based on selected movies.
    Args:
        selected_movies (list of str): List of selected movies by the user.
        selected_model_path (str): Path to the selected model.
    Returns:
        str: Formatted string with top 5 movie recommendations.
    """
    model = load_model(selected_model_path, config, train_data)
    

    # Convert movie names to IDs
    selected_movie_ids = [movie_id_list[movie_list.index(movie)] for movie in selected_movies]

    # Predict using the model
    input_data = torch.tensor(selected_movie_ids,device='cuda').unsqueeze(0)
    with torch.no_grad():
        seq_lengths = torch.full((input_data.size(0),), input_data.size(1), dtype=torch.long,device='cuda')
        output = model(input_data, seq_lengths)

    # Process output to get top 5 recommendations
    _, top_indices = torch.topk(output, 5)
    top_movies = [movie_list[idx] for idx in top_indices[0]]

    return "\n".join(top_movies)

# Create Gradio interface
iface = gr.Interface(
    fn=recommend_movies,
    inputs=[
        gr.Dropdown(choices=movie_list, label="Select movies", multiselect=True, max_choices=10),
        gr.Dropdown(choices=model_files, label="Select model")
    ],
    outputs=gr.Textbox(label="Recommendation"),
    title="Movie Lens Recommender",
    description="Select the movies you have seen in order. The order matters."
)

iface.launch()
