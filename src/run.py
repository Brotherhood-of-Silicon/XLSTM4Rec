import sys
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_flops,
    get_environment,
    set_color,
)
from XLSTM4Rec import xLSTM4Rec
import os


# Configuration
config_path=os.path.relpath('config.yaml')
config = Config(model=xLSTM4Rec, config_file_list=[config_path])
init_seed(config['seed'], config['reproducibility'])
    
# Logger initialization
init_logger(config)
logger = getLogger()
logger.info(sys.argv)
logger.info(config)

# Dataset creation
dataset = create_dataset(config)
logger.info(dataset)

# Dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

# Model loading and initialization
init_seed(config["seed"] + config["local_rank"], config["reproducibility"])

model = xLSTM4Rec(config, train_data.dataset).to(config['device'])
logger.info(model)

# FLOPs calculation
transform = construct_transform(config)
flops = get_flops(model, dataset, config["device"], logger, transform)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

# FLOPs calculation
transform = construct_transform(config)
flops = get_flops(model, dataset, config["device"], logger, transform)
logger.info(set_color("FLOPs", "blue") + f": {flops}")

# Trainer initialization
trainer = Trainer(config, model)

# Model training
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, show_progress=config["show_progress"]
)

# Model evaluation
test_result = trainer.evaluate(
    test_data, show_progress=config["show_progress"]
)
    
# Environment logging
environment_tb = get_environment(config)
logger.info(
    "The running environment of this training is as follows:\n"
    + environment_tb.draw()
)

logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
logger.info(set_color("test result", "yellow") + f": {test_result}")