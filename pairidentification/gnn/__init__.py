from gnn.trainer import GNNTrainer
"""
Python module for holding our PyTorch trainers.
Trainers here inherit from the BaseTrainer and implement the logic for
constructing the model as well as training and evaluation.
"""

def get_trainer(**trainer_args):
    """
    Factory function for retrieving a trainer.
    """
    return GNNTrainer(**trainer_args)