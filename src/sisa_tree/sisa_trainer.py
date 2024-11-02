import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer

from datasets import load_dataset
from sisa_dataset import ShardedGroupDataset
from typing import Dict

class SisaTrainer:
    def __init__(self, sisa_dataset, model_name='distilbert-base-uncased', num_labels=2):
        self.sisa_dataset = sisa_dataset
        self.model_name = model_name
        self.num_labels = num_labels
        self.models = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_dataset(self, shards):
        pass    
    def train_on_shards(self, epochs=3, batch_size=8):
        pass
    def unlearn(self, indices, column_id='year', epochs=3, batch_size=8):
        pass
    
    def aggregate_models(self):
        pass   


if __name__ == '__main__':

    hf_dataset = ShardedGroupDataset(load_dataset("swj0419/BookMIA"))
    hf_dataset.generate_syntetic_column()
    
    
    shards = hf_dataset.group_by_column_id("year")
    print(shards)
    
    sisa_trainer = SisaTrainer(hf_dataset)
    sisa_trainer.train_on_shards(shards)
    # sisa_trainer.unlearn([0, 1, 2, 3, 4], column_id='year')
    # aggregated_model = sisa_trainer.aggregate_models()
    # print(aggregated_model)