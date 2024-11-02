"""Main module."""


from datasets import load_dataset


from datasets.arrow_dataset import Dataset
import numpy as np


class ShardedGroupDataset(Dataset):

    def __init__(self, dataset: Dataset):
        #TODO: use all splits ?
        self.dataset = dataset["train"]
    
    def generate_syntetic_column(self):
        percentages = [0.1, 0.2, 0.3, 0.2, 0.2]  # 10%, 20%, 30%, 20%, 20%
        years = [2021, 2020, 2019, 2018, 2017]
        num_books = len(self.dataset)
        year_distribution = np.random.choice(years, num_books, p=percentages)

        def add_year_column(example, idx):
            example["year"] = year_distribution[idx]
            return example
        
        self.dataset = self.dataset.map(add_year_column, with_indices=True)
        
    def group_by_column_id(self, column_id: str):
        
        shards = {}

        for col_val in self.dataset.unique(column_id):
            shards[col_val] = self.dataset.filter(lambda x: x[column_id] == col_val)

        self.shards = shards
        return shards
    
    def remove_row(self):
        #removes the row from the dataset
        pass
    



        
   
    
if __name__ == '__main__':
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    model_name = 'gaunernst/bert-tiny-uncased'   
    hf_dataset = ShardedGroupDataset(load_dataset("swj0419/BookMIA"))
    hf_dataset.generate_syntetic_column()
    
    shards = hf_dataset.group_by_column_id("year")
    print(shards)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_dataset(dataset):
        def tokenize_function(examples):
            columns = [col for col in examples.keys() if col not in ['year', 'label']]
            
            texts = [
                ' '.join([str(examples[col][i]) for col in columns])
                for i in range(len(examples['label']))
            ]
            
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512
            )
            
            tokenized["labels"] = examples["label"]
            return tokenized 
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        return tokenized_dataset
       

    for shard_key, shard_dataset in shards.items():
        print(f"Training on shard: {shard_key}, size: {len(shard_dataset)}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        
        tokenized_shard = tokenize_dataset(shard_dataset)
        
        training_args = TrainingArguments(
            output_dir=f"./results/{shard_key}",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_steps=10,
            save_strategy='no',
            learning_rate=2e-5,
            weight_decay=0.01,
        )
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            acc = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
            return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_shard,
        )
        
        trainer.train()
