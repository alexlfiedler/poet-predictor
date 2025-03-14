import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class PoemDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name='bert-base-uncased', max_length=512):
        self.data = pd.read_csv(csv_path)
        
        # Filter poets with at least 5 poems
        author_counts = self.data['Author'].value_counts()
        valid_authors = author_counts[author_counts >= 5].index
        self.data = self.data[self.data['Author'].isin(valid_authors)]
        #self.data = self.data[self.data['Author'].isin(author_counts[author_counts >= 5].index)]
        
        self.poems = self.data['Content'].tolist()
        self.authors = pd.factorize(self.data['Author'])[0]  # Encode authors as integers

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length


    def __len__(self):
        return len(self.poems)

    def __getitem__(self, idx):
        poem = self.poems[idx]
        encoded_poem = self.tokenizer(
            poem, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        author = self.authors[idx]
        return encoded_poem['input_ids'].squeeze().to(torch.long), torch.tensor(author, dtype=torch.long)


def get_data_loaders(base_path, batch_size=32):
    dataset = PoemDataset(base_path)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, 
                                            [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
