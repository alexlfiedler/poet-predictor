import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

class PoemDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        
        # Filter poets with at least 5 poems
        author_counts = self.data['author'].value_counts()
        valid_authors = author_counts[author_counts >= 5].index
        self.data = self.data[self.data['author'].isin(valid_authors)]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        poem = self.data.iloc[idx]['Content']
        author = self.data.iloc[idx]['Author']

        # Tokenization
        encoding = self.tokenizer(
            poem,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Convert author to integer label
        label = torch.tensor(self.data['author'].astype('category').cat.codes[idx])

        return { 'input_ids': encoding['input_ids'].squeeze(0),
                 'attention_mask': encoding['attention_mask'].squeeze(0),
                 'label': label }


def get_data_loaders(base_path, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = PoemDataset(base_path, tokenizer)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, 
                                            [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader