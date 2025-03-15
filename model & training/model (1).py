import torch
import torch.nn as nn
from transformers import AutoModel

class PoetClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, num_classes=1012):
        super(PoetClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_out = self.dropout(pooled_output)
        logits = self.fc(dropped_out)
        return logits

