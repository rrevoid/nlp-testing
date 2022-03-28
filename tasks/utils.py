import torch
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from tqdm import tqdm
from IPython.display import clear_output
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_model_and_tokenizer(name):
    torch.manual_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    for param in model.parameters():
        param.requires_grad = False

    if name.startswith("distilbert"):
        for param in model.pre_classifier.parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model, tokenizer

def evaluate_model(model, tokenizer, loader, device='cuda'):
    preds = []
    correct = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            text = data['sentence']
            encoded_input = tokenizer(text, return_tensors='pt', padding=True, 
                                  max_length = 128, truncation=True).to(device)

            labels = data['label'].to(device)
            outputs = model(**encoded_input).logits

            preds += outputs.argmax(axis=1)
            correct += labels

    correct = torch.tensor(correct).cpu()
    preds = torch.tensor(preds).cpu()

    print('Accuracy:', (correct == preds).to(torch.float).mean().item())
    print('F1 score:', f1_score(correct, preds))

def train_model(model, tokenizer, optimizer, criterion, scheduler, loader, 
                num_epochs=1, freq=10, device='cuda'):
    i = 0
    for epoch in range(num_epochs):
        train_losses = []
        val_losses = []

        model.train()
        for data in tqdm(loader):
            text = data['sentence']
            encoded_input = tokenizer(text, return_tensors='pt', padding=True, 
                                      max_length = 128, truncation=True).to(device)
            labels = data['label'].to(device)

            outputs = model(**encoded_input).logits

            loss = criterion(outputs, labels)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            scheduler.step()

            if i % freq == 0:
                clear_output()
                plt.plot(train_losses)
                plt.show()