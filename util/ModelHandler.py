from time import time

import torch
import numpy as np


class ModelHandler():
    def __init__(self, 
                 ModelClass, 
                 model_args_dict,
                 optimizer,
                 optimizer_args_dict,
                 criterion):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.model = ModelClass(**model_args_dict).to(self.device)
        
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args_dict)
        
        self.criterion = criterion()
        
    def train(self, train_dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            loss = 0
            t0 = time()
            for batch, _ in train_dataloader:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                out = self.model(batch)
                
                train_loss = self.criterion(out, batch)
                
                train_loss.backward()
                
                self.optimizer.step()
                
                loss += train_loss.item()                
            loss = loss / len(train_dataloader)
            t1 = time()
            print(f"epoch: {epoch+1}/{epochs}, loss={loss}, {t1-t0:.2f}s")
            
    def predict(self, data_source):
        self.model.eval()
        if isinstance(data_source, torch.Tensor):
            data_source = data_source.to(self.device)
            return self.model(data_source).cpu().detach().numpy()
        elif isinstance(data_source, torch.utils.data.dataloader.DataLoader):
            preds = []
            print_every = 50
            counter = 0
            for batch, _ in data_source:
                batch = batch.to(self.device)
                out = \
                    self.model(batch).cpu().detach().numpy()
                preds.append(out)
                counter += 1
                if counter % print_every == 0:
                    print(f"Instances are extracted: ({counter}/{len(data_source)})")
            preds = np.array(preds).reshape(-1, preds[0].shape[-1])
            return preds
        else:
            raise NotImplementedError()
            
    def extract_representation(self, data_source):
        self.model.eval()
        if isinstance(data_source, torch.Tensor):
            data_source = data_source.to(self.device)
            preds = self.model.extract_representation(data_source).cpu().detach().numpy()
            return preds.squeeze()
        elif isinstance(data_source, torch.utils.data.dataloader.DataLoader):
            preds = []
            print_every = 50
            counter = 0
            for batch, _ in data_source:
                batch = batch.to(self.device)
                out = \
                    self.model.extract_representation(batch).cpu().detach().numpy()
                preds.append(out)
                counter += 1
#                 if counter % print_every == 0:
#                     print(f"Instances are extracted: ({counter}/{len(data_source)})")
            return np.concatenate(preds).squeeze()
            
        else:
            raise NotImplementedError()
            
    