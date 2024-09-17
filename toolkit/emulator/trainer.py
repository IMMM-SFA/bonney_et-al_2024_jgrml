import torch
import time
from datetime import datetime
import os
import json
import yaml
import pandas as pd
import numpy as np
from toolkit.emulator import models

class Trainer:
    def __init__(self, device, output_dir=None):
        self.device = device
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.output_dir = f"runs/run_{timestamp}"
        
    def _write_loss_csv(self, train_losses, valid_losses):
        loss_df = pd.DataFrame([train_losses, valid_losses])
        loss_df = loss_df.transpose()
        loss_df.columns = ["train_loss", "valid_loss"]
        loss_csv_file = os.path.join(self.output_dir, "loss.csv")
        os.makedirs(os.path.dirname(loss_csv_file), exist_ok=True)
        loss_df.to_csv(loss_csv_file)
        return loss_df
        
    def write_checkpoint(self, summary_dict, checkpoint_id):
        assert self.output_dir is not None
        
        # model checkpoint
        checkpoint_file = os.path.join(
            self.output_dir, 
            "checkpoints",
            f"checkpoint_{checkpoint_id}.pth.tar")
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_file)
        
        # summary data
        summary_file = os.path.join(
            self.output_dir, 
            "checkpoints",
            f"summary_{checkpoint_id}.json")
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(summary_file, "w") as outfile: 
            json.dump(summary_dict, outfile)
    
    def load_checkpoint(self, checkpoint_id):
        """Load from checkpoint"""
        assert self.output_dir is not None

        checkpoint_file = os.path.join(
            self.output_dir, 
            "checkpoints",
            f"checkpoint_{checkpoint_id}.pth.tar")
        self.model.load_state_dict(torch.load(checkpoint_file))
    
    def save_config(self):
        config_file = os.path.join(
            self.output_dir, 
            "config.yaml")
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as outfile: 
            yaml.dump(self.config, outfile, default_flow_style=False)
    
    def build(self, config):
        self.config = config

        input_time_steps = config["data"]["input_time_steps"]
        input_dim = config["data"]["input_dim"]
        output_dim = config["data"]["output_dim"]
        
        # train
        self.n_epochs = config["train"]["n_epochs"]

        # model
        model_config = config["model"]
        hidden_dim = model_config["hidden_dim"]
        num_layers = model_config["num_layers"]
        # model_config["input_dim"] = input_dim
        # model_config["output_dim"] = output_dim
        Model = getattr(models, model_config["name"])
        self.model = Model(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            num_layers=num_layers).to(self.device)

        # loss
        loss_config = config["loss"]          
        Loss = getattr(torch.nn, loss_config["name"])
        
        self.loss_func = Loss()
        
        # optimizer
        optim_config = config["optimizer"]
        Optim = getattr(torch.optim, optim_config["name"])
        self.optimizer = Optim(self.model.parameters(), **optim_config["args"])
        
        self.save_config()

    def epoch_train(self, data_loader):
        self.model.train()
        
        total_loss = 0
        
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)

            # Zero the parameter gradients
            self.model.zero_grad()
            self.optimizer.zero_grad()

            # Forward + Backward + Optimize
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            
        train_loss = total_loss / (i + 1)
        
        return train_loss
            
    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        
        total_loss = 0
        
        for i, (batch_input, batch_target) in enumerate(data_loader):
            # load batch data
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)

            # forward pass
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            
            total_loss += batch_loss.item()

            
        valid_loss = total_loss / (i + 1)

        return valid_loss
        
    @torch.no_grad()
    def get_targets_and_predictions(self, data_loader):
        self.model.eval()
        
        inputs = []
        true_values = []
        pred_values = []
        for i, (batch_input, batch_target) in enumerate(data_loader):
            # load batch data
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)

            # forward pass
            batch_output = self.model(batch_input)
            
            inputs.append(batch_input.cpu())
            true_values.append(batch_target.cpu())
            pred_values.append(batch_output.cpu())

        return np.array(torch.concat(inputs).cpu()), np.array(torch.concat(true_values).cpu()), np.array(torch.concat(pred_values).cpu())
    
    def train(self, train_data_loader, valid_data_loader=None, checkpoint_freq=10):

        train_losses = []
        valid_losses = []
        
        # Loop over epochs
        for i in range(self.n_epochs):
            summary = dict(epoch=i)

            # Train on this epoch
            train_summary = {}
            start_time = time.time()
            train_loss = self.epoch_train(train_data_loader)
            train_summary["train_loss"] = train_loss
            train_summary['time'] = time.time() - start_time
            
            for (k, v) in train_summary.items():
                summary[f'train_{k}'] = v

            # Evaluate on this epoch
            if valid_data_loader is not None:
                valid_summary = {}
                start_time = time.time()
                valid_loss = self.evaluate(valid_data_loader)
                valid_summary["loss"] = valid_loss
                valid_summary['time'] = time.time() - start_time
                
                for (k, v) in valid_summary.items():
                    summary[f'valid_{k}'] = v
                    
            # record losses
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            # Save summary, checkpoint
            if (i + 1) % checkpoint_freq == 0 or i + 1 == self.n_epochs:
                # self.save_summary(summary)
                if i + 1 == self.n_epochs:
                    self.write_checkpoint(summary_dict=summary, checkpoint_id="final")
                else:
                    self.write_checkpoint(summary_dict=summary, checkpoint_id=i)
            
                # write losses to csv
                self._write_loss_csv(train_losses, valid_losses)

        return
    
