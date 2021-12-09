import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # either 3 or 6


GENOMES = { "mouse" : "/users/kcochran/genomes/mm10_no_alt_analysis_set_ENCODE.fasta",
            "human" : "/users/kcochran/genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta" }

ROOT = "/users/kcochran/projects/cs197_cross_species_domain_adaptation/"
DATA_DIR = ROOT + "data/"

SPECIES = ["mouse", "human"]

TFS = ["CTCF", "CEBPA", "HNF4A", "RXRA"]


import gzip
import random
import numpy as np
from pyfaidx import Fasta
from torch.utils.data import Dataset
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, log_loss
import torch
from torch.utils.data import DataLoader


### Data Loaders


class TrainGenerator(Dataset):
    letter_dict = {
        'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],
        'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],
        'T':[0,0,0,1],'N':[0,0,0,0]}

    def __init__(self, species, tf):
        self.posfile = DATA_DIR + species + "/" + tf + "/chr3toY_pos_shuf.bed.gz"
        self.negfile = DATA_DIR + species + "/" + tf + "/chr3toY_neg_shuf_run1_1E.bed.gz"
        self.converter = Fasta(GENOMES[species])
        self.batchsize = 400
        self.halfbatchsize = self.batchsize // 2
        self.current_epoch = 1

        self.get_coords()
        self.on_epoch_end()


    def __len__(self):
        return self.steps_per_epoch


    def get_coords(self):
        with gzip.open(self.posfile) as posf:
            pos_coords_tmp = [line.decode().split()[:3] for line in posf]  # expecting bed file format
            self.pos_coords = [(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]  # no strand consideration
        with gzip.open(self.negfile) as negf:
            neg_coords_tmp = [line.decode().split()[:3] for line in negf]
            self.neg_coords = [(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
            
        self.steps_per_epoch = int(len(self.pos_coords) / self.halfbatchsize)
        print(self.steps_per_epoch)
                

    def convert(self, coords):
        seqs_onehot = []
        for coord in coords:
            chrom, start, stop = coord
            seq = self.converter[chrom][start:stop].seq
            seq_onehot = np.array([self.letter_dict.get(x,[0,0,0,0]) for x in seq])
            seqs_onehot.append(seq_onehot)

        seqs_onehot = np.array(seqs_onehot)
        return seqs_onehot


    def __getitem__(self, batch_index):	
        # First, get chunk of coordinates
        pos_coords_batch = self.pos_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
        neg_coords_batch = self.neg_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]

        # if train_steps calculation is off, lists of coords may be empty
        assert len(pos_coords_batch) > 0, len(pos_coords_batch)
        assert len(neg_coords_batch) > 0, len(neg_coords_batch)

        # Second, convert the coordinates into one-hot encoded sequences
        pos_onehot = self.convert(pos_coords_batch)
        neg_onehot = self.convert(neg_coords_batch)

        # seqdataloader returns empty array if coords are empty list or not in genome
        assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
        assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]

        # Third, combine bound and unbound sites into one large array, and create label vector
        # We don't need to shuffle here because all these examples will correspond
        # to a simultaneous gradient update for the whole batch
        all_seqs = np.concatenate((pos_onehot, neg_onehot))
        labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],)))

        all_seqs = torch.tensor(all_seqs, dtype=torch.float).permute(0, 2, 1)
        labels = torch.tensor(labels, dtype=torch.float)
        assert all_seqs.shape[0] == self.batchsize, all_seqs.shape[0]
        return all_seqs, labels


    def on_epoch_end(self):
        # switch to next set of negative examples
        prev_epoch = self.current_epoch
        next_epoch = prev_epoch + 1

        # update file where we will retrieve unbound site coordinates from
        prev_negfile = self.negfile
        next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
        self.negfile = next_negfile

        # load in new unbound site coordinates
        with gzip.open(self.negfile) as negf:
            neg_coords_tmp = [line.decode().split()[:3] for line in negf]
            self.neg_coords = [(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]

        # then shuffle positive examples
        random.shuffle(self.pos_coords)


class ValGenerator(Dataset):
    letter_dict = {
        'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],
        'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],
        'T':[0,0,0,1],'N':[0,0,0,0]}

    def __init__(self, species, tf, return_labels = True):
        self.valfile = DATA_DIR + species + "/" + tf + "/chr1_random_1m.bed.gz"
        self.converter = Fasta(GENOMES[species])
        self.batchsize = 1000  # arbitrarily large number that will fit into memory
        self.return_labels = return_labels
        self.get_coords_and_labels()


    def __len__(self):
        return self.steps_per_epoch


    def get_coords_and_labels(self):
        with gzip.open(self.valfile) as f:
            coords_tmp = [line.decode().split()[:4] for line in f]  # expecting bed file format
        
        self.labels = [int(coord[3]) for coord in coords_tmp]
        self.coords = [(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]  # no strand consideration
        
        self.steps_per_epoch = int(len(self.coords) / self.batchsize)

    def convert(self, coords):
        seqs_onehot = []
        for coord in coords:
            chrom, start, stop = coord
            seq = self.converter[chrom][start:stop].seq
            seq_onehot = np.array([self.letter_dict.get(x,[0,0,0,0]) for x in seq])
            seqs_onehot.append(seq_onehot)

        seqs_onehot = np.array(seqs_onehot)
        return seqs_onehot


    def __getitem__(self, batch_index):	
        # First, get chunk of coordinates
        batch_start = batch_index * self.batchsize
        batch_end = (batch_index + 1) * self.batchsize
        coords_batch = self.coords[batch_start : batch_end]

        # if train_steps calculation is off, lists of coords may be empty
        assert len(coords_batch) > 0, len(coords_batch)

        # Second, convert the coordinates into one-hot encoded sequences
        onehot = self.convert(coords_batch)

        # array will be empty if coords are not found in the genome
        assert onehot.shape[0] > 0, onehot.shape[0]

        onehot = torch.tensor(onehot, dtype=torch.float).permute(0, 2, 1)
        
        if self.return_labels:
            labels = self.labels[batch_start : batch_end]
            labels = torch.tensor(labels, dtype=torch.float)
            return onehot, labels
        else:
            return onehot



### Performance Metric Functions

def print_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    preds = preds.squeeze()

    # this is the binary cross-entropy loss, same as in training
    print("Loss:\t", log_loss(labels, preds))
    print("auROC:\t", roc_auc_score(labels, preds))
    auPRC = average_precision_score(labels, preds)
    print("auPRC:\t", auPRC)
    print_confusion_matrix(preds, labels)
    return auPRC

def print_confusion_matrix(preds, labels):
    npthresh = np.vectorize(lambda t: 1 if t >= 0.5 else 0)
    preds_binarized = npthresh(preds)
    conf_matrix = confusion_matrix(labels, preds_binarized)
    print("Confusion Matrix (at t = 0.5):\n", conf_matrix)



### Model Architecture and Training Loop

class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.input_seq_len = 500
        num_conv_filters = 240
        lstm_hidden_units = 32
        fc_layer1_units = 1024
        fc_layer2_units = 512
        
        # Defining the layers to go into our model
        # (see the forward function for how they fit together)
        self.conv = torch.nn.Conv1d(4, num_conv_filters, kernel_size=20, padding=0)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(15, stride=15, padding=0)
        self.lstm = torch.nn.LSTM(input_size=num_conv_filters,
                                  hidden_size=lstm_hidden_units,
                                  batch_first=True)
        self.fc1 = torch.nn.Linear(in_features=lstm_hidden_units,
                                   out_features=fc_layer1_units)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(in_features=fc_layer1_units,
                                   out_features=fc_layer2_units)
        self.fc_final = torch.nn.Linear(in_features=fc_layer2_units,
                                        out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

        # The loss function we'll use -- binary cross-entropy
        # (this is the standard loss to use for binary classification)
        self.loss = torch.nn.BCELoss()

        # We'll store performance metrics during training in these lists
        self.train_loss_by_epoch = []
        self.source_val_loss_by_epoch = []
        self.source_val_auprc_by_epoch = []
        self.target_val_loss_by_epoch = []
        self.target_val_auprc_by_epoch = []

        # We'll record the best model we've seen yet each epoch
        self.best_state_so_far = self.state_dict()
        self.best_auprc_so_far = 1


    def forward(self, X):
        X_1 = self.relu(self.conv(X))
        # LSTM is expecting input of shape (batches, seq_len, conv_filters)
        X_2 = self.maxpool(X_1).permute(0, 2, 1)
        X_3, _ = self.lstm(X_2)
        X_4 = X_3[:, -1]  # only need final output of LSTM
        X_5 = self.relu(self.fc1(X_4))
        X_6 = self.dropout(X_5)
        X_7 = self.sigmoid(self.fc2(X_6))
        y = self.sigmoid(self.fc_final(X_7)).squeeze()
        return y
    
    def validation(self, data_loader):
        # only run this within torch.no_grad() context!
        losses = []
        preds = []
        labels = []
        for seqs_onehot_batch, labels_batch in data_loader:
            # push batch through model, get predictions, calculate loss
            preds_batch = self(seqs_onehot_batch.squeeze().cuda())
            labels_batch = labels_batch.squeeze()
            loss_batch = self.loss(preds_batch, labels_batch.cuda())
            losses.append(loss_batch.item())

            # storing labels + preds for auPRC calculation later
            labels.extend(labels_batch.detach().numpy())  
            preds.extend(preds_batch.cpu().detach().numpy())
            
        return np.array(losses), np.array(preds), np.array(labels)


    def fit(self, train_gen, source_val_data_loader, target_val_data_loader,
            optimizer, epochs=15):
        
        for epoch in range(epochs):
            torch.cuda.empty_cache()  # clear memory to keep stuff from blocking up
            
            print("=== Epoch " + str(epoch + 1) + " ===")
            print("Training...")
            self.train()
            
            # using a batch size of 1 here because the generator returns
            # many examples in each batch
            train_data_loader = DataLoader(train_gen,
                               batch_size = 1, shuffle = True)

            train_losses = []
            train_preds = []
            train_labels = []
            for seqs_onehot_batch, labels_batch in train_data_loader:
                # reset the optimizer; need to do each batch after weight update
                optimizer.zero_grad()

                # push batch through model, get predictions, and calculate loss
                preds = self(seqs_onehot_batch.squeeze().cuda())
                labels_batch = labels_batch.squeeze()
                loss_batch = self.loss(preds, labels_batch.cuda())

                # brackpropogate the loss and update model weights accordingly
                loss_batch.backward()
                optimizer.step()
                
                train_losses.append(loss_batch.item())
                train_labels.extend(labels_batch)
                train_preds.extend(preds.cpu().detach().numpy())

            self.train_loss_by_epoch.append(np.mean(train_losses))
            print_metrics(train_preds, train_labels)
            
            # load new set of negative examples for next epoch
            train_gen.on_epoch_end()

            
            # Assess model performance on same-species validation set
            print("Evaluating on source validation data...")
            
            # Since we don't use gradients during model evaluation,
            # the following two lines let the model predict for many examples
            # more efficiently (without having to keep track of gradients)
            self.eval()
            with torch.no_grad():
                source_val_losses, source_val_preds, source_val_labels = self.validation(source_val_data_loader)

                print("Validation loss:", np.mean(source_val_losses))
                self.source_val_loss_by_epoch.append(np.mean(source_val_losses))

                # calc auPRC over source validation set
                source_val_auprc = print_metrics(source_val_preds, source_val_labels)
                self.source_val_auprc_by_epoch.append(source_val_auprc)

                # check if this is the best performance we've seen so far
                # if yes, save the model weights -- we'll use the best model overall
                # for later analyses
                if source_val_auprc < self.best_auprc_so_far:
                    self.best_auprc_so_far = source_val_auprc
                    self.best_state_so_far = self.state_dict()
                
                
                # now repeat for target species data 
                print("Evaluating on target validation data...")
                
                target_val_losses, target_val_preds, target_val_labels = self.validation(target_val_data_loader)

                print("Validation loss:", np.mean(target_val_losses))
                self.target_val_loss_by_epoch.append(np.mean(target_val_losses))

                # calc auPRC over source validation set
                target_val_auprc = print_metrics(target_val_preds, target_val_labels)
                self.target_val_auprc_by_epoch.append(target_val_auprc)
                



### Setup + Train

# setup generators / data loaders for training and validation

# we'll make the training data loader in the training loop,
# since we need to update some of the examples used each epoch
train_gen = TrainGenerator("mouse", "CTCF")

source_val_gen = ValGenerator("mouse", "CTCF")
# using a batch size of 1 here because the generator returns
# many examples in each batch
source_val_data_loader = DataLoader(source_val_gen, batch_size = 1, shuffle = False)

target_val_gen = ValGenerator("human", "CTCF")
target_val_data_loader = DataLoader(target_val_gen, batch_size = 1, shuffle = False)

# initialize the model
model = BasicModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train!
model.cuda()
model.fit(train_gen, source_val_data_loader, target_val_data_loader, optimizer, epochs = 15)
model.cpu()

print("Done training!")
