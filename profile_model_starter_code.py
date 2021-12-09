import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

GENOMES = { "mm10" : "/users/kcochran/genomes/mm10_no_alt_analysis_set_ENCODE.fasta",
            "hg38" : "/users/kcochran/genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta" }

ROOT = "/users/kcochran/projects/cs197_cross_species_domain_adaptation/"
DATA_DIR = ROOT + "data/"

SPECIES = ["mouse", "human"]

TFS = ["CTCF", "CEBPA", "HNF4A", "RXRA"]


import gzip
from collections import defaultdict
import random
import numpy as np
from pyfaidx import Fasta
from torch.utils.data import Dataset
import pyBigWig


def expand_window(start, end, target_len):
    midpoint = (start + end) / 2
    if not midpoint.is_integer() and target_len % 2 == 0:
        midpoint = midpoint - 0.5
    if midpoint.is_integer() and target_len % 2 != 0:
        midpoint = midpoint - 0.5
    new_start = midpoint - target_len / 2
    new_end = midpoint + target_len / 2
    
    assert new_start.is_integer(), new_start
    assert new_end.is_integer(), new_end
    assert new_start >= 0
    assert new_end - new_start == target_len, (new_end, new_start, target_len)
    
    return int(new_start), int(new_end)


class Generator(Dataset):
    letter_dict = {
        'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],
        'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],
        'T':[0,0,0,1],'N':[0,0,0,0]}

    def __init__(self, species, tf, train_val_test,
                 seq_len = 2114, profile_len = 1000, return_labels = True):
        
        assert train_val_test in ["train", "val", "test"]
        ## note: kelly will give you these files, but they are basically the same as normal peak files
        if train_val_test == "train":
            self.peakfile = PEAKS_DIR + species + "/" + tf + "/filtered_peaks_chr3toY.bed"
        elif train_val_test == "val":
            self.peakfile = PEAKS_DIR + species + "/" + tf + "/filtered_peaks_chr1.bed"
        else:
            self.peakfile = PEAKS_DIR + species + "/" + tf + "/filtered_peaks_chr2.bed"
            
        self.pos_bw = BIGWIGS_DIR + species + "/" + tf + "/final.pos.bigWig"
        self.neg_bw = BIGWIGS_DIR + species + "/" + tf + "/final.neg.bigWig"
        self.prof_len = profile_len
        self.max_jitter = 0
        self.return_labels = return_labels
        
        self.genome_file = GENOMES[species]
        self.seq_len = seq_len

        self.set_len()
        self.coords = self.get_coords()
        self.seqs_onehot = self.convert(self.coords)
        self.profiles, self.logcounts = self.get_profiles_and_logcounts(self.coords)


    def __len__(self):
        return self.len
    
    
    def set_len(self):
        with open(self.peakfile) as f:
            self.len = sum([1 for _ in f])


    def get_coords(self):
        with open(self.peakfile) as posf:
            coords_tmp = [line.split()[:3] for line in posf]  # expecting bed file format
        
        coords = []
        for coord in coords_tmp:
            chrom, start, end = coord[0], int(coord[1]), int(coord[2])
            window_start, window_end = expand_window(start, end,
                                                     self.seq_len + 2 * self.max_jitter)
            coords.append((coord[0], window_start, window_end))  # no strand consideration
        return coords
            

    def get_profiles_and_logcounts(self, coords):
        profiles = []
        logcounts = []

        with pyBigWig.open(self.pos_bw) as pos_bw_reader:
            with pyBigWig.open(self.neg_bw) as neg_bw_reader:
                for chrom, start, end in coords:
                    # need to trim the profile length to match model output size
                    # this is smaller than the input size bc of the receptive field
                    # and deconv layer kernel width
                    prof_start, prof_end = expand_window(start, end,
                                                 self.prof_len + 2 * self.max_jitter)
                    
                    pos_profile = np.array(pos_bw_reader.values(chrom, prof_start, prof_end))
                    pos_profile[np.isnan(pos_profile)] = 0
                    neg_profile = np.array(neg_bw_reader.values(chrom, prof_start, prof_end))
                    neg_profile[np.isnan(neg_profile)] = 0
                    profile = np.array([pos_profile, neg_profile])
                    
                    pos_logcount = np.log(np.sum(pos_profile) + 1)
                    neg_logcount = np.log(np.sum(neg_profile) + 1)
                    logcount = np.array([pos_logcount, neg_logcount])

                    profiles.append(profile)
                    logcounts.append(logcount)
                    
        profiles = np.array(profiles)
        logcounts = np.array(logcounts)
        return profiles, logcounts
                

    def convert(self, coords):
        seqs_onehot = []
        with Fasta(self.genome_file) as converter:
            for coord in coords:
                chrom, start, stop = coord
                assert chrom in converter
                seq = converter[chrom][start:stop].seq
                seq_onehot = np.array([self.letter_dict.get(x,[0,0,0,0]) for x in seq])
                seqs_onehot.append(seq_onehot)

        seqs_onehot = np.array(seqs_onehot)
        return seqs_onehot


    def __getitem__(self, batch_index):	
        # get coordinates
        onehot = self.seqs_onehot[batch_index]
        assert onehot.shape[0] > 0, onehot.shape

        onehot = torch.tensor(onehot, dtype=torch.float).permute(1, 0)
        
        if not self.return_labels:
            return onehot
        else:
            # get profiles and logcounts for the two strands
            profiles = self.profiles[batch_index]
            logcounts = self.logcounts[batch_index]

            profiles = torch.tensor(profiles, dtype=torch.float)
            logcounts = torch.tensor(logcounts, dtype=torch.float)
            return onehot, profiles, logcounts


import torch
from attr_prior_utils import *
from torch.utils.data import DataLoader


def MLLLoss(logps, true_counts):
    """ Adapted from Alex. - Jacob
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1) 
    return -torch.mean(log_fact_sum - log_prod_fact + log_prod_exp)


def trim_profile_by_len(prof, true_prof_len, add_batch_axis = False):
    if len(prof.shape) == 3:
        midpoint = prof.shape[2] / 2
        return prof[:, :, int(midpoint - true_prof_len / 2) : int(midpoint + true_prof_len / 2)]
    
    if len(prof.shape) == 2:
        midpoint = prof.shape[1] / 2
        return prof[:, int(midpoint - true_prof_len / 2) : int(midpoint + true_prof_len / 2)]
    else:
        midpoint = prof.shape[0] / 2
        if add_batch_axis:
            return prof[None, int(midpoint - true_prof_len / 2) : int(midpoint + true_prof_len / 2)]
        else:
            return prof[int(midpoint - true_prof_len / 2) : int(midpoint + true_prof_len / 2)]


# modified from Jacob

class BPNetModel(torch.nn.Module):
    def __init__(self, n_filters=64,
                 n_layers=6,
                 input_seq_len=2114, output_prof_len=1000):
        super(BPNetModel, self).__init__()
        self.n_layers = n_layers
        self.input_seq_len = input_seq_len
        self.output_prof_len = output_prof_len
        
        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21)
        self.rconvs = torch.nn.ModuleList([
        torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, dilation=2**i) for i in range(1, self.n_layers+1)])
        self.penultimate_conv = torch.nn.Conv1d(in_channels = n_filters, out_channels = 2, kernel_size=75)
        self.final_conv = torch.nn.Conv1d(in_channels = 2, out_channels = 2, kernel_size=1, groups=1)
        self.relu = torch.nn.ReLU()
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        
        # determining output len of last dilated layer
        #last_out_size = self.input_seq_len
        #for i in range(n_layers + 1):
        #    if i == 0:
        #        last_out_size = last_out_size - (21 - 1)
        #    else:
        #        last_out_size = last_out_size - (2**i * (3 - 1))
        #    print(last_out_size)
        
        self.pool = torch.nn.AvgPool1d(self.output_prof_len)
        self.linear = torch.nn.Linear(in_features = n_filters, out_features = 2)
        self.counts_conv = torch.nn.Conv1d(in_channels = 2, out_channels = 2, kernel_size=1, groups=1)
        
        # store performance metrics
        self.train_profile_losses_by_epoch = []
        self.train_counts_losses_by_epoch = []
        self.val_profile_losses_by_epoch = []
        self.val_counts_losses_by_epoch = []
        self.val_counts_corr_by_epoch = []
        self.target_val_profile_losses_by_epoch = []
        self.target_val_counts_losses_by_epoch = []
        
        # for early stopping
        self.best_state_for_profiles = self.state_dict()
        self.best_profile_metric = float("inf")
        self.target_profile_metric = float("inf")
        
        
    def forward(self, X):
        X = self.relu(self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.relu(self.rconvs[i](X))
            X = torch.add(X, X_conv)  # maybe don't add the final time?

        y_profile = self.penultimate_conv(X)
        y_profile = self.final_conv(y_profile)
        y_profile = y_profile.squeeze()
        
        # global average pooling
        X = trim_profile_by_len(X, self.output_prof_len)
        y_logcounts = self.pool(X)[:, :, 0]
        y_logcounts = self.linear(y_logcounts)
        y_logcounts = y_logcounts[:, :, None]
        y_logcounts = self.counts_conv(y_logcounts)
        
        return y_profile, y_logcounts

    
    def fit(self, train_data_loader, optimizer,
            source_val_data_loader, target_val_data_loader,
        max_epochs=30, counts_weight = 50, verbose = True):
        
        torch.backends.cudnn.enabled = True
        
        if verbose:
            print("Epoch\tTrain_Prof\tTrain_Counts\tVal_Prof\tVal_Counts\tTarget_Val_Prof\tTarget_Val_Counts")
        
        for epoch in range(max_epochs):
            torch.cuda.empty_cache()
            
            # training loop
            torch.set_grad_enabled(True)
            self.train()
            
            train_profile_losses = []
            train_prior_losses = []
            train_logcounts_losses = []
            for seq_onehot_batch, true_profile_batch, true_logcounts_batch in train_data_loader:
                optimizer.zero_grad()
                seq_onehot_batch = seq_onehot_batch.cuda()
                
                
                # Attribution prior stuff
                ##########
                
                seq_onehot_batch.requires_grad = True  # Reset gradient required
                
                _, pred_logcounts = self(seq_onehot_batch)
                
                for ex_idx in range(seq_onehot_batch.shape[0]):   # along batch axis
                    seq_onehot = seq_onehot_batch[ex_idx:ex_idx+1]
                    pred_logits, _ = self(seq_onehot)
                    pred_logits_trimmed = trim_profile_by_len(pred_logits, self.output_prof_len)
                    pred_profile = self.logsoftmax(pred_logits_trimmed)

                    norm_pred_logits = pred_logits_trimmed - torch.mean(pred_logits_trimmed, dim=-1, keepdim=True)
                    norm_pred_logits = norm_pred_logits * pred_profile
                    
                    # Compute the gradients of the output with respect to the input
                    input_grads, = torch.autograd.grad(norm_pred_logits, seq_onehot,
                        grad_outputs=torch.ones(norm_pred_logits.size()).cuda(),
                        retain_graph=True, create_graph=True)
                        # We'll be operating on the gradient itself, so we need to create the graph

                    input_grads = input_grads * seq_onehot  # Gradient * input

                    att_prior_loss = fourier_att_prior_loss(input_grads, freq_limit, limit_softness,
                            att_prior_grad_smooth_sigma)

                    att_prior_loss.backward(retain_graph=True)
                    train_prior_losses.append(att_prior_loss.item())

                    #########

                    true_profile = true_profile_batch[ex_idx].cuda()
                    profile_loss = MLLLoss(pred_profile, true_profile)
                    #print(profile_loss.item())
                    profile_loss.backward(retain_graph=True)  # this bool is needed for second backward call
                    train_profile_losses.append(profile_loss.item())
                
                true_logcounts_batch = true_logcounts_batch.cuda()
                logcounts_loss = torch.nn.MSELoss()(true_logcounts_batch.squeeze(), pred_logcounts.squeeze())
                logcounts_loss = logcounts_loss * counts_weight
                logcounts_loss.backward()
                tmp = logcounts_loss.item()
                train_logcounts_losses.append(tmp)
                
                optimizer.step()
                
                
            # getting validation set performance
            self.eval()
            
            val_profile_losses = []
            val_logcounts_losses = []
            for seq_onehot, true_profile, true_logcounts in source_val_data_loader:
                seq_onehot = seq_onehot.cuda()
                true_profile = true_profile.cuda()
                true_logcounts = true_logcounts.cuda()
                
                pred_logits, pred_logcounts = self(seq_onehot)
                pred_logits_trimmed = trim_profile_by_len(pred_logits, self.output_prof_len)
                pred_profile_trimmed = self.logsoftmax(pred_logits_trimmed)
                profile_loss = MLLLoss(pred_profile_trimmed, true_profile)
                val_profile_losses.append(profile_loss.item())
                
                logcounts_loss = torch.nn.MSELoss()(true_logcounts.squeeze(), pred_logcounts.squeeze())
                logcounts_loss = logcounts_loss * counts_weight
                val_logcounts_losses.append(logcounts_loss.item())
                
            target_val_profile_losses = []
            target_val_logcounts_losses = []
            for seq_onehot, true_profile, true_logcounts in target_val_data_loader:
                seq_onehot = seq_onehot.cuda()
                true_profile = true_profile.cuda()
                true_logcounts = true_logcounts.cuda()
                
                pred_logits, pred_logcounts = self(seq_onehot)
                pred_logits_trimmed = trim_profile_by_len(pred_logits, self.output_prof_len)
                pred_profile_trimmed = self.logsoftmax(pred_logits_trimmed)
                profile_loss = MLLLoss(pred_profile_trimmed, true_profile)
                target_val_profile_losses.append(profile_loss.item())
                
                logcounts_loss = torch.nn.MSELoss()(true_logcounts.squeeze(), pred_logcounts.squeeze())
                logcounts_loss = logcounts_loss * counts_weight
                target_val_logcounts_losses.append(logcounts_loss.item())
                
            # report results of validation set performance
            to_print = [np.mean(train_profile_losses),
                        np.mean(train_logcounts_losses),
                        np.mean(val_profile_losses),
                        np.mean(val_logcounts_losses),
                        np.mean(target_val_profile_losses),
                        np.mean(target_val_logcounts_losses)]
            print(epoch + 1, "\t", "\t\t".join([str(x) for x in to_print]))#, "", corr)
            
            # save train/val losses
            self.train_profile_losses_by_epoch.append(np.mean(train_profile_losses))
            self.train_counts_losses_by_epoch.append(np.mean(train_logcounts_losses))
            self.val_profile_losses_by_epoch.append(np.mean(val_profile_losses))
            self.val_counts_losses_by_epoch.append(np.mean(val_logcounts_losses))
            self.target_val_profile_losses_by_epoch.append(np.mean(target_val_profile_losses))
            self.target_val_counts_losses_by_epoch.append(np.mean(target_val_logcounts_losses))
            #self.val_counts_corr_by_epoch.append(corr)
            
            # for early stopping
            if np.mean(val_profile_losses) < self.best_profile_metric:
                self.best_profile_metric = np.mean(val_profile_losses)
                self.target_profile_metric = np.mean(target_val_profile_losses)
                self.best_state_for_profiles = self.state_dict()


train_species = "mouse"
val_species = "human"
tf = "CTCF"
batch_size = 16
counts_weight = 0
num_filters = 32
num_layers = 6
learning_rate = 0.001


source_val_data_loader = DataLoader(Generator(train_species, tf, "val"),
                                        batch_size = 16, shuffle = False)

target_val_data_loader = DataLoader(Generator(val_species, tf, "val"),
                                        batch_size = 16, shuffle = False)

train_data_loader = DataLoader(Generator(train_species, tf, "train"),
                               batch_size = batch_size, shuffle = True)

model = BPNetModel(n_filters=num_filters, n_layers=num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train!
print("Training...")
                        
model.cuda()
model.fit(train_data_loader, optimizer, source_val_data_loader, target_val_data_loader,
                                 counts_weight = counts_weight)
model.cpu()

model.load_state_dict(model.best_state_for_profiles)

print("Best-model auPRC, source species:", model.best_profile_metric)
print("Best-model auPRC, target species:", model.target_profile_metric)
