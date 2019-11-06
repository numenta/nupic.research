import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import random
from os.path import expanduser
import argparse
import torch
from torch.utils.data import DataLoader
from rsm_samplers import MNISTSequenceSampler
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss, MSELoss
import rsm
import numpy as np
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import time
import rsm_samplers
import rsm
import util
import baseline_models

BSZ = 300
PAGI9 = [[2, 4, 0, 7, 8, 1, 6, 1, 8], [2, 7, 4, 9, 5, 9, 3, 1, 0], [5, 7, 3, 4, 1, 3, 1, 6, 4], [1, 3, 7, 5, 2, 5, 5, 3, 4], [
    2, 9, 1, 9, 2, 8, 3, 2, 7], [1, 2, 6, 4, 8, 3, 5, 0, 3], [3, 8, 0, 5, 6, 4, 1, 3, 9], [4, 7, 5, 3, 7, 6, 7, 2, 4]]
PLOT_INT = 100


predictor = rsm.RSMPredictor(
                d_in=28**2,
                d_out=10,
                hidden_size=1200
            )


class BPTTTrainer():
    def __init__(self, model, loader, k1=1, k2=30, predictor=None, 
                 bsz=BSZ, mbs=100, 
                 use_optimizer=True, lr=1e-5,
                 link_cell_state=False):
        self.k1 = k1
        self.k2 = k2
        self.lr = lr
        self.model = model
        self.loader = loader
        self.predictor = predictor
        self.predictor_loss = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.pred_optimizer = torch.optim.Adam(params=self.predictor.parameters(), lr=lr)
        self.bsz = bsz
        self.loss_module = torch.nn.MSELoss()
        self.retain_graph = self.k1 < self.k2
        self.link_cell_state = link_cell_state
        self.epoch = 0
        self.mbs = mbs
        self.total_mbs = 0
        self.use_optimizer = use_optimizer

        # Predictor counts
        self.total_samples = 0
        self.correct_samples = 0
        self.total_loss = 0.0
        
        if torch.cuda.is_available():
            print("setup: Using cuda")
            self.device = torch.device("cuda")
            seed = random.randint(0, 10000)
            torch.cuda.manual_seed(seed)
        else:
            print("setup: Using cpu")
            self.device = torch.device("cpu")
            
        self.model.to(self.device)
        self.predictor.to(self.device)

    def one_step_module(self, inp, hidden):
        out, new_hidden = self.model(inp, hidden)
        return (out, new_hidden)
    
    def _repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self._repackage_hidden(v) for v in h)
        
    def predict(self, i, out, pred_tgts):
        self.pred_optimizer.zero_grad()
        pred_distr, pred_logits = self.predictor(out.detach())
        loss = self.predictor_loss(pred_logits, pred_tgts)
        loss.backward()
        self.pred_optimizer.step()
        _, class_predictions = torch.max(pred_distr, 1)
        correct_arr = class_predictions == pred_tgts
        self.total_samples += pred_tgts.size(0)
        self.correct_samples += correct_arr.sum().item()        
        interval = PLOT_INT if (not self.mbs or self.mbs > 10000) else self.mbs
        if (i+1) % interval == 0:
            train_acc = 100.*self.correct_samples/self.total_samples
            print(i, "train acc: %.3f%%, loss: %.3f" % (train_acc, self.total_loss))
            self.writer.add_scalar('train_acc', train_acc, self.total_mbs)
            self.writer.add_scalar('train_loss', self.total_loss, self.total_mbs) 
            self.total_samples = 0
            self.correct_samples = 0
            self.total_loss = 0.0

    def init(self):
        outputs = []
        targets = []
        states = [(None, self.model.init_hidden(self.bsz))]
        return (outputs, targets, states)

    def run(self, epochs=0, filename=None):
        write_path = expanduser("~/nta/results/SMNIST_LSTM/%s" % filename)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        self.writer = SummaryWriter(write_path)
        if epochs:
            self.loader.batch_sampler.max_batches = self.mbs
            while self.epoch < epochs:
                self.train()
                self.epoch += 1
        else:
            # Continuous just run up to mbs
            self.loader.batch_sampler.max_batches = self.mbs
            self.train()
        self.writer.close()

    def train(self):
        self.total_samples = 0
        self.correct_samples = 0
        self.total_loss = 0.0
        
        outputs, targets, states = self.init()

        for i, (inp, target, pred_tgts, input_labels) in enumerate(self.loader):
            inp = inp.to(self.device)
            target = target.to(self.device)
            pred_tgts = pred_tgts.to(self.device)

            batch_loss = 0.0
            state = self._repackage_hidden(states[-1][1])
            for h in state:
                h.requires_grad=True
            output, new_state = self.one_step_module(inp, state)

            outputs.append(output)
            targets.append(target)
            while len(outputs) > self.k1:
                # Delete stuff that is too old
                del outputs[0]
                del targets[0]

            states.append((state, new_state))
            while len(states) > self.k2:
                # Delete stuff that is too old
                del states[0]
                
            if (i+1)%self.k1 == 0:
                self.optimizer.zero_grad()
                
                # backprop last module (keep graph only if they ever overlap)
                start = time.time()
                for j in range(self.k2-1):
                    # print('j', j)
                    if j < self.k1:
                        loss = self.loss_module(outputs[-j-1], targets[-j-1])
                        batch_loss += loss.item()
                        loss.backward(retain_graph=True)

                    # if we get all the way back to the "init_state", stop
                    if states[-j-2][0] is None:
                        break
                    curr_h_grad = states[-j-1][0][0].grad
                    curr_c_grad = states[-j-1][0][1].grad                    
                    states[-j-2][1][0].backward(curr_h_grad, retain_graph=self.retain_graph)
                    if self.link_cell_state:
                        states[-j-2][1][1].backward(curr_c_grad, retain_graph=self.retain_graph)                    
                # print("opt step, batch loss: %.3f" % batch_loss)
                if self.use_optimizer:
                    self.optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                    for p in self.model.parameters():
                        p.data.add_(-self.lr, p.grad.data)

            self.total_loss += batch_loss
            self.total_mbs += 1
            
            self.predict(i, output, pred_tgts)
            
            if i > self.mbs:
                print("Done")
                return

if __name__ == "__main__":
    optparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    optparser.add_argument(
        "-m",
        "--mbs",
        dest="mbs",
        type=int,
        default=100,
        help="# of minibatches",
    )
    optparser.add_argument(
        "-t",
        "--total",
        dest="total",
        type=int,
        default=350000,
        help="Total batches"
    )    
    optparser.add_argument(
        "-k",
        "--k2",
        dest="k2",
        type=int,
        default=30,
        help="k2"
    )
    optparser.add_argument(
        "-f",
        "--fixed",
        dest="fixed",
        action="store_true",
        default=False,
        help="Fixed MNIST digit"
    )    
    optparser.add_argument(
        "-n",
        "--noise",
        dest="noise",
        action="store_true",
        default=False,
        help="Noise buffer between subsequences"
    ) 
    optparser.add_argument(
        "-c",
        "--clip",
        dest="clip",
        action="store_true",
        default=False,
        help="Gradient clipping"
    )   
    optparser.add_argument(
        "-l",
        "--lr",
        dest="lr",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    optparser.add_argument(
        "-d",
        "--nhid",
        dest="nhid",
        type=int,
        default=450,
        help="Hidden dim"
    )    
    optparser.add_argument(
        "-s",
        "--cs",
        dest="cs",
        action="store_true",
        default=False,
        help="Reconnect cell state"
    )
    opts = optparser.parse_args()    

    model = baseline_models.LSTMModel(
                    vocab_size=10,
                    nhid=opts.nhid,
                    d_in=28**2,
                    d_out=28**2
                )
    print("LSTM parameters: ", util.count_parameters(model))


    dataset = rsm_samplers.MNISTBufferedDataset(expanduser("~/nta/datasets"), download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))
                                                ]),)

    sampler = rsm_samplers.MNISTSequenceSampler(dataset, 
                                                sequences=PAGI9, 
                                                batch_size=BSZ,
                                                noise_buffer=opts.noise,
                                                random_mnist_images=not opts.fixed)

    loader = DataLoader(dataset,
                 batch_sampler=sampler,
                 collate_fn=rsm_samplers.pred_sequence_collate)    

    continuous = opts.total == opts.mbs
    if continuous:
        epochs = 0
    else:
        epochs = int(opts.total / opts.mbs)
    trainer = BPTTTrainer(model, loader, predictor=predictor, k1=1, k2=opts.k2, mbs=opts.mbs, use_optimizer=not opts.clip, lr=opts.lr, link_cell_state=opts.cs)
    filename = "%d_mbs-%d_epochs-%d_k2-%d_fixed-%s_noise-%s_clip-%s_cs-%s_lr-%s_nhid-%d" % (int(time.time()), opts.mbs, epochs, opts.k2, opts.fixed, opts.noise, opts.clip, opts.cs, opts.lr, opts.nhid)
    trainer.run(epochs=epochs, filename=filename)

