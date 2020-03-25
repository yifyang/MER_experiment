### This is a copy of GEM from https://github.com/facebookresearch/GradientEpisodicMemory.
### In order to ensure complete reproducability, we do not change the file and treat it as a baseline.

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import quadprog

from .common import MLP, ResNet18


# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def store_layer_grad(layers, grads_layer, grad_dims_layer, tid, is_cifar):
    """
        This stores parameter gradients at each layers of past tasks.
        layers: layers in neural network
        grads_layer: gradients at each layer
        grad_dims_layer: list with number of parameters per layers
        tid: task id
    """
    if is_cifar:
        layer_num = 0
        for layer in layers:
            grads_layer[layer_num][:, tid].fill_(0.0)
            cnt = 0
            for param in layer.parameters():
                if param.grad is not None:
                    beg = 0 if cnt == 0 else sum(grad_dims_layer[layer_num][:cnt])
                    en = sum(grad_dims_layer[layer_num][:cnt + 1])
                    grads_layer[layer_num][beg: en, tid].copy_(param.grad.data.view(-1))
                cnt += 1
            layer_num += 1
    else:
        layer_num = 0
        for param in layers():
            grads_layer[layer_num][:, tid].fill_(0.0)
            cnt = 0
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims_layer[layer_num][:cnt])
                en = sum(grad_dims_layer[layer_num][:cnt + 1])
                grads_layer[layer_num][beg: en, tid].copy_(param.grad.data.view(-1))
            cnt += 1
            layer_num += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = (args.data_file == 'cifar100.pt' or
                         args.data_file == 'cifar100_20.pt' or
                         args.data_file == 'cifar100_20_o.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.n_tasks = n_tasks
        self.gpu = args.cuda

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            self.n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(self.n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.is_cifar:
            layers = [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
            self.for_layer = layers
            self.grad_dims_layer = []
            layer_num = 0
            self.grads_layer = []
            for layer in layers:
                self.grad_dims_layer.append([])
                for param in layer.parameters():
                    self.grad_dims_layer[layer_num].append(param.data.numel())
                self.grads_layer.append(torch.Tensor(sum(self.grad_dims_layer[layer_num]), n_tasks))
                if args.cuda:
                    self.grads_layer[-1] = self.grads_layer[-1].cuda()
                layer_num += 1
        else:
            self.for_layer = self.parameters
            self.grad_dims_layer = []
            layer_num = 0
            self.grads_layer = []
            for param in self.parameters():
                self.grad_dims_layer.append([param.data.numel()])
                self.grads_layer.append(torch.Tensor(sum(self.grad_dims_layer[layer_num]), n_tasks))
                if args.cuda:
                    self.grads_layer[-1] = self.grads_layer[-1].cuda()
                layer_num += 1

        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        if args.cuda:
            self.cuda()

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                   self.is_cifar)
                ptloss = self.ce(
                    self.forward(
                        Variable(self.memory_data[past_task]),
                        past_task)[:, offset1: offset2],
                    Variable(self.memory_labs[past_task] - offset1))
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)
                store_layer_grad(self.for_layer, self.grads_layer,
                                 self.grad_dims_layer, past_task, self.is_cifar)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        dotp_list = [[0] * (self.n_tasks - 1)]  # record dotp and return
        dotp_layers = [[0] * (self.n_tasks - 1)] * len(self.grads_layer)
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            store_layer_grad(self.for_layer, self.grads_layer,
                             self.grad_dims_layer, t, self.is_cifar)

            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))

            dotp_layers = []
            for layer_num in range(len(self.grads_layer)):
                dotp_layer_temp = []
                for pre_task in indx:
                    dotp_layer_temp.append(
                        torch.cosine_similarity(self.grads_layer[layer_num][:, t],
                                                self.grads_layer[layer_num][:, pre_task],
                                                dim=0).item())
                dotp_layer_temp += [0] * ((self.n_tasks - 1) - len(dotp_layer_temp))
                dotp_layers.append(dotp_layer_temp)
                """
                dotp_layer_temp = torch.mm(self.grads_layer[layer_num][:, t].unsqueeze(0),
                            self.grads_layer[layer_num].index_select(1, indx)).tolist()[0]
                dotp_layer_temp += [0] * ((self.n_tasks-1) - len(dotp_layer_temp))
                dotp_layers.append(dotp_layer_temp)
                """

        self.opt.step()

        return dotp_layers