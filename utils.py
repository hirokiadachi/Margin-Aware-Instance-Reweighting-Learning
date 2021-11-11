import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_probabilistic_margin(logits, targets, num_classes):
    """
    Calculating probabilistic margin
      - logits: model output for natural/adversarial samples
      - targets: ground truth labels corresponed input samples
      - num_classes: the number of classes
    """
    probs = torch.softmax(logits, dim=1)
    onehot = torch.eye(num_classes)[targets].cuda()
    true = torch.masked_select(probs, onehot.bool())
    top2 = torch.topk(probs*(1 - onehot), k=1, dim=1, largest=True)[0]
    return true - top2

def calc_loss(xent, targets, logits_adv, logits_nat=None, weight=None, kl_lambda=None, training_type='at'):
    """
    Calculating the classification loss
      - xent: cross entropy loss function
      - targets: ground truth labels corresponed input samples
      - logits_adv: model output for adversarial samples
      - logits_nat: model output for natural samples
      - weight: weight values for each samples. weight values are one for all samples during burn-in period.
      - kl_lambda: coefficient to scale the kl divergence
      - training_type: training type (at, trades, or mart)
    """
    probs_nat = torch.softmax(logits_nat, dim=1)
    probs_adv = torch.softmax(logits_adv, dim=1)
    batch = targets.size(0)
    kldiv = nn.KLDivLoss(reduction='none')
    onehot = torch.eye(probs_adv.size(1))[targets].cuda()
    if training_type == 'at':
        xent_loss = -torch.log(torch.masked_select(probs_adv, onehot.bool()))
        kl_loss = None
        loss = torch.sum(weight * xent_loss) / batch
    elif training_type == 'trades':
        xent_loss = -torch.log(torch.masked_select(probs_nat, onehot.bool()))
        kl_loss = kldiv(torch.log(probs_adv), probs_nat).sum(dim=1)
        loss = (torch.sum(xent_loss) + kl_lambda * torch.sum(weight * kl_loss)) / batch
    elif training_type == 'mart':
        gt_probs = torch.masked_select(probs_adv, onehot.bool())
        xent_loss = -torch.log(gt_probs)
        kl_loss = kldiv(torch.log(probs_adv), probs_nat).sum(dim=1)
        loss = (torch.sum(xent_loss) + kl_lambda * torch.sum(weight * kl_loss) * (1 - gt_probs))/batch

    return loss

def pgd_attk(model, criterion, inputs, targets, alpha, epsilon, num_iters, training_type, lower=0, upper=1):
    noise = torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon).cuda()
    x = torch.clamp(inputs.detach()+noise.detach(), min=lower, max=upper)
    if training_type == 'trades':
        logits_nat = model(inputs)
        criterion = nn.KLDivLoss(size_average=False)
    for _ in range(num_iters):
        x.requires_grad_()
        logits = model(x)
        
        if training_type == 'trades':
            loss = criterion(F.log_softmax(logits, dim=1),
                             F.softmax(logits_nat, dim=1))
        else:
            loss = xent(logits, targets)
        grads = torch.autograd.grad(loss, x)[0]
        
        x = x.data.detach() + alpha*torch.sign(grads).detach()
        x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
        x = torch.clamp(x, min=lower, max=upper)
    
    return x