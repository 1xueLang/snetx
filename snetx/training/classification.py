import torch
from torch.cuda import amp
import torch.nn.functional as F

import snetx.snn.algorithm as snnalgo

def train_dvs(net, dataloader, optimizer, criterion, scaler, device, args):
    net.train()
    correct = running_loss = sumup = 0.0
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if scaler == None:
            out = net(inputs)
            loss = criterion(labels, out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        else:
            with amp.autocast():
                out = net(inputs)
                loss = criterion(labels, out)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += out.mean(dim=1).argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))


def train_static(net, dataloader, optimizer, criterion, scaler, device, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = snnalgo.temporal_repeat(inputs, args.T)
            
        if scaler == None:
            out = net(inputs)
            loss = criterion(labels, out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with amp.autocast():
                out = net(inputs)
                loss = criterion(labels, out)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += out.mean(dim=1).argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))

def sync_train_dvs(net, dataloader, optimizer, criterion, scaler, device, args):
    net.train()
    correct = running_loss = sumup = 0.0
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if scaler == None:
            out = net(inputs)
            loss = criterion(labels, out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        else:
            with amp.autocast():
                out = net(inputs)
                loss = criterion(labels, out)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        torch.cuda.synchronize()
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += out.mean(dim=1).argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0 and device == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))


def sync_train_static(net, dataloader, optimizer, criterion, scaler, device, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = snnalgo.temporal_repeat(inputs, args.T)
            
        if scaler == None:
            out = net(inputs)
            loss = criterion(labels, out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with amp.autocast():
                out = net(inputs)
                loss = criterion(labels, out)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        torch.cuda.synchronize()
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += out.mean(dim=1).argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0 and device == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))

@torch.no_grad()
def validate(net, dataloader, device, args, static=True):
    net.eval()
    correct = 0.
    sumup = 0.
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if static:
            inputs = snnalgo.temporal_repeat(inputs, args.T)
            
        out = net(inputs)
            
        sumup += inputs.shape[0]
        correct += out.mean(dim=1).argmax(dim=1).eq(labels).sum().item()
        
    net.train()
    return correct, sumup


def train_top5(net, dataloader, optimizer, criterion, scaler, device, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = snnalgo.temporal_repeat(inputs, args.T)
            
        if scaler == None:
            out = net(inputs)
            loss = criterion(labels, out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with amp.autocast():
                out = net(inputs)
                loss = criterion(labels, out)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        out = out.mean(dim=1)
        
        top5 += (out.topk(5, dim=1, largest=True, sorted=True)[1]).eq(labels.unsqueeze(dim=1)).sum().item()
        correct += out.argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%, Top5: {(top5 / sumup) * 100:.2f}')
        
    return correct, top5, sumup, float(running_loss / len(dataloader))

@torch.no_grad()
def validate_top5(net, dataloader, device, args, static=True):
    net.eval()
    correct = 0.
    sumup = 0.
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if static:
            inputs = snnalgo.temporal_repeat(inputs, args.T)
            
        out = net(inputs).mean(dim=1)
        
        sumup += inputs.shape[0]
        correct += out.argmax(dim=1).eq(labels).sum().item()
        top5 += (out.topk(5, dim=1, largest=True, sorted=True)[1]).eq(labels.unsqueeze(dim=1)).sum().item()
        
    net.train()
    return correct, top5, sumup
