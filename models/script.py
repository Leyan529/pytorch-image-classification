import importlib
import threading
from tqdm import tqdm
import torch
from utils.helpers import get_lr
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import os

def fit_model(model_train, model, crietion, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, opt):
    Epoch, cuda, fp16, scaler, ema, local_rank = opt.end_epoch, opt.Cuda, opt.fp16, opt.scaler, opt.ema, opt.local_rank
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0
    val_accuracy    = 0

    if local_rank == 0:
        print('Start Train')        
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value  = crietion(outputs, targets)
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value  = crietion(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss_value.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train_eval = model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()

            outputs     = model_train_eval(images)
            loss_value  = crietion(outputs, targets)
            
            val_loss    += loss_value.item()
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy    += accuracy.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.epoch_loss(total_loss/(epoch_step+1), val_loss/(epoch_step_val+1), epoch+1)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        torch.save(save_state_dict, '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (loss_history.log_dir, epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val))
        # best epoch weights
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))
        # last epoch weights
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))
        
    


def get_fit_func(opt):
    return fit_model