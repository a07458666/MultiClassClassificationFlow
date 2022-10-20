import numpy as np
from sklearn import metrics
import torch
import datasets
import models
from instrumentation import compute_metrics
import losses
import datetime
import os
from tqdm import tqdm

from logger import *
from flowModule.flow import cnf
from flowModule.utils import standard_normal_logprob, linear_rampup, mix_match
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from flowModule.losses.vicreg import vicreg_loss_func



def create_model(P):
    cond_size = P['feat_dim']
    flow_modules = '8-8-8-8'
    model = cnf(P['num_classes'], flow_modules, cond_size, 1).cuda()
    model = model.cuda()
    return model

def train_flow(model_flow, feats, target):
    # feats = F.normalize(feats, dim=1)
    feats = feats.unsqueeze(1)
    target = target.unsqueeze(1)
    delta_p = torch.zeros(target.shape[0], target.shape[1], 1).cuda()
    # print("feats", feats.size())
    # print("target", target.size())
    # print("delta_p", delta_p.size())
    approx21, delta_log_p2 = model_flow(target, feats, delta_p)
    
    approx2 = standard_normal_logprob(approx21).view(target.size()[0], -1).sum(1, keepdim=True)
    delta_log_p2 = delta_log_p2.view(target.size()[0], target.shape[1], 1).sum(1)
    log_p2 = (approx2 - delta_log_p2)

    loss_flow = -log_p2.mean()
    return loss_flow

def test_flow(P, model_flow, feats, mean = 0, std = 0, sample_n = 1):
    with torch.no_grad():
        batch_size = feats.size()[0]
        feats = feats.unsqueeze(1)
        # feature = F.normalize(feature, dim=1)
        feats = feats.repeat(sample_n, 1, 1)
        input_z = torch.normal(mean = mean, std = std, size=(sample_n * batch_size , P['num_classes'])).unsqueeze(1).cuda()
        delta_p = torch.zeros(input_z.shape[0], input_z.shape[1], 1).cuda()

        approx21, _ = model_flow(input_z, feats, delta_p, reverse=True)

        # probs = torch.clamp(approx21, min=0, max=1)
        probs = approx21.view(sample_n, -1, P['num_classes'])
        probs_mean = torch.mean(probs, dim=0, keepdim=False)
        # probs_mean = F.normalize(probs_mean, dim=1, p=1)
        return probs_mean

def run_train_flow(P):

    dataset = datasets.get_data(P)
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )


    model = models.ImageClassifier(P)
    
    # create flow model
    model_flow = create_model(P)

    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    linear_classifier_params = [param for param in list(model.linear_classifier.parameters()) if param.requires_grad]
    opt_params = [
        {'params': feature_extractor_params, 'lr' : P['lr']},
        {'params': linear_classifier_params, 'lr' : P['lr_mult'] * P['lr']}
    ]
  
    if P['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=P['lr'])
    elif P['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(opt_params, lr=P['lr'], momentum=0.9, weight_decay=0.001)
    
    # flow optimizer
    optimizer_flow = torch.optim.SGD(model_flow.parameters(), lr=P['lr_f'], momentum=0.9, weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, P['num_epochs'], P['lr'] / 100)
    schedulerFlow = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_flow, P['num_epochs'], P['lr_f'] / 100)

    # training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model_flow.to(device)

    bestmap_val = 0
    bestmap_test = 0

    # EMA
    model_ema = ExponentialMovingAverage(model.parameters(), decay=P['decay'])
    model_flow_ema = ExponentialMovingAverage(model_flow.parameters(), decay=P['decay'])

    for epoch in range(1, P['num_epochs']+1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                y_pred_flow_ema = np.zeros((len(dataset[phase]), P['num_classes']))
                y_pred_flow = np.zeros((len(dataset[phase]), P['num_classes']))
                y_pred_ema = np.zeros((len(dataset[phase]), P['num_classes']))
                y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
                y_true = np.zeros((len(dataset[phase]), P['num_classes']))
                batch_stack = 0

            
            with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloader[phase]):
                    # Move data to GPU
                    image = batch['image'].to(device, non_blocking=True)
                    label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                    label_vec_true = batch['label_vec_true'].clone().numpy()
                    idx = batch['idx']

                    # Forward pass
                    optimizer.zero_grad()
                    optimizer_flow.zero_grad()

                    logits, feats = model(image, get_feature = True)
                    
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)
                    
                    if phase == 'train':
                        loss_flow = train_flow(model_flow, feats, label_vec_obs)
                        loss_BCE, correction_idx = losses.compute_batch_loss(logits, label_vec_obs, P)
                        # lossCE = torch.nn.CrossEntropyLoss()
                        # _, labels = label_vec_obs.max(dim=1)
                        # loss_CE = lossCE(logits, labels)
                        loss = loss_BCE + loss_flow
                        # loss = loss_CE
                        # loss = loss_flow
                        # loss = loss_BCE

                        loss.backward()
                        optimizer.step()
                        optimizer_flow.step()

                        model_ema.update()
                        model_flow_ema.update()

                        if P['mod_scheme'] is 'LL-Cp' and correction_idx is not None:
                            dataset[phase].label_matrix_obs[idx[correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0

                    else:
                        # EMA pred
                        with model_ema.average_parameters():
                            with model_flow_ema.average_parameters():
                                # EMA pred resnet
                                logits_ema, feats_ema = model(image, get_feature = True)
                                preds_ema = torch.sigmoid(logits_ema)
                                preds_ema_np = preds_ema.cpu().numpy()

                                # EMA pred flow
                                logits_flow_ema = test_flow(P, model_flow, feats_ema)
                                preds_flow_ema = torch.sigmoid(logits_flow_ema)
                                preds_flow_ema_np = preds_flow_ema.cpu().numpy()

                        # pred renset
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]

                        #pred flow
                        logits_flow = test_flow(P, model_flow, feats)
                        preds_flow = torch.sigmoid(logits_flow)
                        preds_flow_np = preds_flow.cpu().numpy()

                        y_pred_flow_ema[batch_stack : batch_stack+this_batch_size] = preds_flow_ema_np
                        y_pred_flow[batch_stack : batch_stack+this_batch_size] = preds_flow_np
                        y_pred_ema[batch_stack : batch_stack+this_batch_size] = preds_ema_np
                        y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
                        y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
                        batch_stack += this_batch_size

        metrics = compute_metrics(y_pred, y_true)
        metrics_ema = compute_metrics(y_pred_ema, y_true)
        metrics_flow = compute_metrics(y_pred_flow, y_true)
        metrics_flow_ema = compute_metrics(y_pred_flow_ema, y_true)

            
        scheduler.step()
        schedulerFlow.step()
        
        del y_pred_flow_ema
        del y_pred_flow
        del y_pred_ema
        del y_pred
        del y_true

        map_val = metrics['map']
        map_val_flow =  metrics_flow['map']
        P['clean_rate'] -= P['delta_rel']
                
        print(f"Epoch {epoch} : val mAP {map_val:.3f}, flow mAP {map_val_flow:.3f}")
        log_mAP(phase, epoch, map_val, map_val_flow, metrics_ema['map'], metrics_flow_ema['map'])
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            
            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            path = os.path.join(P['save_path'], 'bestmodel.pt')
            path_flow = os.path.join(P['save_path'], 'bestmodel_flow.pt')
            path_ema = os.path.join(P['save_path'], 'bestmodel_ema.pt')
            path_flow_ema = os.path.join(P['save_path'], 'bestmodel_flow_ema.pt')
            torch.save((model.state_dict(), P), path)
            torch.save((model_flow.state_dict(), P), path_flow)
            torch.save((model_ema.state_dict(), P), path_ema)
            torch.save((model_flow_ema.state_dict(), P), path_flow_ema)
        
        elif bestmap_val - map_val > 15:
            print('Early stopped.')
            break
        
    # Test phase

    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)

    model_state_flow, _ = torch.load(path_flow)
    model_flow.load_state_dict(model_state_flow)

    model_state_ema, _ = torch.load(path_ema)
    model_ema.load_state_dict(model_state_ema)

    model_state_flow_ema, _ = torch.load(path_flow_ema)
    model_flow_ema.load_state_dict(model_state_flow_ema)

    phase = 'test'
    
    model.eval()
    model_flow.eval()

    y_pred_flow_ema = np.zeros((len(dataset[phase]), P['num_classes']))
    y_pred_flow = np.zeros((len(dataset[phase]), P['num_classes']))
    y_pred_ema = np.zeros((len(dataset[phase]), P['num_classes']))
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))

    batch_stack = 0
    with torch.set_grad_enabled(phase == 'train'):
        for batch in tqdm(dataloader[phase]):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            idx = batch['idx']

            # Forward pass
            optimizer.zero_grad()
            optimizer_flow.zero_grad()

            # EMA pred
            with model_ema.average_parameters():
                with model_flow_ema.average_parameters():
                    # EMA pred resnet
                    logits_ema, feats_ema = model(image, get_feature = True)
                    if logits_ema.dim() == 1:
                        logits_ema = torch.unsqueeze(logits_ema, 0)
                    preds_ema = torch.sigmoid(logits_ema)
                    preds_ema_np = preds_ema.cpu().numpy()  

                    # EMA pred flow
                    logits_flow_ema = test_flow(P, model_flow, feats_ema)
                    preds_flow_ema = torch.sigmoid(logits_flow_ema)
                    preds_flow_ema_np = preds_flow_ema.cpu().numpy()    

            ## pred
            logits, feats = model(image, get_feature = True)
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)          
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]

            # pred flow
            logits_flow = test_flow(P, model_flow, feats)
            preds_flow = torch.sigmoid(logits_flow)
            preds_flow_np = preds_flow.cpu().numpy()

            y_pred_flow_ema[batch_stack : batch_stack+this_batch_size] = preds_flow_ema_np
            y_pred_ema[batch_stack : batch_stack+this_batch_size] = preds_ema_np
            y_pred_flow[batch_stack : batch_stack+this_batch_size] = preds_flow_np
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    metrics_flow = compute_metrics(y_pred_flow, y_true)
    
    metrics_ema = compute_metrics(y_pred_ema, y_true)
    metrics_flow_ema = compute_metrics(y_pred_flow_ema, y_true)
    
    map_test = metrics['map']
    map_test_flow = metrics_flow['map']
    log_mAP(phase, epoch, map_test, map_test_flow, metrics_ema['map'], metrics_flow_ema['map'])
    print('Training procedure completed!')
    print(f'Test mAP : {map_test:.3f}, mAP Flow : {map_test_flow:.3f} when trained until epoch {bestmap_epoch}')
    