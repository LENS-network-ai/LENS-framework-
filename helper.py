import torch
from utils.metrics import ConfusionMatrix  # Ensure you have a metrics module

def collate(batch):
    image = [b['image'] for b in batch]
    label = [b['label'] for b in batch]
    id = [b['id'] for b in batch]
    adj_s = [b['adj_s'] for b in batch]
    return {'image': image, 'label': label, 'id': id, 'adj_s': adj_s}

def preparefeatureLabel(batch_graph, batch_label, batch_adjs, n_features: int = 512):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0
    for i in range(batch_size):
        labels[i] = batch_label[i]
        max_node_num = max(max_node_num, batch_graph[i].shape[0])
    
    masks = torch.zeros(batch_size, max_node_num)
    adjs = torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feat = torch.zeros(batch_size, max_node_num, n_features)
    for i in range(batch_size):
        cur_node_num = batch_graph[i].shape[0]
        batch_node_feat[i, 0:cur_node_num] = batch_graph[i]
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]
        masks[i, 0:cur_node_num] = 1  
    node_feat = batch_node_feat.cuda()
    labels = labels.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()
    return node_feat, labels, adjs, masks

class Trainer(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)
        self.saved_pruned_adjs = {}   # To store pruned adjacencies by WSI ID
        self.original_edges = {}      # To store original edge counts
        
    def get_scores(self):
        return self.metrics.get_scores()
        
    def reset_metrics(self):
        self.metrics.reset()
    
    def train(self, sample, model, n_features: int = 512):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'], n_features=n_features)
        pred, labels, loss, pruned_adj = model.forward(node_feat, labels, adjs, masks)
        
        # Save pruned adjacency and original edge count (using the unpruned 'adjs')
        wsi_id = sample['id'][0]
        self.saved_pruned_adjs[wsi_id] = pruned_adj.cpu().detach()
        self.original_edges[wsi_id] = (adjs > 0).sum().item()
        
        # Detach tensors before converting to numpy
        pred_detached = pred.detach()
        labels_detached = labels.detach()
        if pred_detached.ndim== 1:
           pred_class = pred_detached
        else: 
        # Get predicted class (argmax)
           pred_class = torch.argmax(pred_detached, dim=1)
        
        # Convert to numpy arrays and ensure they're 1D
        pred_numpy = pred_class.cpu().numpy().reshape(-1)
        label_numpy = labels_detached.cpu().numpy().reshape(-1)
        
        # Update metrics with reshaped arrays
        self.metrics.update(pred_numpy, label_numpy)
        
        return pred, labels, loss, pruned_adj

class Evaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)
    
    def get_scores(self):
        return self.metrics.get_scores()
        
    def reset_metrics(self):
        self.metrics.reset()
        
    def eval_test(self, sample, model, graphcam_flag=False, n_features: int = 512):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'], n_features=n_features)
        with torch.no_grad():
            pred, labels, loss, pruned_adj = model.forward(node_feat, labels, adjs, masks)
        if pred.ndim== 1:
           pred_class = pred
        else:   
        # Get predicted class (argmax)
           pred_class = torch.argmax(pred, dim=1)
        
        # Convert to numpy arrays and ensure they're 1D
        pred_numpy = pred_class.cpu().numpy().reshape(-1)
        label_numpy = labels.cpu().numpy().reshape(-1)
        
        # Update metrics with reshaped arrays
        self.metrics.update(pred_numpy, label_numpy)
            
        return pred, labels, loss, pruned_adj
