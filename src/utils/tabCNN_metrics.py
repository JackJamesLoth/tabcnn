#######################################################################
# NOTE: these metrics were taken directly from the TabCNN gitHub repo #
# https://github.com/andywiggins/tab-cnn/blob/master/model/Metrics.py
# They were modifed to work with PyTorch tensors                      #
#######################################################################

import torch

def tab2pitch(tab):
    # Assuming tab shape is (batch_size, num_strings, num_frets)
    batch_size, num_strings, num_frets = tab.shape
    
    # Prepare tensor to hold the batch of pitch vectors
    # New shape will be (batch_size, 44)
    batch_pitch_vector = torch.zeros((batch_size, 45), dtype=torch.float32, device=tab.device)
    
    # Pre-defined string pitches
    string_pitches = torch.tensor([40, 45, 50, 55, 59, 64], device=tab.device)
    
    for batch_idx in range(batch_size):
        for string_num in range(num_strings):
            fret_vector = tab[batch_idx, string_num]
            fret_class = torch.argmax(fret_vector, -1)
            if fret_class > 0:
                pitch_num = fret_class + string_pitches[string_num] - 41
                batch_pitch_vector[batch_idx, pitch_num] = 1.0
    
    return batch_pitch_vector

def tab2bin(tab):
    batch_size, num_strings, num_frets = tab.shape
    tab_arr_batched = torch.zeros((batch_size, num_strings, num_frets), dtype=torch.float32, device=tab.device)
    for i in range(batch_size):
        fret_classes = torch.argmax(tab[i], dim=-1)
        for string_num in range(num_strings):
            fret_class = fret_classes[string_num]
            # If the string is pressed (fret_class > 0), mark the corresponding fret
            if fret_class > 0:
                fret_num = fret_class - 1
                tab_arr_batched[i, string_num, fret_num] = 1.0
    return tab_arr_batched

def pitch_precision(pred, gt):
    pitch_pred = tab2pitch(pred)
    pitch_gt = tab2pitch(gt)
    numerator = torch.sum(pitch_pred * pitch_gt).flatten()
    denominator = torch.sum(pitch_pred).flatten()
    precision = (1.0 * numerator) / denominator
    
    return precision

def pitch_recall(pred, gt):
    pitch_pred = tab2pitch(pred)
    pitch_gt = tab2pitch(gt)
    numerator = torch.sum(torch.mul(pitch_pred, pitch_gt).flatten())
    denominator = torch.sum(pitch_gt.flatten())
    return (1.0 * numerator) / denominator

def pitch_f_measure(pred, gt):
    p = pitch_precision(pred, gt)
    r = pitch_recall(pred, gt)
    # Use torch.where to safely handle division by zero
    f = torch.where((p + r) > 0, (2 * p * r) / (p + r), torch.tensor(0.0))
    return f

def tab_precision(pred, gt):
    # Assuming tab2bin is compatible with PyTorch tensors and returns tensors
    tab_pred = tab2bin(pred)
    tab_gt = tab2bin(gt)
    numerator = torch.sum(tab_pred * tab_gt).float()
    denominator = torch.sum(tab_pred).float()
    
    # Handling division by zero
    if denominator == 0:
        return torch.tensor(0.0)
    return numerator / denominator

def tab_recall(pred, gt):
    tab_pred = tab2bin(pred)
    tab_gt = tab2bin(gt)
    numerator = torch.sum(tab_pred * tab_gt).item()
    denominator = torch.sum(tab_gt).item()
    
    # Return the recall
    return (1.0 * numerator) / denominator if denominator != 0 else 0

def tab_f_measure(pred, gt):
    p = tab_precision(pred, gt)
    r = tab_recall(pred, gt)
    # Use torch.where to avoid division by zero
    f = (2 * p * r) / (p + r)
    return f

def tab_disamb(pred, gt):
    tp = tab_precision(pred, gt)
    pp = pitch_precision(pred, gt)
    return tp / pp