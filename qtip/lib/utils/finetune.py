import torch
from torch import nn


def save_linear(module, path):
    saved_layer = torch.load(path, map_location=torch.device('cpu'))
    saved_layer['SU'] = module.SU.data.to(torch.float16)
    saved_layer['SV'] = (
        module.SV.data.float() /
        saved_layer['Wscale'].float().to(module.SV.data.device)).cpu()
    if module.tlut is not None:
        saved_layer['tlut'] = module.tlut.data.to(torch.float16)
    torch.save(saved_layer, path)

def weighted_mse(out, target, weight_gpd):
    '''
        out : (bs, seq_len, d)
        target : (bs, seq_len, d)
        weight_gpd : (bs, seq_len, g)
    '''
    if weight_gpd is None:
        return nn.MSELoss()(out, target)
    g = weight_gpd.shape[-1]
    # Compute squared error
    sq_err = (out - target) ** 2  # (bs, seq_len, d)

    bs, seq_len, d = out.shape
    weight_gpd = weight_gpd.unsqueeze(-1)  # (bs, seq_len, g, 1)
    weight_gpd = weight_gpd.repeat(1, 1, 1, d // g)  # (bs, seq_len, g, d/g)
    weight_gpd = weight_gpd.view(bs, seq_len, d)  # Reshape to (bs, seq_len, d)

    # Apply weighted mean along the last dimension
    weighted_loss = torch.sum(sq_err * weight_gpd, dim=-1)  / weight_gpd.sum(dim=-1).clamp_min(1e-8) # (bs, seq_len)

    return weighted_loss.mean()

def weighted_mse_test():
    out = torch.rand(2, 3, 4)
    target = torch.rand(2, 3, 4)
    weight_gpd = torch.ones(2, 3, 4) * 1000
    print(weighted_mse(out, target, weight_gpd))
    print(weighted_mse(out, target, None))


def calculate_weighted_mse_loss(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for data in dataloader:
            if len(data) == 2:
                source, target = data
                block_hess = None
            elif len(data) == 3:
                source, target, block_hess = data # source : (bs, seq_len, d), target : (bs, seq_len, d), block_hess : (bs, seq_len, 4)
                block_hess = block_hess.to(device)
            if position_ids is None:
                position_ids = torch.arange(source.shape[1],
                                            device=device).unsqueeze(0)
            target = target.to(device, non_blocking=True)
            total_loss += weighted_mse(layer(source.to(device),
                                             position_ids=position_ids)[0],
                                       target, block_hess)
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()


def calculate_ce_loss_model(model, dataloader, start_dev, in_q, out_q):
    model.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            in_q.put(source)
            output = model(source.to(start_dev))['logits'][:, :-1].contiguous()
            output = output.view(-1, output.shape[-1])
            target = out_q.get().to(output.device)
            target = target.view(-1, target.shape[-1])
            total_loss += nn.CrossEntropyLoss()(output, target)
            ct += 1
    model.train()
    return (total_loss / ct).cpu().item()


if __name__ == "__main__":
    weighted_mse_test()