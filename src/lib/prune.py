import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .ablate import AblateGPT
from .modified_sparsegpt import Hall_SparseGPT
from .probe_extractor import ProbeExtractor, get_truthful_qa_pairs, random_probes
from .truth_data import load_quads, select_known_quads, tokenize_quads, ContrastAccumulator

# Probe / contrast key for each pruned linear: the space of that linear's input
# (k/v share q's input, up shares gate's input)
LINEAR_KEY = {'q_proj': 'qkv_in', 'k_proj': 'qkv_in', 'v_proj': 'qkv_in', 'o_proj': 'o_in',
              'gate_proj': 'gate_up_in', 'up_proj': 'gate_up_in', 'down_proj': 'down_in'}
KEY_SOURCE = {'q_proj': 'qkv_in', 'o_proj': 'o_in', 'gate_proj': 'gate_up_in', 'down_proj': 'down_in'}


def _layer_out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def _catch_layer0_inputs(model, layers, batches, dev):
    """Inputs to decoder layer 0 (and that layer's kwargs) for each batch of input_ids."""
    caught, kwargs_list = [], []

    class BatchCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            caught.append(inp.detach())
            kwargs_list.append({
                'attention_mask': kwargs.get('attention_mask'),
                'position_ids': kwargs.get('position_ids'),
                'position_embeddings': kwargs.get('position_embeddings'),
            })
            raise ValueError

    layers[0] = BatchCatcher(layers[0])
    for ids, _ in batches:
        try:
            model(ids.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    return caught, kwargs_list

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = getattr(model, 'hf_device_map', {})["model.embed_tokens"]
    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = getattr(model, 'hf_device_map', {})["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, cache.get('position_embeddings') 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = getattr(model, 'hf_device_map', {})[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, **({'position_embeddings': position_embeddings} if position_embeddings is not None else {}))[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, **({'position_embeddings': position_embeddings} if position_embeddings is not None else {}))[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        dev = getattr(model, 'hf_device_map', {})["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            dev = getattr(model, 'hf_device_map', {})[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, **({'position_embeddings': position_embeddings} if position_embeddings is not None else {}))[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=args.percdamp, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, **({'position_embeddings': position_embeddings} if position_embeddings is not None else {}))[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_hall(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')

    contrast_mode = args.hall_direction in ("contrast", "contrast_random")
    all_probes = None
    contrast_batches = []
    diagnostics = []
    if args.hall_direction == "random":
        # Control: same penalty strength along a random unit direction
        print(f'Using random probe directions (seed {args.hall_direction_seed})')
        all_probes = random_probes(model, seed=args.hall_direction_seed)
    elif contrast_mode:
        # Contrastive truth subspace from CounterFact 2x2 quads (see lib/truth_data.py)
        quads = load_quads(args.contrast_file)
        quads, stats = select_known_quads(model, tokenizer, quads, args.contrast_nquads, dev,
                                          filter_known=not args.contrast_no_filter)
        contrast_batches, n_used = tokenize_quads(tokenizer, quads, args.contrast_quads_per_batch)
        print(f'Contrast quads: scored {stats["scored"]}, kept {stats["kept"]} known, using {n_used} '
              f'({len(contrast_batches)} batches); rank {args.contrast_rank}, control weight {args.contrast_control_weight}'
              + (f', RANDOM subspace (seed {args.hall_direction_seed})' if args.hall_direction == "contrast_random" else ''))
        if n_used == 0:
            raise ValueError("No contrast quads left after filtering")
    else:
        print('Extracting truthful and hallucinated pairs')
        prompt_pairs = get_truthful_qa_pairs(num_samples=args.probe_nsamples)
        truthful_prompts = [p[0] for p in prompt_pairs]
        hallucinated_prompts = [p[1] for p in prompt_pairs]

        print('Extracting Truthfulness Probes...')
        extractor = ProbeExtractor(model, tokenizer, device=dev)
        all_probes = extractor.extract_all_layers_probes(truthful_prompts, hallucinated_prompts)

    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        dev = getattr(model, 'hf_device_map', {})["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    # Contrast statements travel through the same partially-pruned prefix as the calibration data,
    # so the truth subspace at layer i is measured in the model whose layer i is being pruned.
    if contrast_mode:
        cinps, ckwargs = _catch_layer0_inputs(model, layers, contrast_batches, dev)
        torch.cuda.empty_cache()

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            dev = getattr(model, 'hf_device_map', {})[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = Hall_SparseGPT(subset[name])

        penalties = {}
        if contrast_mode:
            accs, cur, chandles = {}, {}, []
            tail = getattr(args, 'contrast_tail', 1)

            def acc_hook(key):
                def tmp(_, inp, out):
                    x = inp[0]
                    last = cur['last'].to(x.device)
                    rows = torch.arange(x.shape[0], device=x.device)
                    # offset m counts back from the last token; m > 0 reaches the object tokens
                    # (never the BOS token at position 0)
                    for m in range(tail):
                        accs[key].add(x[rows, (last - m).clamp(min=1)])
                return tmp

            for name in subset:
                src = name.split('.')[-1]
                if src in KEY_SOURCE:
                    key = KEY_SOURCE[src]
                    accs[key] = ContrastAccumulator(subset[name].in_features, subset[name].weight.device)
                    chandles.append(subset[name].register_forward_hook(acc_hook(key)))
            for b, (_, last) in enumerate(contrast_batches):
                cur['last'] = last
                layer(cinps[b], **ckwargs[b])
            for h in chandles:
                h.remove()

            for k_idx, (key, acc) in enumerate(sorted(accs.items())):
                seed = None
                if args.hall_direction == "contrast_random":
                    seed = args.hall_direction_seed * 1_000_003 + i * 10 + k_idx
                P, info = acc.penalty(rank=args.contrast_rank, control_weight=args.contrast_control_weight, random_seed=seed)
                acc.free()
                penalties[key] = P
                info.update(layer=i, key=key)
                diagnostics.append(info)
                print(f'layer {i} {key}: ' + ', '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in info.items() if k not in ('layer', 'key')))
            del accs
            torch.cuda.empty_cache()

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            
            # map the linear layer names to probe keys (each probe is taken from that layer's input)
            probe_key = None
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                probe_key = 'qkv_in'
            elif 'o_proj' in name:
                probe_key = 'o_in'
            elif 'gate_proj' in name or 'up_proj' in name:
                probe_key = 'gate_up_in'
            elif 'down_proj' in name:
                probe_key = 'down_in'

            if contrast_mode:
                P = penalties.get(probe_key)
                if P is not None:
                    coverage = gpts[name].add_subspace_penalty(P, alpha=args.hall_alpha, scale_mode=args.hall_alpha_scale)
                    print(f'Adding {probe_key} contrast-subspace penalty (tr(PH)/mean diag H = {coverage:.3g})')
                    diagnostics.append({'layer': i, 'linear': name, 'coverage': coverage})
                else:
                    print(f'Warning: no contrast signal for {name}, pruning standardly.')
                print('Pruning ...')
                gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=args.percdamp, blocksize=128)
                gpts[name].free()
                continue

            # fetch the specific probe for this layer and key
            probe_vector = all_probes[i].get(probe_key)

            # inject penalty into hessian
            if probe_vector is not None:
                print(f'Adding penalty using {probe_key} probe...')
                gpts[name].add_truthfulness_penalty(probe_vector, alpha=args.hall_alpha, scale_mode=args.hall_alpha_scale)
            else:
                print(f'Warning: No probe found for {name}, pruning standardly.')

            print('Pruning ...')
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=args.percdamp, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]

        if contrast_mode:
            # propagate the contrast statements through the now-pruned layer
            cinps = [_layer_out(layer(cinps[b], **ckwargs[b])) for b in range(len(cinps))]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if contrast_mode and args.save:
        import json, os
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, 'contrast_diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f, indent=1)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        dev = getattr(model, 'hf_device_map', {})["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            dev = getattr(model, 'hf_device_map', {})[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, **({'position_embeddings': position_embeddings} if position_embeddings is not None else {}))[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, **({'position_embeddings': position_embeddings} if position_embeddings is not None else {}))[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()