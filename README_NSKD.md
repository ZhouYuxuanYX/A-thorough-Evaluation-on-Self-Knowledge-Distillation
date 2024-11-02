NSKD method is designed as a plugin module as a criterion.

You can use as following example (Now support ViTs only): 

```python 
#### A Part of Training Progress ####

model = ViTFeatureExtractor.warp(model)
model = DDP(model, device_ids=[local_rank])
criterion = USKDLoss(...)

for samples, targets in data_loader: 
    samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)

    outputs, features = model(samples)
    original_model = model.module.unwrap()
    loss = ols_criterion(features, outputs, targets) 
```