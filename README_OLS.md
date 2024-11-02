ols method is designed as a plugin module as a criterion.

You can use as following example: 

```python 
#### A Part of Training Progress ####

ols_criterion = OnlineLabelSmoothing(...)

for samples, targets in data_loader: 
    samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)

    outputs = model(samples)
    loss = ols_criterion(outputs, targets) # Your targets can be one-hot or just label.
```