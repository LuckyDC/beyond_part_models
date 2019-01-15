from ignite.engine import Engine


def create_reid_engine(model, optimizer, criterion, device=None, non_blocking=False):
    if device is not None:
        model.to(device)

    def _process_func(engine, batch):
        model.train()

        inputs, label, _ = batch

        if device is not None:
            inputs = inputs.to(device, non_blocking=non_blocking)
            label = label.to(device, non_blocking=non_blocking)

        pred = model(inputs)
        num_parts = pred.size(0) // label.size(0)
        label = label.repeat(num_parts)

        optimizer.zero_grad()
        loss = criterion(pred, label) * num_parts
        loss.backward()
        optimizer.step()

        return pred.data.cpu(), label.data.cpu(), loss.item()

    return Engine(_process_func)
