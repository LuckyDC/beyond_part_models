from torch.autograd import no_grad

from ignite.engine import Events
from ignite.engine import Engine


def create_train_engine(model, optimizer, criterion, device=None, non_blocking=False):
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


def create_eval_engine(model, device=None, non_blocking=False):
    if device is not None:
        model.to(device)

    def _process_func(engine, batch):
        model.eval()

        inputs, label, cam_id = batch

        if device is not None:
            inputs = inputs.to(device, non_blocking=non_blocking)

        with no_grad():
            feat = model(inputs)

        return feat.data.cpu(), label.cpu(), cam_id.cpu()

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])

    return engine
