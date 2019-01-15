import logging

from ignite.handlers import Timer
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events
from ignite.metrics import Accuracy

from engine.create_reid_engine import create_reid_engine
from engine.scalar_metric import ScalarMetric


def create_train_engine(model, optimizer, criterion, lr_scheduler=None, logger=None, device=None, non_blocking=False,
                        log_period=10, save_interval=10, save_dir="checkpoints", prefix="model"):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    trainer = create_reid_engine(model, optimizer, criterion, device, non_blocking)

    handler = ModelCheckpoint(save_dir, prefix, save_interval=save_interval, n_saved=1, create_dir=True,
                              save_as_state_dict=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    timer = Timer(average=True)

    acc = Accuracy()
    ce = ScalarMetric()

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
        acc.reset()
        ce.reset()
        timer.reset()

        engine.state.iteration = 0

        if lr_scheduler is not None:
            lr_scheduler.step()
            logger.info("Current learning rate: %.4f." % lr_scheduler.get_lr()[0])

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()

        iteration = engine.state.iteration

        acc.update(engine.state.output[:2])
        ce.update(engine.state.output[2])

        if iteration % log_period == 0:
            epoch = engine.state.epoch
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()

            msg = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\t" % (epoch, iteration, speed)
            msg += "acc: %f\tce: %f" % (acc.compute(), ce.compute())
            logger.info(msg)

            acc.reset()
            ce.reset()
            timer.reset()

    return trainer
