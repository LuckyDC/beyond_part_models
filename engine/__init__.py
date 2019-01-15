import torch
import logging

from ignite.handlers import Timer
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events
from ignite.metrics import Accuracy

from engine.create_reid_engine import create_train_engine
from engine.create_reid_engine import create_eval_engine
from engine.scalar_metric import ScalarMetric

from utils.evaluation import eval_feature


def get_trainer(model, optimizer, criterion, lr_scheduler=None, logger=None, device=None, non_blocking=False,
                log_period=10, save_interval=10, save_dir="checkpoints", prefix="model", query_loader=None,
                gallery_loader=None, validate_interval=None):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    # trainer
    trainer = create_train_engine(model, optimizer, criterion, device, non_blocking)

    # checkpoint handler
    handler = ModelCheckpoint(save_dir, prefix, save_interval=save_interval, n_saved=1, create_dir=True,
                              save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    # metric
    timer = Timer(average=True)

    acc = Accuracy()
    ce = ScalarMetric()

    # evaluator
    evaluator = None
    if not type(validate_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if validate_interval > 0 and query_loader and gallery_loader:
        evaluator = create_eval_engine(model, device, non_blocking)

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
        acc.reset()
        ce.reset()
        timer.reset()

        engine.state.iteration = 0

        if lr_scheduler is not None:
            lr_scheduler.step()
            logger.info("Current learning rate: %.4f." % lr_scheduler.get_lr()[0])

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):

        if evaluator and engine.state.epoch % validate_interval == 0:
            # extract query feature
            evaluator.run(query_loader)

            q_feats = torch.cat(evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(evaluator.state.id_list, dim=0)
            q_cam = torch.cat(evaluator.state.cam_list, dim=0)

            # extract gallery feature
            evaluator.run(gallery_loader)

            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0)
            g_cam = torch.cat(evaluator.state.cam_list, dim=0)

            print(g_feats.shape)

            eval_feature(q_feats, g_feats, q_ids, q_cam, g_ids, g_cam)

        torch.cuda.empty_cache()

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
