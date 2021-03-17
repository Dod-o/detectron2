# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import time
import weakref
from typing import Dict, List, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage


from tensorboardX import SummaryWriter
import os
import cv2
from tqdm import tqdm
import shutil
import xml.dom.minidom
# from .defaults import DefaultPredictor
# from ..data.datasets import get_icard19_dataset


try:
    from azureml.core.run import Run

    aml_run = Run.get_context()
except ImportError:
    aml_run = None

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    Attributes:
        trainer (TrainerBase): A weak reference to the trainer object. Set by the trainer
            when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int
        self.start_iter: int
        self.max_iter: int
        self.storage: EventStorage

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        self.local_rank = self.cfg._rank
        self.logging_steps = self.cfg._logging_steps
        if self.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(
                logdir=os.path.join(self.cfg._output_dir, 'tblogs'))

        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        self.loss_cls, self.logging_loss_cls = 0.0, 0.0
        self.loss_box_reg, self.logging_loss_box_reg = 0.0, 0.0
        self.loss_mask, self.logging_loss_mask = 0.0, 0.0
        self.loss_rpn_cls, self.logging_loss_rpn_cls = 0.0, 0.0
        self.loss_rpn_loc, self.logging_loss_rpn_loc = 0.0, 0.0
        self.loss_mrcnn_total, self.logging_loss_mrcnn_total = 0.0, 0.0

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                iterator = tqdm(range(start_iter, max_iter),
                                desc='Iteration',
                                disable=self.local_rank not in [-1, 0])

                for _, self.iter in enumerate(iterator):
                    self.before_step()
                    self.res = self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                self.logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def user_logging(self):
        self.loss_cls += self.res['loss_cls']
        self.loss_box_reg += self.res['loss_box_reg']
        self.loss_mask += self.res['loss_mask']
        self.loss_rpn_cls += self.res['loss_rpn_cls']
        self.loss_rpn_loc += self.res['loss_rpn_loc']
        self.loss_mrcnn_total += self.res['loss_cls'] + self.res['loss_box_reg'] + self.res['loss_mask'] + self.res[
            'loss_rpn_cls'] + self.res['loss_rpn_loc']

        if self.local_rank in [-1, 0] and self.logging_steps > 0 and self.iter % self.logging_steps == 0:
            self.tb_writer.add_scalar('lr',
                                      self.scheduler.get_last_lr()[0], self.iter)
            self.tb_writer.add_scalar(
                'loss_cls',
                (self.loss_cls - self.logging_loss_cls) / self.logging_steps,
                self.iter,
            )
            self.tb_writer.add_scalar(
                'loss_box_reg',
                (self.loss_box_reg - self.logging_loss_box_reg) / self.logging_steps,
                self.iter,
            )
            self.tb_writer.add_scalar(
                'loss_mask',
                (self.loss_mask - self.logging_loss_mask) / self.logging_steps,
                self.iter,
            )
            self.tb_writer.add_scalar(
                'loss_rpn_cls',
                (self.loss_rpn_cls - self.logging_loss_rpn_cls) / self.logging_steps,
                self.iter,
            )
            self.tb_writer.add_scalar(
                'loss_rpn_loc',
                (self.loss_rpn_loc - self.logging_loss_rpn_loc) / self.logging_steps,
                self.iter,
            )
            self.tb_writer.add_scalar(
                'loss_mrcnn_total',
                (self.loss_mrcnn_total - self.logging_loss_mrcnn_total) / self.logging_steps,
                self.iter,
            )

            if aml_run is not None:
                aml_run.log('lr', self.scheduler.get_last_lr()[0])
                aml_run.log('loss_cls', (self.loss_cls - self.logging_loss_cls) / self.logging_steps)
                aml_run.log('loss_box_reg', (self.loss_box_reg - self.logging_loss_box_reg) / self.logging_steps)
                aml_run.log('loss_mask', (self.loss_mask - self.logging_loss_mask) / self.logging_steps)
                aml_run.log('loss_rpn_cls', (self.loss_rpn_cls - self.logging_loss_rpn_cls) / self.logging_steps)
                aml_run.log('loss_rpn_loc', (self.loss_rpn_loc - self.logging_loss_rpn_loc) / self.logging_steps)
                aml_run.log('loss_mrcnn_total',
                            (self.loss_mrcnn_total - self.logging_loss_mrcnn_total) / self.logging_steps)

            self.logger.info(
                'step: {} | lr: {:.4E} | total_loss: {:6.4f} | table loss: {:6.4f} '.format(
                    self.iter,
                    self.scheduler.get_last_lr()[0],
                    (self.loss_mrcnn_total - self.logging_loss_mrcnn_total) / self.logging_steps,
                    (self.loss_mrcnn_total - self.logging_loss_mrcnn_total) / self.logging_steps,
                ))

            self.logging_loss_cls = self.loss_cls
            self.logging_loss_box_reg = self.loss_box_reg
            self.logging_loss_mask = self.loss_mask
            self.logging_loss_rpn_cls = self.loss_rpn_cls
            self.logging_loss_rpn_loc = self.loss_rpn_loc
            self.logging_loss_mrcnn_total = self.loss_mrcnn_total

    def evaluate(self):
        def calc_table_bbox(table_pred, file_name):
            table_bbox = []

            assert len(table_pred) == len(file_name)

            for i in range(len(table_pred)):
                cur_item = {
                    'filename': file_name[i],
                    'bbox': []
                }

                raw_bbox = table_pred[i]._fields['pred_boxes'].tensor
                res_size = raw_bbox.shape
                for j in range(res_size[0]):
                    cur_bbox = raw_bbox[j].tolist()
                    x0, y0, x1, y1 = map(int, cur_bbox)
                    cur_item['bbox'].append([x0, y0, x0, y1, x1, y1, x1, y0])
                table_bbox.append(cur_item)
            return table_bbox

        def save_table_result(table_result, output_path):
            for i in tqdm(range(len(table_result))):
                pure_filename = table_result[i]['filename']

                doc = xml.dom.minidom.Document()
                root = doc.createElement('document')
                root.setAttribute('filename', table_result[i]['filename'])
                doc.appendChild(root)

                tables = table_result[i]['bbox']
                table_id = 0
                for table in tables:
                    table_id += 1
                    nodeManager = doc.createElement('table')
                    nodeManager.setAttribute('id', str(table_id))
                    bbox_str = '{},{} {},{} {},{} {},{}'.format(table[0], table[1], table[2], table[3], table[4],
                                                                table[5], table[6], table[7])
                    nodeCoords = doc.createElement('Coords')
                    nodeCoords.setAttribute('points', bbox_str)
                    nodeManager.appendChild(nodeCoords)
                    root.appendChild(nodeManager)

                filename = '{}-result.xml'.format(pure_filename)
                fp = open(os.path.join(output_path, filename), 'w')
                doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
                fp.flush()
                fp.close()
                
        
        from .defaults import DefaultPredictor
        from ..data.datasets import get_icard19_dataset
        from ..evaluation import calc_table_score

        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg._output_dir, 'model_{:0=7}.pth'.format(self.iter - 1))

        predictor = DefaultPredictor(self.cfg)
        dataset_dicts = get_icard19_dataset("test", self.cfg._data_dir)

        # all_data = []
        table_preds = []
        image_name = []
        # for d in random.sample(dataset_dicts, 10):
        for d in tqdm(dataset_dicts):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)

            # all_data.append(d)
            table_preds.append(outputs['instances'])
            image_name.append(os.path.basename(d['file_name'])[:-4])

        assert len(table_preds) == len(image_name)

        table_result = calc_table_bbox(table_preds, image_name)
        
        output_path = os.path.join(self.cfg._output_dir, 'checkpoint-{}'.format(self.iter))
        output_path = os.path.join(output_path, 'table_predict')
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        save_table_result(table_result, output_path)
        table_final_score = calc_table_score(output_path)

        resutls = {}
        try:
            resutls.update(table_final_score)
        except:
            resutls.update(
                {
                    'p_six': 0.0, "r_six": 0.0, "f1_six": 0.0,
                    "p_seven": 0.0, "r_seven": 0.0, "f1_seven": 0.0,
                    "p_eight": 0.0, "r_eight": 0.0, "f1_eight": 0.0,
                    "p_nine": 0.0, "r_nine": 0.0, "f1_nine": 0.0,
                    "wF1": 0.0
                }
            )

        for key in resutls.keys():
            self.logger.info('{} = {}\n'.format(key, str(resutls[key])))

        # self.logger.info(resutls)

        for key, value in resutls.items():
            self.tb_writer.add_scalar('eval_threshold:{}_{}'.format(self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, key), value,
                                 self.iter)
            if aml_run is not None:
                aml_run.log('eval_threshold:{}_{}'.format(self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, key), value)

        return resutls


    def after_step(self):
        self.user_logging()
        for h in self._hooks:
            h.after_step()

        if self.iter != 0 and self.iter % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 and self.local_rank in [-1, 0]:
            self.evaluate()




    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        
        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, grad_scaler=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            mrcnn_loss_dict_return = {
                'loss_cls': loss_dict['loss_cls'].item(),
                'loss_box_reg': loss_dict['loss_box_reg'].item(),
                'loss_mask': loss_dict['loss_mask'].item(),
                'loss_rpn_cls': loss_dict['loss_rpn_cls'].item(),
                'loss_rpn_loc': loss_dict['loss_rpn_loc'].item(),
                'mrcnn_total_loss': losses.item(),
            }

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return mrcnn_loss_dict_return