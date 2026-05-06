# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

import torch
import torchtune.modules.common_utils as common_utils
from omegaconf import DictConfig, ListConfig, OmegaConf

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchmetrics.classification import MulticlassCalibrationError

from transformers import AutoTokenizer

from tqdm import tqdm
from copy import deepcopy
import numpy as np

import logging
import json
import os
import shutil

# Import probabilistic reasoning utilities
from probabilistic_reasoning_utils import (
    get_number_token_ids,
    get_english_number_token_info,
    compute_probabilistic_loss,
    extract_number_probabilities,
    evaluate_predictions,
    _find_answer_positions,
    ProbabilisticReasoningDataset,
    probabilistic_reasoning_collate_fn,
    SingleScenarioDataset,
    single_scenario_collate_fn,
)

# Import validation prediction analysis utilities
from analyze_validation_predictions import (
    load_predictions,
    extract_means_and_modes,
    compute_direction_metrics,
    plot_prediction_vs_ground_truth,
    plot_direction_analysis,
    plot_comparison,
    analyze_learning_vs_centering,
)

VAL_SUBSET_SIZE = 1000
VAL_SUBSET_SEED = 42
PYRO_VAL_BATCHES = 100  # max batches evaluated from pyro val/test per epoch


def _get_val_subset(ds):
    """Return a fixed random subset of VAL_SUBSET_SIZE examples from ds."""
    if len(ds) <= VAL_SUBSET_SIZE:
        return ds
    rng = np.random.RandomState(VAL_SUBSET_SEED)
    indices = sorted(rng.choice(len(ds), VAL_SUBSET_SIZE, replace=False).tolist())
    return Subset(ds, indices)


class FullFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Full-parameter SFT recipe for dense transformer-based LLMs such as Llama3. Adapted from
    ``custom_lora_answer_only.py`` by stripping the LoRA-specific machinery: all model parameters
    are trained, no adapter weights are stored, and checkpoints contain the full state dict.

    Focused on the pyro-distribution training path (``pyro_train_json`` in the config).
    The fusion / pyro-fusion / forward-sampling paths from the LoRA variant are still
    present for compatibility but are not the primary target.
    Single-GPU training only; training on CPU is not supported.

    Features:
        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5 or later and will
            be enabled by default if an acceptable torch version is found. Activation offloading can be
            used in conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * gradient accumulation steps.

            For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW.

        - Checkpointing. Full model weights are checkpointed each time validation loss improves.
            The entire state_dict is saved (no LoRA adapter/merged split).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator

    """

    def __init__(self, cfg: DictConfig) -> None:
        self.filename = FILENAME
        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        if self._log_peak_memory_stats and self._device.type == "cpu":
            log.info(
                "log_peak_memory_stats was set to True, however, training uses cpu. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        # Fusion training: M probabilistic epochs followed by N forward epochs
        self._prob_epochs = cfg.get("prob_epochs", None)
        self._forward_epochs = cfg.get("forward_epochs", None)
        self._is_fusion = (self._prob_epochs is not None and self._forward_epochs is not None)
        if self._is_fusion:
            self.total_epochs = self._prob_epochs + self._forward_epochs

        # Pyro-fusion mode: K pyro epochs (phase 1) then remaining epochs on forward dataset (phase 2).
        # Activated when pyro_fusion_epochs + pyro_train_json + forward_dataset are all set.
        # Takes priority over the older prob_epochs/forward_epochs fusion mode.
        self._pyro_fusion_epochs = cfg.get("pyro_fusion_epochs", None)
        self._is_pyro_fusion = (
            self._pyro_fusion_epochs is not None
            and cfg.get("pyro_train_json", None) is not None
            and cfg.get("forward_dataset", None) is not None
        )
        if self._is_pyro_fusion:
            # Reuse the fusion training-loop machinery
            self._is_fusion = True
            self._prob_epochs = self._pyro_fusion_epochs
            self._forward_epochs = self.total_epochs - self._pyro_fusion_epochs

        # Loss configuration for probabilistic reasoning
        self._loss_on_format_tokens = cfg.get("loss_on_format_tokens", False)
        self._loss_mode = cfg.get("loss_mode", "distribution")  # "distribution" or "mean_only"
        if self._loss_mode not in ("distribution", "mean_only"):
            raise ValueError(
                f"loss_mode must be 'distribution' or 'mean_only', got '{self._loss_mode}'"
            )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        model weights. If resume_from_checkpoint is True, this also includes
        the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            # _update_recipe_state will throw an exception if the recipe state is not correctly loaded
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self.cfg = cfg

        # Compute final output directory BEFORE instantiating the metric logger
        # or writing any config file, so only the suffixed folder is ever created.
        dir_suffix = "_sft"
        if self._loss_mode == "mean_only":
            dir_suffix += "_mean_only"
        else:
            dir_suffix += "_dist"
        self._output_dir = os.path.join(
            os.path.dirname(self._output_dir),
            os.path.basename(self._output_dir) + dir_suffix,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        OmegaConf.update(cfg, "output_dir", self._output_dir)
        OmegaConf.update(cfg, "checkpointer.output_dir", self._output_dir)

        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        self._compile = cfg.compile
        if cfg.device == "npu" and cfg.compile:
            raise ValueError(
                "NPU does not support model compilation. Please set `compile: False` in the config."
            )

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        # hack to toggle to the low cpu ram version of the reparametrize_as_dtype
        # hook based on the config.
        common_utils._use_low_cpu_ram = cfg.get("low_cpu_ram", False)

        # set up model
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=cfg.compile,
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
                        cfg_optimizer=self.cfg.optimizer,
                        model=self._model,
                        opt_state_dict=(
                            None
                        ),
                    )

        # self.meta_optimizer = self._setup_optimizer(
        #     cfg_optimizer=cfg.optimizer,
        #     opt_state_dict=(
        #         checkpoint_dict[training.OPT_KEY]
        #         if self._resume_from_checkpoint
        #         else None
        #     ),
        # )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if self._compile:
            self._loss_fn = training.compile_loss(self._loss_fn)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        log.info("Loss is initialized.")

        # Initialize number token IDs for probabilistic reasoning
        self._number_token_ids = get_number_token_ids(self._tokenizer)
        log.info("Number token IDs initialized for tokens 0-100.")
        log.info(f"Loss config: loss_mode={self._loss_mode}, loss_on_format_tokens={self._loss_on_format_tokens}")

        # Initialize English number word token info
        self._english_number_info = get_english_number_token_info(self._tokenizer)
        n_single = sum(self._english_number_info['is_single_token'])
        log.info(f"English number token info initialized ({n_single}/101 are single-token).")

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        if self._is_pyro_fusion:
            self._pyro_mode = False
            self._setup_pyro_fusion_data(
                cfg_dataset=cfg.dataset,
                pyro_train_json=cfg.pyro_train_json,
                pyro_val_json=cfg.get("pyro_val_json", None),
                pyro_test_json=cfg.get("pyro_test_json", None),
                cfg_forward_dataset=cfg.forward_dataset,
                shuffle=cfg.shuffle,
                batch_size=cfg.batch_size,
            )
            self._dataloader_prob_val = None
            self._dataloader_prob_test = None
        elif self._is_fusion:
            self._pyro_mode = False
            self._setup_fusion_data(
                cfg_prob_dataset=cfg.dataset,
                cfg_forward_dataset=cfg.forward_dataset,
                shuffle=cfg.shuffle,
                batch_size=cfg.batch_size,
            )
            # In fusion mode val/test always come from the probabilistic dataset
            self._dataloader_prob_val = None
            self._dataloader_prob_test = None
        else:
            pyro_train_json = cfg.get("pyro_train_json", None)
            self._pyro_mode = pyro_train_json is not None

            if self._pyro_mode:
                self._setup_pyro_data(
                    cfg_dataset=cfg.dataset,
                    pyro_train_json=pyro_train_json,
                    pyro_val_json=cfg.get("pyro_val_json", None),
                    pyro_test_json=cfg.get("pyro_test_json", None),
                    shuffle=cfg.shuffle,
                    batch_size=cfg.batch_size,
                )
                self._dataloader_prob_val = None  # not used in pyro mode (_dataloader_v holds pyro val)
            else:
                collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
                self._setup_data(
                    cfg_dataset=cfg.dataset,
                    shuffle=cfg.shuffle,
                    batch_size=cfg.batch_size,
                    collate_fn=collate_name,
                )
                # Optional probabilistic-reasoning val/test sets (for forward-sampling mode).
                # Both use the _prob_ prefix consistently; webppl_ naming is reserved for
                # pyro mode where it distinguishes the cross-eval from the pyro train split.
                max_seq_len = cfg.get("max_seq_len", 2048)
                prob_val_json  = cfg.get("prob_val_json",  None)
                prob_test_json = cfg.get("prob_test_json", None)
                self._dataloader_prob_val  = None
                self._dataloader_prob_test = None
                self._dataloader_webppl_val  = None  # unused in non-pyro mode
                self._dataloader_webppl_test = None  # unused in non-pyro mode
                if prob_val_json:
                    ds_prob_val = _get_val_subset(
                        ProbabilisticReasoningDataset(prob_val_json, self._tokenizer, max_seq_length=max_seq_len)
                    )
                    sampler_prob_val = DistributedSampler(ds_prob_val, num_replicas=1, rank=0, shuffle=False, seed=0)
                    self._dataloader_prob_val = DataLoader(
                        ds_prob_val, batch_size=cfg.batch_size, sampler=sampler_prob_val,
                        collate_fn=probabilistic_reasoning_collate_fn,
                    )
                    log.info(f"Prob-reasoning val set loaded: {len(ds_prob_val)} examples from {prob_val_json}")
                if prob_test_json:
                    ds_prob_test = _get_val_subset(
                        ProbabilisticReasoningDataset(prob_test_json, self._tokenizer, max_seq_length=max_seq_len)
                    )
                    sampler_prob_test = DistributedSampler(ds_prob_test, num_replicas=1, rank=0, shuffle=False, seed=0)
                    self._dataloader_prob_test = DataLoader(
                        ds_prob_test, batch_size=cfg.batch_size, sampler=sampler_prob_test,
                        collate_fn=probabilistic_reasoning_collate_fn,
                    )
                    log.info(f"Prob-reasoning test set loaded: {len(ds_prob_test)} examples from {prob_test_json}")

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = 0
        for dataloader in self._dataloaderlist:
            self._steps_per_epoch += len(dataloader) // self._gradient_accumulation_steps
        
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.global_step = self.epochs_run * self._steps_per_epoch

        # # Learning rate scheduler can only be set up after number of steps
        # # has been computed
        # self._lr_scheduler = self._setup_lr_scheduler(
        #     cfg_lr_scheduler=cfg.lr_scheduler,
        #     num_training_steps=self.total_epochs * self._steps_per_epoch,
        #     last_epoch=self.global_step - 1,
        # )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """

        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        log.info(f" Profiler config after instantiation: {profiler_cfg}")

        self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
        if profiler_cfg["enabled"]:
            self.profiler_wait_steps = profiler_cfg["wait_steps"]
            self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
            self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        log.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)
        return model

    def _setup_optimizer(
        self, 
        cfg_optimizer: DictConfig,
        model, 
        opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler
    
    def _setup_data_helper(self, ds, shuffle, batch_size, collate_fn, packed):

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
        )

        return sampler, dataloader

    def _setup_pyro_fusion_data(
        self,
        cfg_dataset: DictConfig,
        pyro_train_json: str,
        pyro_val_json: Optional[str],
        pyro_test_json: Optional[str],
        cfg_forward_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> None:
        """
        Set up data loaders for pyro-fusion mode.

        Phase 1 (pyro_fusion_epochs epochs): train on pyro_train_json.
        Phase 2 (remaining epochs): train on cfg_forward_dataset.
        Both phases validate/test on:
          - pyro val/test (_dataloader_v / _dataloader_test)
          - webppl val/test from cfg_dataset (_dataloader_webppl_val / _dataloader_webppl_test)
        """
        max_seq_len = self.cfg.get("max_seq_len", 2048)
        packed = False

        # ── Phase 1 train: pyro JSON ───────────────────────────────────────────
        ds_pyro_train = ProbabilisticReasoningDataset(
            pyro_train_json, self._tokenizer, max_seq_length=max_seq_len
        )
        sampler, dataloader = self._setup_data_helper(
            ds_pyro_train, shuffle, batch_size, probabilistic_reasoning_collate_fn, packed
        )
        self._samplerlist = [sampler]
        self._dataloaderlist = [dataloader]

        # ── Phase 2 train: forward dataset ────────────────────────────────────
        ds_fwd_train, _, _ = config.instantiate(cfg_forward_dataset, self._tokenizer)
        self._samplerlist_forward = []
        self._dataloaderlist_forward = []
        for ds in ds_fwd_train:
            sampler, dataloader = self._setup_data_helper(
                ds, shuffle, batch_size, single_scenario_collate_fn, packed
            )
            self._samplerlist_forward.append(sampler)
            self._dataloaderlist_forward.append(dataloader)

        # ── Pyro val ──────────────────────────────────────────────────────────
        self._sampler_v = []
        self._dataloader_v = []
        if pyro_val_json:
            ds_pyro_val = ProbabilisticReasoningDataset(
                pyro_val_json, self._tokenizer, max_seq_length=max_seq_len
            )
            sampler, dataloader = self._setup_data_helper(
                ds_pyro_val, False, batch_size, probabilistic_reasoning_collate_fn, packed
            )
            self._sampler_v = [sampler]
            self._dataloader_v = [dataloader]

        # ── Pyro test ─────────────────────────────────────────────────────────
        self._sampler_test = []
        self._dataloader_test = []
        if pyro_test_json:
            ds_pyro_test = ProbabilisticReasoningDataset(
                pyro_test_json, self._tokenizer, max_seq_length=max_seq_len
            )
            sampler, dataloader = self._setup_data_helper(
                ds_pyro_test, False, batch_size, probabilistic_reasoning_collate_fn, packed
            )
            self._sampler_test = [sampler]
            self._dataloader_test = [dataloader]

        # ── WebPPL val + test (probabilistic_reasoning splits) ────────────────
        _, ds_webppl_val_list, ds_webppl_test_list = config.instantiate(cfg_dataset, self._tokenizer)

        self._dataloader_webppl_val = None
        if ds_webppl_val_list:
            ds_pv = ds_webppl_val_list[0]
            sampler_pv = DistributedSampler(ds_pv, num_replicas=1, rank=0, shuffle=False, seed=0)
            self._dataloader_webppl_val = DataLoader(
                ds_pv, batch_size=batch_size, sampler=sampler_pv,
                collate_fn=probabilistic_reasoning_collate_fn,
            )

        self._dataloader_webppl_test = None
        if ds_webppl_test_list:
            ds_pt = ds_webppl_test_list[0]
            sampler_pt = DistributedSampler(ds_pt, num_replicas=1, rank=0, shuffle=False, seed=0)
            self._dataloader_webppl_test = DataLoader(
                ds_pt, batch_size=batch_size, sampler=sampler_pt,
                collate_fn=probabilistic_reasoning_collate_fn,
            )

        log.info(
            f"Pyro-fusion data loaded: {len(self._dataloaderlist[0])} pyro-train batches (phase 1), "
            f"{len(self._dataloaderlist_forward[0])} fwd-train batches (phase 2)"
        )
        if self._dataloader_v:
            log.info(f"  Pyro val : {len(self._dataloader_v[0])} batches (capped at {PYRO_VAL_BATCHES} per eval)")
        if self._dataloader_test:
            log.info(f"  Pyro test: {len(self._dataloader_test[0])} batches (capped at {PYRO_VAL_BATCHES} per eval)")
        if self._dataloader_webppl_val:
            log.info(f"  WebPPL val : {len(self._dataloader_webppl_val)} batches (full)")
        if self._dataloader_webppl_test:
            log.info(f"  WebPPL test: {len(self._dataloader_webppl_test)} batches (full)")

    def _setup_fusion_data(
        self,
        cfg_prob_dataset: DictConfig,
        cfg_forward_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> None:
        """
        Load both probabilistic and forward datasets for fusion training.

        Probabilistic dataset provides train (phase 1), val, and test loaders.
        Forward dataset provides only the train loader for phase 2.
        Val/test always use probabilistic_reasoning_collate_fn throughout.
        """
        # Load probabilistic dataset (train + val + test)
        ds_prob_train, ds_prob_dev, ds_prob_test = config.instantiate(cfg_prob_dataset, self._tokenizer)

        # Load forward dataset (train only; val/test come from probabilistic)
        ds_fwd_train, _, _ = config.instantiate(cfg_forward_dataset, self._tokenizer)

        packed = False

        # Phase 1 train: probabilistic, with ground_truth_bins
        self._samplerlist = []
        self._dataloaderlist = []
        for ds in ds_prob_train:
            sampler, dataloader = self._setup_data_helper(
                ds, shuffle, batch_size, probabilistic_reasoning_collate_fn, packed
            )
            self._samplerlist.append(sampler)
            self._dataloaderlist.append(dataloader)

        # Phase 2 train: forward-sampling, no bins (standard CE loss)
        self._samplerlist_forward = []
        self._dataloaderlist_forward = []
        for ds in ds_fwd_train:
            sampler, dataloader = self._setup_data_helper(
                ds, shuffle, batch_size, single_scenario_collate_fn, packed
            )
            self._samplerlist_forward.append(sampler)
            self._dataloaderlist_forward.append(dataloader)

        # Val/test always from probabilistic dataset
        self._sampler_v = []
        self._dataloader_v = []
        self._sampler_test = []
        self._dataloader_test = []
        for ds in ds_prob_test:
            ds = _get_val_subset(ds)
            sampler, dataloader = self._setup_data_helper(
                ds, False, batch_size, probabilistic_reasoning_collate_fn, packed
            )
            self._sampler_test.append(sampler)
            self._dataloader_test.append(dataloader)
        for ds in ds_prob_dev:
            ds = _get_val_subset(ds)
            sampler, dataloader = self._setup_data_helper(
                ds, False, batch_size, probabilistic_reasoning_collate_fn, packed
            )
            self._sampler_v.append(sampler)
            self._dataloader_v.append(dataloader)

        log.info(
            f"Fusion data loaded: {len(self._dataloaderlist[0])} prob-train batches (phase 1), "
            f"{len(self._dataloaderlist_forward[0])} fwd-train batches (phase 2), "
            f"{len(self._dataloader_v[0])} val batches, {len(self._dataloader_test[0])} test batches"
        )

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        ds_list_train, ds_list_dev, ds_list_test = config.instantiate(cfg_dataset, self._tokenizer)
        packed = False

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")

        # Choose train collate function based on dataset type.
        # When training on SingleScenarioDataset (no bins), use the simpler collate fn
        # that omits ground_truth_bins; the loss step will fall back to standard CE loss.
        # Val/test always use probabilistic_reasoning_collate_fn (they have bins).
        if ds_list_train and isinstance(ds_list_train[0], SingleScenarioDataset):
            train_collate_fn = single_scenario_collate_fn
            log.info("Using single_scenario_collate_fn for training (no bins); "
                     "val/test will use probabilistic_reasoning_collate_fn.")
        else:
            train_collate_fn = probabilistic_reasoning_collate_fn

        self._samplerlist = []
        self._dataloaderlist = []
        for ds in ds_list_train:
            sampler, dataloader = self._setup_data_helper(ds, shuffle, batch_size, train_collate_fn, packed)
            self._samplerlist.append(sampler)
            self._dataloaderlist.append(dataloader)

        self._sampler_v = []
        self._dataloader_v = []
        self._sampler_test = []
        self._dataloader_test = []
        for ds in ds_list_test:
            ds = _get_val_subset(ds)
            sampler, dataloader = self._setup_data_helper(ds, False, batch_size, probabilistic_reasoning_collate_fn, packed)
            self._sampler_test.append(sampler)
            self._dataloader_test.append(dataloader)
        for ds in ds_list_dev:
            ds = _get_val_subset(ds)
            sampler, dataloader = self._setup_data_helper(ds, False, batch_size, probabilistic_reasoning_collate_fn, packed)
            self._sampler_v.append(sampler)
            self._dataloader_v.append(dataloader)

        log.info("Dataset and Sampler are initialized.")
        log.info(f"Loaded probabilistic reasoning dataset: {len(self._dataloaderlist[0])} train, "
                f"{len(self._dataloader_v[0])} val, {len(self._dataloader_test[0])} test batches")

        return

    def _setup_pyro_data(
        self,
        cfg_dataset: DictConfig,
        pyro_train_json: str,
        pyro_val_json: Optional[str],
        pyro_test_json: Optional[str],
        shuffle: bool,
        batch_size: int,
    ) -> None:
        """
        Set up data loaders for pyro/pytorch_mcmc training mode.

        Training comes from pyro_train_json (full dataset).
        Val and test come from pyro_val_json / pyro_test_json, capped at
        PYRO_VAL_BATCHES batches per evaluation (fixed first batches, seed=0).
        WebPPL val and test (probabilistic_reasoning val/test splits) are loaded
        from cfg_dataset into _dataloader_webppl_val/test and evaluated in full
        every epoch.
        """
        max_seq_len = self.cfg.get("max_seq_len", 2048)
        packed = False

        # ── Pyro train ────────────────────────────────────────────────────────
        ds_pyro_train = ProbabilisticReasoningDataset(
            pyro_train_json, self._tokenizer, max_seq_length=max_seq_len
        )
        sampler, dataloader = self._setup_data_helper(
            ds_pyro_train, shuffle, batch_size, probabilistic_reasoning_collate_fn, packed
        )
        self._samplerlist = [sampler]
        self._dataloaderlist = [dataloader]

        # ── Pyro val (no subset; capped in _run_epoch_validation) ─────────────
        self._sampler_v = []
        self._dataloader_v = []
        if pyro_val_json:
            ds_pyro_val = ProbabilisticReasoningDataset(
                pyro_val_json, self._tokenizer, max_seq_length=max_seq_len
            )
            sampler, dataloader = self._setup_data_helper(
                ds_pyro_val, False, batch_size, probabilistic_reasoning_collate_fn, packed
            )
            self._sampler_v = [sampler]
            self._dataloader_v = [dataloader]

        # ── Pyro test (same) ─────────────────────────────────────────────────
        self._sampler_test = []
        self._dataloader_test = []
        if pyro_test_json:
            ds_pyro_test = ProbabilisticReasoningDataset(
                pyro_test_json, self._tokenizer, max_seq_length=max_seq_len
            )
            sampler, dataloader = self._setup_data_helper(
                ds_pyro_test, False, batch_size, probabilistic_reasoning_collate_fn, packed
            )
            self._sampler_test = [sampler]
            self._dataloader_test = [dataloader]

        # ── WebPPL val + test (probabilistic_reasoning val/test splits) ────────
        _, ds_webppl_val_list, ds_webppl_test_list = config.instantiate(cfg_dataset, self._tokenizer)

        # Pyro mode never uses _dataloader_prob_val/test (those are for forward-sampling mode)
        self._dataloader_prob_val = None
        self._dataloader_prob_test = None

        self._dataloader_webppl_val = None
        if ds_webppl_val_list:
            ds_pv = ds_webppl_val_list[0]
            sampler_pv = DistributedSampler(ds_pv, num_replicas=1, rank=0, shuffle=False, seed=0)
            self._dataloader_webppl_val = DataLoader(
                ds_pv, batch_size=batch_size, sampler=sampler_pv,
                collate_fn=probabilistic_reasoning_collate_fn,
            )

        self._dataloader_webppl_test = None
        if ds_webppl_test_list:
            ds_pt = ds_webppl_test_list[0]
            sampler_pt = DistributedSampler(ds_pt, num_replicas=1, rank=0, shuffle=False, seed=0)
            self._dataloader_webppl_test = DataLoader(
                ds_pt, batch_size=batch_size, sampler=sampler_pt,
                collate_fn=probabilistic_reasoning_collate_fn,
            )

        log.info(f"Pyro data loaded: {len(self._dataloaderlist[0])} train batches")
        if self._dataloader_v:
            log.info(f"  Pyro val : {len(self._dataloader_v[0])} batches total "
                     f"(capped at {PYRO_VAL_BATCHES} per eval)")
        if self._dataloader_test:
            log.info(f"  Pyro test: {len(self._dataloader_test[0])} batches total "
                     f"(capped at {PYRO_VAL_BATCHES} per eval)")
        if self._dataloader_webppl_val:
            log.info(f"  WebPPL val : {len(self._dataloader_webppl_val)} batches (full)")
        if self._dataloader_webppl_test:
            log.info(f"  WebPPL test: {len(self._dataloader_webppl_test)} batches (full)")

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Full model weights with key MODEL_KEY
        - Optimizer state and recipe state if training is not complete

        Note: Always saves to epoch_0 folder to override previous checkpoints.
        """
        ckpt_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # if training is in-progress, checkpoint the optimizer state as well
        if intermediate_checkpoint:
            ckpt_dict.update(
                {
                    training.OPT_KEY: self._optimizer.state_dict(),
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self.epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        # Full model state dict (moved to CPU to avoid a GPU-side copy)
        state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}
        ckpt_dict.update({training.MODEL_KEY: state_dict})

        # Always save to epoch_0 folder to override previous checkpoints
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=0,
            intermediate_checkpoint=intermediate_checkpoint,
        )
        # Save metadata file documenting which epoch this checkpoint is from
        ckpt_dir = os.path.join(self._output_dir, "epoch_0")
        with open(os.path.join(ckpt_dir, "checkpoint_epoch.txt"), "w") as f:
            f.write(f"{epoch}\n")

        log.info(f"Checkpoint saved (actual epoch: {epoch}, saved to epoch_0 folder)")

    def _loss_step(self, batch: Dict[str, torch.Tensor], model):
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")
        ground_truth_bins = batch.pop("ground_truth_bins", None)
        num_queries = batch.pop("num_queries", None)
        batch.pop("mask", None)  # 2-D loss-region mask; not an attention mask

        # run model
        with self.activations_handling_ctx:
            logits = model(**batch)

        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we dont need to slice the logits. We just add an ignore index to labels.
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        )

        # Handle chunked logits: convert to single tensor for probabilistic loss
        if isinstance(logits, list):
            # Concatenate chunked logits back into single tensor
            logits_tensor = torch.cat(logits, dim=1)
        else:
            logits_tensor = logits

        # Flatten for standard loss computation
        labels_flat = labels.reshape(-1)
        logits_flat = logits_tensor.reshape(-1, logits_tensor.size(-1))

        total_elements = (labels_flat != self._loss_fn.ignore_index).sum()

        # Debug: Print first training example
        if not self._debug_printed_train:
            self._debug_printed_train = True
            self._debug_print_example(
                logits_tensor, labels, "TRAIN", self._tokenizer,
                ground_truth_bins=ground_truth_bins,
                num_queries=num_queries
            )

        # Compute loss with probabilistic reasoning if bins are provided
        if ground_truth_bins is not None and hasattr(self, '_number_token_ids'):
            # Use probabilistic reasoning loss (expects tensor, not list)
            loss, loss_dict = compute_probabilistic_loss(
                logits=logits_tensor,
                labels=labels,
                ground_truth_bins=ground_truth_bins,
                number_token_ids=self._number_token_ids,
                ce_weight=1.0 if self._loss_on_format_tokens else 0.0,
                dist_weight=1.0,
                ignore_index=self._loss_fn.ignore_index,
                num_queries=num_queries,
                tokenizer=self._tokenizer,
                loss_mode=self._loss_mode,
            )
            # Extract metrics from loss_dict
            ll = -loss_dict['ce_loss'] * total_elements.detach().float().cpu().numpy()
            metrics = [0, 0, ll]  # Placeholder for acc, ece

            # Enhanced logging
            if 'dist_loss' in loss_dict and loss_dict['dist_loss'] > 0:
                log.info(f"Loss: Total={loss_dict['total_loss']:.4f} | CE={loss_dict['ce_loss']:.4f} | Dist={loss_dict['dist_loss']:.4f}")
                if num_queries is not None:
                    log.info(f"  Batch size: {logits_tensor.size(0)}, Queries: {num_queries[0].item() if len(num_queries) > 0 else 'N/A'}")
            else:
                log.info(f"Loss: Total={loss_dict['total_loss']:.4f} | CE={loss_dict['ce_loss']:.4f} | Dist=0.0 (WARNING: No answer positions found!)")
        else:
            # Standard loss computation
            # Pass non-flattened tensors so CEChunkedOutputLoss can chunk along dim=1
            loss = self._loss_fn(logits, labels)
            ll = -(loss*total_elements).detach().float().cpu().numpy()
            metrics = [0, 0, ll]  # Use 0 for acc/ece since get_metrics expects chunked format

        # free logits otherwise it peaks backward memory
        del logits
        del logits_tensor

        return loss, metrics


    def get_metrics(self, logits, labels):
        """
        Compute accuracy and ECE metrics.
        Note: For probabilistic reasoning, dedicated evaluation is in evaluate_probabilistic_predictions.
        This function is kept for compatibility with the training loop.
        """
        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self._loss_fn.num_output_chunks, dim=1)
        ]
        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        logits = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]
        correct = 0
        all_count = 0
        ece = 0

        # For probabilistic reasoning, return placeholder metrics
        # The actual evaluation uses evaluate_probabilistic_predictions
        ece_metric = MulticlassCalibrationError(num_classes=4, n_bins=10, norm='l1')
        for i in range(len(logits)):
            tokens = torch.argmax(logits[i], 1)
            indices = torch.tensor(torch.tensor(labels[i] != -100).int())
            if torch.sum(indices) == 0:
                continue
            else:
                indices = indices.nonzero()
                for idx in indices:
                    if tokens[idx] == labels[i][idx]:
                        correct += 1
                    all_count += 1

        if all_count != 0:
            acc = correct / all_count
        else:
            acc = 0
            ece = 0

        del logits
        return acc, ece

    def _debug_print_example(self, logits, labels, split_name, tokenizer, ground_truth_bins=None, num_queries=None):
        """
        Debug helper to print model output and labels for the first example.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Shifted labels [batch_size, seq_len]
            split_name: Name of the split (TRAIN, VAL, TEST)
            tokenizer: Tokenizer for decoding
            ground_truth_bins: Optional ground truth distributions [batch_size, max_queries, 101]
            num_queries: Optional number of queries per example [batch_size]
        """
        log.info(f"\n{'='*60}")
        log.info(f"DEBUG: First {split_name} example")
        log.info(f"{'='*60}")

        # Get first example
        example_logits = logits[0]  # [seq_len, vocab_size]
        example_labels = labels[0]  # [seq_len]

        # Get model predictions (argmax)
        predictions = torch.argmax(example_logits, dim=-1)  # [seq_len]

        # Find positions where labels are not -100 (i.e., output positions)
        valid_mask = example_labels != -100
        valid_positions = valid_mask.nonzero(as_tuple=True)[0]

        log.info(f"Sequence length: {example_logits.size(0)}")
        log.info(f"Number of valid (non-masked) positions: {valid_positions.size(0)}")

        if valid_positions.size(0) > 0:
            # Get predictions and labels at valid positions
            valid_preds = predictions[valid_mask].cpu().tolist()
            valid_labels = example_labels[valid_mask].cpu().tolist()

            # Decode to words
            try:
                pred_words = tokenizer.decode(valid_preds)
                label_words = tokenizer.decode(valid_labels)
            except Exception as e:
                pred_words = f"[Decode error: {e}]"
                label_words = f"[Decode error: {e}]"

            log.info(f"\n--- Valid positions (output tokens) ---")
            log.info(f"Positions: {valid_positions[:20].cpu().tolist()}{'...' if len(valid_positions) > 20 else ''}")
            log.info(f"Label token IDs: {valid_labels[:20]}{'...' if len(valid_labels) > 20 else ''}")
            log.info(f"Pred token IDs:  {valid_preds[:20]}{'...' if len(valid_preds) > 20 else ''}")
            log.info(f"Labels as text: {label_words[:200]}{'...' if len(label_words) > 200 else ''}")
            log.info(f"Preds as text:  {pred_words[:200]}{'...' if len(pred_words) > 200 else ''}")

            # Show token-by-token comparison for first few tokens
            log.info(f"\n--- Token-by-token (first 10 valid positions) ---")
            for i, pos in enumerate(valid_positions[:10].cpu().tolist()):
                label_tok = example_labels[pos].item()
                pred_tok = predictions[pos].item()
                try:
                    label_word = tokenizer.decode([label_tok])
                    pred_word = tokenizer.decode([pred_tok])
                except:
                    label_word = f"[{label_tok}]"
                    pred_word = f"[{pred_tok}]"
                match = "✓" if label_tok == pred_tok else "✗"
                log.info(f"  Pos {pos}: label={label_tok} '{label_word}' | pred={pred_tok} '{pred_word}' {match}")

            # ADDED: Show answer positions for probabilistic reasoning
            if ground_truth_bins is not None and hasattr(self, '_number_token_ids'):
                log.info(f"\n--- Probabilistic Reasoning Answer Positions ---")

                n_queries = num_queries[0].item() if num_queries is not None else ground_truth_bins.size(1)
                log.info(f"Number of queries: {n_queries}")

                # Find answer positions (replicate logic from compute_distribution_loss)
                number_token_set = set(self._number_token_ids.tolist())
                answer_positions = []

                for pos in range(example_labels.size(0)):
                    label_val = example_labels[pos].item()
                    if label_val == -100:
                        continue

                    # Method 1: Direct token ID match
                    if label_val in number_token_set:
                        answer_positions.append(pos)
                        continue

                    # Method 2: Decode and check if it's a number
                    try:
                        decoded = tokenizer.decode([label_val]).strip()
                        if decoded.isdigit() and 0 <= int(decoded) <= 100:
                            answer_positions.append(pos)
                    except:
                        pass

                log.info(f"Found {len(answer_positions)} answer positions: {answer_positions[:10]}")

                # Show mapping from answer position to query
                for q in range(min(n_queries, len(answer_positions))):
                    pos = answer_positions[q]
                    label_val = example_labels[pos].item()
                    pred_val = predictions[pos].item()

                    try:
                        label_decoded = tokenizer.decode([label_val]).strip()
                        pred_decoded = tokenizer.decode([pred_val]).strip()
                    except:
                        label_decoded = f"[{label_val}]"
                        pred_decoded = f"[{pred_val}]"

                    # Get ground truth stats
                    gt_bins = ground_truth_bins[0, q].cpu().numpy()
                    gt_mean = sum(i * gt_bins[i] for i in range(101))
                    gt_mode = int(gt_bins.argmax())

                    # Get predicted distribution over number tokens
                    number_logits = example_logits[pos, self._number_token_ids].detach().cpu()
                    pred_probs = torch.softmax(number_logits, dim=0).float().numpy()
                    pred_mean = sum(i * pred_probs[i] for i in range(101))
                    pred_mode = int(pred_probs.argmax())

                    log.info(f"\n  Query {q+1} → Position {pos}:")
                    log.info(f"    Ground truth: '{label_decoded}' (token {label_val})")
                    log.info(f"    Predicted:    '{pred_decoded}' (token {pred_val}) {'✓' if label_decoded == pred_decoded else '✗'}")
                    log.info(f"    GT distribution: mean={gt_mean:.1f}, mode={gt_mode}")
                    log.info(f"    Predicted dist:  mean={pred_mean:.1f}, mode={pred_mode}")
                    log.info(f"    Top-5 GT probs: {sorted([(i, gt_bins[i]) for i in range(101)], key=lambda x: -x[1])[:5]}")
                    log.info(f"    Top-5 pred probs: {sorted([(i, pred_probs[i]) for i in range(101)], key=lambda x: -x[1])[:5]}")
        else:
            log.info("No valid (non-masked) positions found in labels!")

        log.info(f"{'='*60}\n")

    def evaluate_probabilistic_predictions(self, logits, ground_truth_bins, num_queries=None, labels=None):
        """
        Evaluate predicted probability distributions against ground truth.

        When loss_mode="distribution", computes distributional metrics (KL, CE over 101 bins, MAE).
        When loss_mode="mean_only", computes standard CE at answer positions using the
        ground truth mean token as the single correct target.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            ground_truth_bins: Ground truth distributions [batch_size, max_queries, 101]
            num_queries: Number of valid queries per example [batch_size]
            labels: Labels tensor [batch_size, seq_len] to identify answer positions

        Returns:
            metrics: Dictionary with KL divergence, cross-entropy, and MAE
        """
        batch_size = logits.size(0)

        if self._loss_mode == "mean_only":
            return self._evaluate_mean_only(logits, ground_truth_bins, num_queries, labels)

        all_metrics = []

        # Create a set of number token IDs for fast lookup
        number_token_set = set(self._number_token_ids.tolist())

        for i in range(batch_size):
            # Get ground truth
            gt_bins = ground_truth_bins[i]

            # Determine number of valid queries
            if num_queries is not None:
                n_queries = num_queries[i].item()
            else:
                n_queries = gt_bins.size(0)

            # Find answer positions for this example
            example_labels = labels[i] if labels is not None else None

            pred_dists = []
            gt_dists = []

            # For each query, extract prediction at the corresponding answer position
            for q in range(n_queries):
                pred_probs = extract_number_probabilities(
                    logits[i],
                    self._number_token_ids,
                    method='answer_position',
                    labels=example_labels,
                    query_idx=q,
                )
                pred_dists.append(pred_probs.detach().float().cpu().numpy())
                gt_dists.append(gt_bins[q].cpu().numpy())

            if pred_dists:
                metrics = evaluate_predictions(pred_dists, gt_dists)
                all_metrics.append(metrics)

        # Aggregate across batch
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                if key in ('pred_means', 'gt_means'):
                    combined = []
                    for m in all_metrics:
                        combined.extend(m[key])
                    avg_metrics[key] = combined
                else:
                    avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
            return avg_metrics
        else:
            return {'kl_divergence': 0.0, 'cross_entropy': 0.0, 'cross_entropy_mean': 0.0,
                    'cross_entropy_dist': 0.0, 'mean_abs_error': 0.0, 'mean_abs_error_dist': 0.0,
                    'pred_means': [], 'gt_means': []}

    def _evaluate_mean_only(self, logits, ground_truth_bins, num_queries=None, labels=None):
        """
        Evaluate in mean_only mode: standard CE at answer positions targeting the
        ground truth mean token, plus MAE between predicted and ground truth means.

        Returns same metric keys as the distribution mode for compatibility.
        """
        batch_size = logits.size(0)
        device = logits.device
        number_token_ids = self._number_token_ids.to(device)

        # Find answer positions for the whole batch
        answer_positions = _find_answer_positions(labels, self._number_token_ids, self._tokenizer)

        total_ce = 0.0
        total_ce_mean = 0.0
        total_ce_dist = 0.0
        total_mae = 0.0
        total_mae_dist = 0.0
        total_queries = 0
        pred_means = []
        gt_means = []

        for i in range(batch_size):
            gt_bins = ground_truth_bins[i]
            n_queries = num_queries[i].item() if num_queries is not None else gt_bins.size(0)
            positions = answer_positions[i]

            for q in range(min(n_queries, len(positions))):
                pos = positions[q]

                # Ground truth mean (rounded to nearest int) as the target token
                gt_dist = gt_bins[q].cpu().numpy()
                gt_mean_float = float(sum(idx * gt_dist[idx] for idx in range(101)))
                gt_mean_int = max(0, min(100, int(round(gt_mean_float))))
                gt_token_id = number_token_ids[gt_mean_int]

                # Standard CE at this position: -log P(gt_token) using full vocab
                log_probs = torch.nn.functional.log_softmax(logits[i, pos], dim=0)
                ce = -log_probs[gt_token_id].item()
                total_ce += ce

                # Predicted distribution over 101 number tokens (subspace softmax)
                number_logits = logits[i, pos, number_token_ids]
                pred_probs = torch.softmax(number_logits, dim=0).detach().float().cpu().numpy()
                pred_mean = float(sum(idx * pred_probs[idx] for idx in range(101)))

                # CE (mean GT): -log pred_probs[gt_mean_int] in 101-token subspace
                eps = 1e-10
                total_ce_mean += -np.log(pred_probs[gt_mean_int] + eps)

                # CE (full dist): -sum(gt_dist * log(pred_probs)) in 101-token subspace
                total_ce_dist += -float(np.sum(gt_dist * np.log(pred_probs + eps)))

                # MAE: |pred_mean - gt_mean_float|
                total_mae += abs(pred_mean - gt_mean_float)

                # MAE (dist): L1 distance between full distributions
                total_mae_dist += float(np.sum(np.abs(pred_probs - gt_dist)))

                pred_means.append(pred_mean)
                gt_means.append(gt_mean_float)

                total_queries += 1

        if total_queries > 0:
            return {
                'kl_divergence': 0.0,  # Not applicable in mean_only mode
                'cross_entropy': total_ce / total_queries,
                'cross_entropy_mean': total_ce_mean / total_queries,
                'cross_entropy_dist': total_ce_dist / total_queries,
                'mean_abs_error': total_mae / total_queries,
                'mean_abs_error_dist': total_mae_dist / total_queries,
                'pred_means': pred_means,
                'gt_means': gt_means,
            }
        else:
            return {'kl_divergence': 0.0, 'cross_entropy': 0.0, 'cross_entropy_mean': 0.0,
                    'cross_entropy_dist': 0.0, 'mean_abs_error': 0.0, 'mean_abs_error_dist': 0.0,
                    'pred_means': [], 'gt_means': []}

    def _collect_batch_predictions(
        self,
        logits: torch.Tensor,
        labels_shifted: torch.Tensor,
        ground_truth_bins: torch.Tensor,
        num_queries: torch.Tensor,
        loader_idx: int,
        batch_idx: int,
    ) -> list:
        """
        Collect predictions from a single batch during validation.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels_shifted: Shifted labels [batch_size, seq_len]
            ground_truth_bins: Ground truth distributions [batch_size, max_queries, 101]
            num_queries: Number of queries per example [batch_size]
            loader_idx: Index of the data loader
            batch_idx: Index of the batch

        Returns:
            List of prediction dictionaries for this batch
        """
        batch_predictions = []
        batch_size = logits.size(0)

        for i in range(batch_size):
            n_queries = num_queries[i].item() if num_queries is not None else ground_truth_bins[i].size(0)

            example_predictions = []
            example_ground_truth = []

            for q in range(n_queries):
                # Extract predicted distribution
                pred_probs = extract_number_probabilities(
                    logits[i],
                    self._number_token_ids,
                    method='answer_position',
                    labels=labels_shifted[i],
                    query_idx=q,
                )

                # Get ground truth distribution
                gt_bins = ground_truth_bins[i, q].cpu().numpy().tolist()
                pred_bins = pred_probs.detach().float().cpu().numpy().tolist()

                # Compute summary statistics
                gt_mean = sum(idx * gt_bins[idx] for idx in range(101))
                pred_mean = sum(idx * pred_bins[idx] for idx in range(101))

                example_predictions.append({
                    'query_idx': q,
                    'predicted_distribution': pred_bins,
                    'predicted_mean': pred_mean,
                    'predicted_mode': int(np.argmax(pred_bins)),
                })

                example_ground_truth.append({
                    'query_idx': q,
                    'ground_truth_distribution': gt_bins,
                    'ground_truth_mean': gt_mean,
                    'ground_truth_mode': int(np.argmax(gt_bins)),
                })

            batch_predictions.append({
                'loader_idx': loader_idx,
                'batch_idx': batch_idx,
                'example_idx': i,
                'num_queries': n_queries,
                'predictions': example_predictions,
                'ground_truth': example_ground_truth,
            })

        return batch_predictions

    def _save_predictions_to_file(self, predictions: list, epoch: int, filename: str = None) -> str:
        """
        Save collected predictions to a JSON file.

        Args:
            predictions: List of prediction dictionaries
            epoch: Current epoch number
            filename: Optional filename (if not provided, uses val_predictions_current.json)

        Returns:
            Path to saved file
        """
        pred_dir = os.path.join(self._output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

        if filename is None:
            filename = 'val_predictions_current.json'

        filepath = os.path.join(pred_dir, filename)

        output_data = {
            'epoch': epoch,
            'num_examples': len(predictions),
            'examples': predictions,
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        log.info(f"Saved validation predictions to {filepath} ({len(predictions)} examples)")
        return filepath

    def _get_current_predictions_path(self) -> str:
        """Get path to current validation predictions file."""
        pred_dir = os.path.join(self._output_dir, 'predictions')
        return os.path.join(pred_dir, 'val_predictions_current.json')

    def _copy_predictions_to_checkpoint(self, epoch: int) -> None:
        """
        Copy current validation predictions to checkpoint file.

        Args:
            epoch: Epoch number for the checkpoint
        """
        current_path = self._get_current_predictions_path()

        if not os.path.exists(current_path):
            log.warning(f"Current predictions file not found: {current_path}")
            return

        pred_dir = os.path.join(self._output_dir, 'predictions')
        checkpoint_path = os.path.join(pred_dir, f'val_predictions_epoch_{epoch}_best.json')

        # Copy the file
        shutil.copy2(current_path, checkpoint_path)

        # Update the epoch in the copied file
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        data['epoch'] = epoch
        data['tag'] = 'best'

        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

        log.info(f"Copied validation predictions to checkpoint: {checkpoint_path}")

    def run_prediction_analysis(self, epoch: int) -> None:
        """
        Run analysis on validation predictions to assess learning vs centering.

        Compares pre-training predictions with current checkpoint predictions.

        Args:
            epoch: Current epoch number
        """
        pred_dir = os.path.join(self._output_dir, 'predictions')
        analysis_dir = os.path.join(pred_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # Files to analyze: pretrain and current best
        pretrain_path = os.path.join(pred_dir, 'val_predictions_epoch_-1_pretrain.json')
        current_path = os.path.join(pred_dir, f'val_predictions_epoch_{epoch}_best.json')

        files_to_analyze = []
        if os.path.exists(pretrain_path):
            files_to_analyze.append(('pretrain', pretrain_path))
        if os.path.exists(current_path):
            files_to_analyze.append((f'epoch_{epoch}', current_path))

        if not files_to_analyze:
            log.warning("No prediction files found for analysis")
            return

        all_results = {}

        for name, filepath in files_to_analyze:
            try:
                data = load_predictions(filepath)
                pred_means, gt_means, _, _ = extract_means_and_modes(data)

                if len(pred_means) == 0:
                    continue

                metrics = compute_direction_metrics(pred_means, gt_means)
                all_results[name] = {
                    'metrics': metrics,
                    'pred_means': pred_means,
                    'gt_means': gt_means,
                    'epoch': data.get('epoch', 0),
                    'tag': data.get('tag', ''),
                }

                # Log analysis summary
                analysis_text = analyze_learning_vs_centering(pred_means, gt_means)
                log.info(f"\n{name} Analysis:\n{analysis_text}")

                # Generate plots
                plot_prediction_vs_ground_truth(
                    pred_means, gt_means,
                    title=f'{name} (Epoch {epoch})',
                    output_path=os.path.join(analysis_dir, f'{name}_scatter.png')
                )

                plot_direction_analysis(
                    pred_means, gt_means,
                    title=f'{name} (Epoch {epoch})',
                    output_path=os.path.join(analysis_dir, f'{name}_direction.png')
                )

            except Exception as e:
                log.warning(f"Error analyzing {filepath}: {e}")

        # Generate comparison plot if we have both pretrain and current
        if len(all_results) >= 2:
            try:
                plot_comparison(
                    all_results,
                    output_path=os.path.join(analysis_dir, f'comparison_epoch_{epoch}.png')
                )
            except Exception as e:
                log.warning(f"Error generating comparison plot: {e}")

        log.info(f"Analysis figures saved to: {analysis_dir}")

    def save_validation_predictions(self, epoch: int, tag: str = "") -> None:
        """
        Run validation and save predictions. Used only for pre-training baseline.

        Args:
            epoch: Current epoch number (-1 for pre-training)
            tag: Optional tag to add to filename (e.g., "pretrain")
        """
        self._model.eval()

        all_predictions = []

        with torch.no_grad():
            for loader_idx, loader in enumerate(self._dataloader_v):
                if len(loader) <= 10:
                    continue

                for batch_idx, batch in enumerate(loader):
                    utils.batch_to_device(batch, self._device)
                    labels = batch.pop("labels")
                    ground_truth_bins = batch.pop("ground_truth_bins", None)
                    num_queries = batch.pop("num_queries", None)
                    batch.pop("mask", None)

                    if ground_truth_bins is None:
                        continue

                    # Run model
                    with self.activations_handling_ctx:
                        logits = self._model(**batch)

                    # Handle chunked logits
                    if isinstance(logits, list):
                        logits = torch.cat(logits, dim=1)

                    # Shift labels to match logit positions
                    labels_shifted = torch.hstack(
                        (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                    )

                    # Collect predictions
                    batch_preds = self._collect_batch_predictions(
                        logits, labels_shifted, ground_truth_bins, num_queries,
                        loader_idx, batch_idx
                    )
                    all_predictions.extend(batch_preds)

                    del logits

        # Save predictions with explicit filename for pre-training
        if tag:
            filename = f'val_predictions_epoch_{epoch}_{tag}.json'
        else:
            filename = f'val_predictions_epoch_{epoch}.json'
        self._save_predictions_to_file(all_predictions, epoch, filename)

        self._model.train()

    def compute_and_save_output_probabilities(self, epoch: int, tag: str = "") -> None:
        """
        Compute output probabilities for both numeric digit tokens and English word
        tokens at answer positions, then save to JSON.

        For each answer position, extracts:
        - numeric_probs: softmax over digit tokens "0"-"100" (101 values)
        - english_probs: softmax over first-tokens of English words "zero"-"one hundred" (101 values)

        For multi-token English words (e.g., "forty-five"), the probability is that of
        the first token only (an upper bound). Metadata indicates which words are single-token.

        Args:
            epoch: Current epoch (-1 for pre-training)
            tag: Tag for filename (e.g., "pretrain", "checkpoint")
        """
        self._model.eval()

        number_token_ids = self._number_token_ids.to(self._device)
        english_first_ids = self._english_number_info['first_token_ids'].to(self._device)

        # Create set of number token IDs for finding answer positions
        number_token_set = set(self._number_token_ids.tolist())

        all_examples = []

        with torch.no_grad():
            for loader_idx, loader in enumerate(self._dataloader_v):
                if len(loader) <= 10:
                    continue

                for batch_idx, batch in enumerate(loader):
                    utils.batch_to_device(batch, self._device)
                    labels = batch.pop("labels")
                    ground_truth_bins = batch.pop("ground_truth_bins", None)
                    num_queries = batch.pop("num_queries", None)
                    batch.pop("mask", None)

                    if ground_truth_bins is None:
                        continue

                    # Run model
                    with self.activations_handling_ctx:
                        logits = self._model(**batch)

                    # Handle chunked logits
                    if isinstance(logits, list):
                        logits = torch.cat(logits, dim=1)

                    # Shift labels to match logit positions
                    labels_shifted = torch.hstack(
                        (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                    )

                    batch_size = logits.size(0)

                    for i in range(batch_size):
                        n_queries = num_queries[i].item() if num_queries is not None else ground_truth_bins[i].size(0)
                        example_labels = labels_shifted[i]
                        example_logits = logits[i]  # [seq_len, vocab_size]

                        # Find answer positions (same logic as compute_distribution_loss)
                        answer_positions = []
                        for pos in range(example_labels.size(0)):
                            label_val = example_labels[pos].item()
                            if label_val == -100:
                                continue
                            if label_val in number_token_set:
                                answer_positions.append(pos)
                                continue
                            try:
                                decoded = self._tokenizer.decode([label_val]).strip()
                                if decoded.isdigit() and 0 <= int(decoded) <= 100:
                                    answer_positions.append(pos)
                            except Exception:
                                pass

                        queries_data = []
                        for q in range(min(n_queries, len(answer_positions))):
                            pos = answer_positions[q]
                            pos_logits = example_logits[pos]  # [vocab_size]

                            # Numeric token probabilities
                            numeric_logits = pos_logits[number_token_ids]  # [101]
                            numeric_probs = torch.softmax(numeric_logits, dim=0).cpu().tolist()

                            # English word first-token probabilities
                            english_logits = pos_logits[english_first_ids]  # [101]
                            english_probs = torch.softmax(english_logits, dim=0).cpu().tolist()

                            # Ground truth
                            gt = ground_truth_bins[i, q].cpu().tolist()

                            queries_data.append({
                                'query_idx': q,
                                'numeric_probs': numeric_probs,
                                'english_probs': english_probs,
                                'ground_truth': gt,
                            })

                        all_examples.append({
                            'loader_idx': loader_idx,
                            'batch_idx': batch_idx,
                            'example_idx': i,
                            'num_queries': n_queries,
                            'queries': queries_data,
                        })

                    del logits

        # Build metadata about English words
        english_metadata = {
            'words': self._english_number_info['words'],
            'is_single_token': self._english_number_info['is_single_token'],
            'first_token_words': self._english_number_info['first_token_words'],
        }

        # Save to file
        pred_dir = os.path.join(self._output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        filepath = os.path.join(pred_dir, f'output_probs_epoch_{epoch}{suffix}.json')

        output_data = {
            'epoch': epoch,
            'tag': tag,
            'english_word_metadata': english_metadata,
            'num_examples': len(all_examples),
            'examples': all_examples,
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        log.info(f"Saved output probabilities (numeric + English) to {filepath} ({len(all_examples)} examples)")

        self._model.train()

    def _compute_prob_metrics(self, dataloader) -> Tuple[float, float, float, float, float]:
        """
        Compute MAE and cross-entropy variants over a ProbabilisticReasoningDataset dataloader.

        Returns (mae_mean, ce_mode, ce_mean, ce_dist, mae_dist) where:
          mae_mean  - MAE between predicted and GT expected values (scalar)
          ce_mode   - cross-entropy in the current loss_mode (for backward compat)
          ce_mean   - cross-entropy against mean token in 101-token subspace
          ce_dist   - cross-entropy against full 101-bin distribution
          mae_dist  - L1 distance between predicted and GT distributions
        """
        all_mae = []
        all_ce  = []
        all_ce_mean = []
        all_ce_dist = []
        all_mae_dist = []

        self._model.eval()
        with torch.no_grad():
            for batch in dataloader:
                utils.batch_to_device(batch, self._device)
                tokens            = batch["tokens"]
                labels            = batch["labels"]
                ground_truth_bins = batch["ground_truth_bins"]
                num_queries       = batch.get("num_queries", None)

                with self.activations_handling_ctx:
                    logits = self._model(tokens=tokens)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)

                # Shift labels by 1 to align with logit positions
                B = labels.size(0)
                ignore_col = torch.full((B, 1), -100, dtype=labels.dtype, device=labels.device)
                labels_shifted = torch.cat([labels[:, 1:], ignore_col], dim=1)

                prob_metrics = self.evaluate_probabilistic_predictions(
                    logits, ground_truth_bins, num_queries, labels=labels_shifted
                )
                all_mae.append(prob_metrics['mean_abs_error'])
                all_ce.append(prob_metrics['cross_entropy'])
                all_ce_mean.append(prob_metrics['cross_entropy_mean'])
                all_ce_dist.append(prob_metrics['cross_entropy_dist'])
                all_mae_dist.append(prob_metrics['mean_abs_error_dist'])
                del logits

        self._model.train()
        mae      = float(np.mean(all_mae))      if all_mae      else 0.0
        ce       = float(np.mean(all_ce))       if all_ce       else 0.0
        ce_mean  = float(np.mean(all_ce_mean))  if all_ce_mean  else 0.0
        ce_dist  = float(np.mean(all_ce_dist))  if all_ce_dist  else 0.0
        mae_dist = float(np.mean(all_mae_dist)) if all_mae_dist else 0.0
        return mae, ce, ce_mean, ce_dist, mae_dist

    def _run_epoch_validation(self, curr_epoch: int, best_loss_v: float) -> float:
        """Run validation and test evaluation. Used both pre-training (epoch=-1) and per-epoch."""
        self._model.eval()

        # Initialize list to collect validation predictions
        self._current_val_predictions = []

        # Unseen data from unseen dataset
        for loader_idx, loader in enumerate(self._dataloader_v):
            if len(loader) <= 10:
                continue

            log.info(len(self._dataloader_test))

            with torch.no_grad():

                for idx, batch in enumerate(loader):

                    if self._pyro_mode and idx >= PYRO_VAL_BATCHES:
                        break

                    utils.batch_to_device(batch, self._device)
                    labels = batch.pop("labels")
                    ground_truth_bins = batch.pop("ground_truth_bins", None)
                    num_queries = batch.pop("num_queries", None)
                    batch.pop("mask", None)

                    with self.activations_handling_ctx:
                        logits = self._model(**batch)

                    # Handle chunked logits: convert to single tensor if needed
                    if isinstance(logits, list):
                        logits = torch.cat(logits, dim=1)

                    # Compute standard loss for LL
                    labels_shifted = torch.hstack(
                        (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                    )

                    # Debug: Print first validation example
                    if not self._debug_printed_val:
                        self._debug_printed_val = True
                        self._debug_print_example(
                            logits, labels_shifted, "VAL", self._tokenizer,
                            ground_truth_bins=ground_truth_bins,
                            num_queries=num_queries
                        )

                    ignore_index = self._loss_fn.ignore_index
                    total_elements = (labels_shifted != ignore_index).sum()
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels_shifted.reshape(-1),
                        ignore_index=ignore_index,
                    )
                    ll = -(loss * total_elements).detach().float().cpu().numpy()
                    self.curr_ll_v.append(ll)

                    # Probabilistic evaluation (use shifted labels to match logit positions)
                    if ground_truth_bins is not None:
                        prob_metrics = self.evaluate_probabilistic_predictions(
                            logits, ground_truth_bins, num_queries, labels=labels_shifted
                        )
                        self.curr_kl_v.append(prob_metrics['kl_divergence'])
                        self.curr_mae_v.append(prob_metrics['mean_abs_error'])
                        self.curr_mae_dist_v.append(prob_metrics['mean_abs_error_dist'])
                        self.curr_pred_means_v.extend(prob_metrics['pred_means'])
                        self.curr_gt_means_v.extend(prob_metrics['gt_means'])
                        self.curr_acc_v.append(0)  # Placeholder
                        self.curr_ce_mean_v.append(prob_metrics['cross_entropy_mean'])
                        self.curr_ce_dist_v.append(prob_metrics['cross_entropy_dist'])

                        # Collect predictions for saving
                        batch_preds = self._collect_batch_predictions(
                            logits, labels_shifted, ground_truth_bins, num_queries,
                            loader_idx, idx
                        )
                        self._current_val_predictions.extend(batch_preds)
                    else:
                        self.curr_kl_v.append(0)
                        self.curr_mae_v.append(0)
                        self.curr_mae_dist_v.append(0)
                        self.curr_acc_v.append(0)
                        self.curr_ce_mean_v.append(0)
                        self.curr_ce_dist_v.append(0)

                    del logits

                for idx, batch in enumerate(self._dataloader_test[loader_idx]):

                    if self._pyro_mode and idx >= PYRO_VAL_BATCHES:
                        break
                    if batch['labels'].shape[-1] >= LENGTH_CUTOFF:
                        continue
                    utils.batch_to_device(batch, self._device)
                    labels = batch.pop("labels")
                    ground_truth_bins = batch.pop("ground_truth_bins", None)
                    num_queries = batch.pop("num_queries", None)
                    batch.pop("mask", None)

                    # run model
                    with self.activations_handling_ctx:
                        logits = self._model(**batch)

                    # Handle chunked logits: convert to single tensor if needed
                    if isinstance(logits, list):
                        logits = torch.cat(logits, dim=1)

                    # Compute standard loss for LL
                    labels_shifted = torch.hstack(
                        (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
                    )

                    # Debug: Print first test example
                    if not self._debug_printed_test:
                        self._debug_printed_test = True
                        self._debug_print_example(
                            logits, labels_shifted, "TEST", self._tokenizer,
                            ground_truth_bins=ground_truth_bins,
                            num_queries=num_queries
                        )

                    ignore_index = self._loss_fn.ignore_index
                    total_elements = (labels_shifted != ignore_index).sum()
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels_shifted.reshape(-1),
                        ignore_index=ignore_index,
                    )
                    ll = -(loss * total_elements).detach().float().cpu().numpy()
                    self.curr_ll_test.append(ll)

                    # Probabilistic evaluation (use shifted labels to match logit positions)
                    if ground_truth_bins is not None:
                        prob_metrics = self.evaluate_probabilistic_predictions(
                            logits, ground_truth_bins, num_queries, labels=labels_shifted
                        )
                        self.curr_kl_test.append(prob_metrics['kl_divergence'])
                        self.curr_mae_test.append(prob_metrics['mean_abs_error'])
                        self.curr_mae_dist_test.append(prob_metrics['mean_abs_error_dist'])
                        self.curr_pred_means_test.extend(prob_metrics['pred_means'])
                        self.curr_gt_means_test.extend(prob_metrics['gt_means'])
                        self.curr_acc_test.append(0)  # Placeholder
                        self.curr_ce_mean_test.append(prob_metrics['cross_entropy_mean'])
                        self.curr_ce_dist_test.append(prob_metrics['cross_entropy_dist'])
                    else:
                        self.curr_kl_test.append(0)
                        self.curr_mae_test.append(0)
                        self.curr_mae_dist_test.append(0)
                        self.curr_acc_test.append(0)
                        self.curr_ce_mean_test.append(0)
                        self.curr_ce_dist_test.append(0)

                    del logits

            torch.cuda.empty_cache()

        # Save current validation predictions (overwrites previous)
        if self._current_val_predictions:
            self._save_predictions_to_file(
                self._current_val_predictions, curr_epoch
            )

        self._model.to(torch.device(self._device))
        self._model.train()

        torch.cuda.empty_cache()

        if curr_epoch >= 0:
            self.epochs_run += 1
        log.info('')
        log.info(f'----------- END OF EPOCH {curr_epoch} -----------')
        log.info('')
        log.info(f'TRAIN ACCURACY: {np.mean(self.curr_acc) if self.curr_acc else 0}')
        log.info(f'VALIDATION LL: {np.mean(self.curr_ll_v) if self.curr_ll_v else 0}')
        log.info(f'TEST LL: {np.mean(self.curr_ll_test) if self.curr_ll_test else 0}')
        log.info(f'VALIDATION KL: {np.mean(self.curr_kl_v) if self.curr_kl_v else 0}')
        log.info(f'TEST KL: {np.mean(self.curr_kl_test) if self.curr_kl_test else 0}')
        log.info(f'VALIDATION MAE (mean): {np.mean(self.curr_mae_v) if self.curr_mae_v else 0}')
        log.info(f'TEST MAE (mean): {np.mean(self.curr_mae_test) if self.curr_mae_test else 0}')
        log.info(f'VALIDATION MAE (dist L1): {np.mean(self.curr_mae_dist_v) if self.curr_mae_dist_v else 0}')
        log.info(f'TEST MAE (dist L1): {np.mean(self.curr_mae_dist_test) if self.curr_mae_dist_test else 0}')
        log.info(f'VALIDATION CE (mean GT): {np.mean(self.curr_ce_mean_v) if self.curr_ce_mean_v else 0}')
        log.info(f'TEST CE (mean GT): {np.mean(self.curr_ce_mean_test) if self.curr_ce_mean_test else 0}')
        log.info(f'VALIDATION CE (full dist): {np.mean(self.curr_ce_dist_v) if self.curr_ce_dist_v else 0}')
        log.info(f'TEST CE (full dist): {np.mean(self.curr_ce_dist_test) if self.curr_ce_dist_test else 0}')
        pearson_v = float(np.corrcoef(self.curr_pred_means_v, self.curr_gt_means_v)[0, 1]) if len(self.curr_pred_means_v) >= 2 else 0.0
        pearson_test = float(np.corrcoef(self.curr_pred_means_test, self.curr_gt_means_test)[0, 1]) if len(self.curr_pred_means_test) >= 2 else 0.0
        log.info(f'VALIDATION PEARSON: {pearson_v}')
        log.info(f'TEST PEARSON: {pearson_test}')
        log.info('')
        log.info(f'----------- END OF EPOCH {curr_epoch} -----------')
        log.info('')

        self.accs.append(np.mean(self.curr_acc) if self.curr_acc else 0)
        self.accs_v.append(np.mean(self.curr_acc_v) if self.curr_acc_v else 0)
        self.accs_test.append(np.mean(self.curr_acc_test) if self.curr_acc_test else 0)
        np.savetxt(self.filename +'_accs_train.csv', np.array(self.accs))
        np.savetxt(self.filename +'_accs_val.csv', np.array(self.accs_v))
        np.savetxt(self.filename +'_accs_test.csv', np.array(self.accs_test))
        self.curr_acc = []
        self.curr_acc_v = []
        self.curr_acc_test = []


        self.lls.append(np.mean(self.curr_ll) if self.curr_ll else 0)
        self.lls_v.append(np.mean(self.curr_ll_v) if self.curr_ll_v else 0)
        self.lls_test.append(np.mean(self.curr_ll_test) if self.curr_ll_test else 0)
        np.savetxt(self.filename +'_lls_train.csv', np.array(self.lls))
        np.savetxt(self.filename +'_lls_val.csv', np.array(self.lls_v))
        np.savetxt(self.filename +'_lls_test.csv', np.array(self.lls_test))
        self.curr_ll = []
        self.curr_ll_v = []
        self.curr_ll_test = []

        # Save probabilistic metrics
        self.kls_v.append(np.mean(self.curr_kl_v) if self.curr_kl_v else 0)
        self.kls_test.append(np.mean(self.curr_kl_test) if self.curr_kl_test else 0)
        np.savetxt(self.filename +'_kls_val.csv', np.array(self.kls_v))
        np.savetxt(self.filename +'_kls_test.csv', np.array(self.kls_test))
        self.curr_kl_v = []
        self.curr_kl_test = []

        curr_mae_v_mean = np.mean(self.curr_mae_v) if self.curr_mae_v else float('inf')
        self.maes_v.append(curr_mae_v_mean)
        self.maes_test.append(np.mean(self.curr_mae_test) if self.curr_mae_test else 0)
        np.savetxt(self.filename +'_maes_val.csv', np.array(self.maes_v))
        np.savetxt(self.filename +'_maes_test.csv', np.array(self.maes_test))
        self.curr_mae_v = []
        self.curr_mae_test = []

        self.maes_dist_v.append(np.mean(self.curr_mae_dist_v) if self.curr_mae_dist_v else 0)
        self.maes_dist_test.append(np.mean(self.curr_mae_dist_test) if self.curr_mae_dist_test else 0)
        np.savetxt(self.filename +'_maes_dist_val.csv', np.array(self.maes_dist_v))
        np.savetxt(self.filename +'_maes_dist_test.csv', np.array(self.maes_dist_test))
        self.curr_mae_dist_v = []
        self.curr_mae_dist_test = []

        self.ces_mean_v.append(np.mean(self.curr_ce_mean_v) if self.curr_ce_mean_v else 0)
        self.ces_mean_test.append(np.mean(self.curr_ce_mean_test) if self.curr_ce_mean_test else 0)
        np.savetxt(self.filename +'_ces_mean_val.csv', np.array(self.ces_mean_v))
        np.savetxt(self.filename +'_ces_mean_test.csv', np.array(self.ces_mean_test))
        self.curr_ce_mean_v = []
        self.curr_ce_mean_test = []

        self.ces_dist_v.append(np.mean(self.curr_ce_dist_v) if self.curr_ce_dist_v else 0)
        self.ces_dist_test.append(np.mean(self.curr_ce_dist_test) if self.curr_ce_dist_test else 0)
        np.savetxt(self.filename +'_ces_dist_val.csv', np.array(self.ces_dist_v))
        np.savetxt(self.filename +'_ces_dist_test.csv', np.array(self.ces_dist_test))
        self.curr_ce_dist_v = []
        self.curr_ce_dist_test = []

        self.pearsons_v.append(pearson_v)
        self.pearsons_test.append(pearson_test)
        np.savetxt(self.filename + '_pearsons_val.csv', np.array(self.pearsons_v))
        np.savetxt(self.filename + '_pearsons_test.csv', np.array(self.pearsons_test))
        self.curr_pred_means_v = []
        self.curr_gt_means_v = []
        self.curr_pred_means_test = []
        self.curr_gt_means_test = []

        # Probabilistic-reasoning val evaluation — only run when a dedicated
        # prob_val_json loader is configured (forward-sampling runs).  For
        # probabilistic-dataset runs the main val loop above already covered
        # self._dataloader_v[0], so a second pass would be redundant.
        _prob_eval_loader = self._dataloader_prob_val
        if _prob_eval_loader is not None:
            mean_prob_mae, mean_prob_ce, mean_prob_ce_mean, _, mean_prob_mae_dist = \
                self._compute_prob_metrics(_prob_eval_loader)
            self.prob_maes_v.append(mean_prob_mae)
            self.prob_ces_dist_v.append(mean_prob_ce)
            self.prob_ces_mean_v.append(mean_prob_ce_mean)
            self.prob_maes_dist_v.append(mean_prob_mae_dist)
            np.savetxt(self.filename + '_prob_maes_val.csv',      np.array(self.prob_maes_v))
            np.savetxt(self.filename + '_prob_ces_dist_val.csv',  np.array(self.prob_ces_dist_v))
            np.savetxt(self.filename + '_prob_ces_mean_val.csv',  np.array(self.prob_ces_mean_v))
            np.savetxt(self.filename + '_prob_maes_dist_val.csv', np.array(self.prob_maes_dist_v))
            log.info(f'PROB-VAL MAE (mean): {mean_prob_mae:.3f}  MAE (dist): {mean_prob_mae_dist:.4f}')
            log.info(f'PROB-VAL CE (dist): {mean_prob_ce:.4f}  CE (mean GT): {mean_prob_ce_mean:.4f}')

        # WebPPL val evaluation
        if self._dataloader_webppl_val is not None:
            mae, ce, ce_mean, _, mae_dist = self._compute_prob_metrics(self._dataloader_webppl_val)
            self.webppl_maes_v.append(mae)
            self.webppl_ces_dist_v.append(ce)
            self.webppl_ces_mean_v.append(ce_mean)
            self.webppl_maes_dist_v.append(mae_dist)
            np.savetxt(self.filename + '_webppl_maes_val.csv',      np.array(self.webppl_maes_v))
            np.savetxt(self.filename + '_webppl_ces_dist_val.csv',  np.array(self.webppl_ces_dist_v))
            np.savetxt(self.filename + '_webppl_ces_mean_val.csv',  np.array(self.webppl_ces_mean_v))
            np.savetxt(self.filename + '_webppl_maes_dist_val.csv', np.array(self.webppl_maes_dist_v))
            log.info(f'WEBPPL-VAL MAE (mean): {mae:.3f}  MAE (dist): {mae_dist:.4f}')
            log.info(f'WEBPPL-VAL CE (dist): {ce:.4f}  CE (mean GT): {ce_mean:.4f}')

        # WebPPL test evaluation
        if self._dataloader_webppl_test is not None:
            mae, ce, ce_mean, _, mae_dist = self._compute_prob_metrics(self._dataloader_webppl_test)
            self.webppl_maes_test.append(mae)
            self.webppl_ces_dist_test.append(ce)
            self.webppl_ces_mean_test.append(ce_mean)
            self.webppl_maes_dist_test.append(mae_dist)
            np.savetxt(self.filename + '_webppl_maes_test.csv',      np.array(self.webppl_maes_test))
            np.savetxt(self.filename + '_webppl_ces_dist_test.csv',  np.array(self.webppl_ces_dist_test))
            np.savetxt(self.filename + '_webppl_ces_mean_test.csv',  np.array(self.webppl_ces_mean_test))
            np.savetxt(self.filename + '_webppl_maes_dist_test.csv', np.array(self.webppl_maes_dist_test))
            log.info(f'WEBPPL-TEST MAE (mean): {mae:.3f}  MAE (dist): {mae_dist:.4f}')
            log.info(f'WEBPPL-TEST CE (dist): {ce:.4f}  CE (mean GT): {ce_mean:.4f}')

        # Probabilistic-reasoning test evaluation (pyro mode and any run with _dataloader_prob_test)
        if self._dataloader_prob_test is not None:
            mae, ce, ce_mean, _, mae_dist = self._compute_prob_metrics(self._dataloader_prob_test)
            self.prob_maes_test.append(mae)
            self.prob_ces_dist_test.append(ce)
            self.prob_ces_mean_test.append(ce_mean)
            self.prob_maes_dist_test.append(mae_dist)
            np.savetxt(self.filename + '_prob_maes_test.csv',      np.array(self.prob_maes_test))
            np.savetxt(self.filename + '_prob_ces_dist_test.csv',  np.array(self.prob_ces_dist_test))
            np.savetxt(self.filename + '_prob_ces_mean_test.csv',  np.array(self.prob_ces_mean_test))
            np.savetxt(self.filename + '_prob_maes_dist_test.csv', np.array(self.prob_maes_dist_test))
            log.info(f'PROB-TEST MAE (mean): {mae:.3f}  MAE (dist): {mae_dist:.4f}')
            log.info(f'PROB-TEST CE (dist): {ce:.4f}  CE (mean GT): {ce_mean:.4f}')

        # Save checkpoint if validation cross-entropy loss improved (lower is better)
        if curr_epoch >= 0:
            # Prefer webppl CE (pyro mode), then prob CE (forward-sampling), then pyro val CE
            if self.webppl_ces_dist_v:
                curr_loss_v_mean = self.webppl_ces_dist_v[-1]
            elif self.prob_ces_dist_v:
                curr_loss_v_mean = self.prob_ces_dist_v[-1]
            else:
                curr_loss_v_mean = self.ces_dist_v[-1] if self.ces_dist_v else float('inf')
            if curr_loss_v_mean < best_loss_v:
                log.info(f"Validation loss improved: {best_loss_v:.4f} -> {curr_loss_v_mean:.4f}")
                log.info(f"Saving best checkpoint at epoch {curr_epoch}...")
                best_loss_v = curr_loss_v_mean
                self.save_checkpoint(epoch=curr_epoch)
                self._copy_predictions_to_checkpoint(epoch=curr_epoch)
                # Save output probabilities (numeric + English) at checkpoint
                log.info(f"Saving output probabilities at epoch {curr_epoch}...")
                self.compute_and_save_output_probabilities(epoch=curr_epoch, tag="checkpoint")
                # Run prediction analysis when checkpointing and epoch > 10
                if True:
                    log.info(f"Running prediction analysis at epoch {curr_epoch}...")
                    self.run_prediction_analysis(epoch=curr_epoch)
                log.info(f"Best checkpoint saved for epoch {curr_epoch}")
            else:
                log.info(f"Validation loss did not improve: {curr_loss_v_mean:.4f} (best: {best_loss_v:.4f})")

        return best_loss_v

    def train(self) -> None:
        """
        The core training loop.
        """

        if self._compile:
            log.info(
                "NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration."
            )

        running_loss = 0
        num_tokens = 0
        saved_losses = []

        self.curr_acc = []
        self.curr_acc_v = []
        self.curr_acc_test = []
        self.accs = []
        self.accs_v = []
        self.accs_test = []

        self.curr_acc_test_prune = []
        self.accs_test_prune = []


        self.curr_ll = []
        self.curr_ll_v = []
        self.curr_ll_test = []
        self.lls = []
        self.lls_v = []
        self.lls_test = []

        # Probabilistic evaluation metrics
        self.curr_kl_v = []
        self.curr_kl_test = []
        self.kls_v = []
        self.kls_test = []
        self.curr_mae_v = []
        self.curr_mae_test = []
        self.maes_v = []
        self.maes_test = []
        self.curr_mae_dist_v = []
        self.curr_mae_dist_test = []
        self.maes_dist_v = []
        self.maes_dist_test = []
        self.curr_ce_mean_v = []
        self.curr_ce_mean_test = []
        self.ces_mean_v = []
        self.ces_mean_test = []
        self.curr_ce_dist_v = []
        self.curr_ce_dist_test = []
        self.ces_dist_v = []
        self.ces_dist_test = []
        self.curr_pred_means_v = []
        self.curr_gt_means_v = []
        self.curr_pred_means_test = []
        self.curr_gt_means_test = []
        self.pearsons_v = []
        self.pearsons_test = []

        # Probabilistic-reasoning val metrics (populated when prob_val_json is set)
        self.prob_maes_v = []
        self.prob_ces_dist_v = []
        self.prob_ces_mean_v = []
        self.prob_maes_dist_v = []

        # Probabilistic-reasoning test metrics (populated in pyro mode)
        self.prob_maes_test = []
        self.prob_ces_dist_test = []
        self.prob_ces_mean_test = []
        self.prob_maes_dist_test = []

        # WebPPL val/test metrics (populated when webppl_val/test_json are set)
        self.webppl_maes_v = []
        self.webppl_ces_dist_v = []
        self.webppl_ces_mean_v = []
        self.webppl_maes_dist_v = []
        self.webppl_maes_test = []
        self.webppl_ces_dist_test = []
        self.webppl_ces_mean_test = []
        self.webppl_maes_dist_test = []

        best_acc = 0
        best_mae_v = float('inf')  # Track best validation MAE (lower is better)
        best_loss_v = float('inf')  # Track best validation cross-entropy loss (lower is better)

        # Debug flags to print first example only once
        self._debug_printed_train = False
        self._debug_printed_val = False
        self._debug_printed_test = False

        if NOTRAIN:
            self.total_epochs = 1

        # Save pre-training predictions (before any fine-tuning)
        log.info("Saving pre-training validation predictions...")
        self.save_validation_predictions(epoch=-1, tag="pretrain")

        # Save pre-training output probabilities (numeric + English word tokens)
        log.info("Saving pre-training output probabilities...")
        self.compute_and_save_output_probabilities(epoch=-1, tag="pretrain")

        # Run full validation epoch before any training (epoch -1 = pre-training baseline)
        log.info("Running pre-training validation epoch...")
        best_loss_v = self._run_epoch_validation(-1, best_loss_v)

        # Persistent iterators — survive across epochs so that when the dataloader
        # is longer than MAX_BATCHES_PER_EPOCH we resume from where we left off
        # rather than restarting from the beginning each epoch.
        MAX_BATCHES_PER_EPOCH = 200

        # Phase 1 (probabilistic) iterator
        _dataloader_len_prob = len(self._dataloaderlist[0])
        _cap_batches_prob = _dataloader_len_prob > MAX_BATCHES_PER_EPOCH
        _iter_epoch_prob = 0
        for i in range(len(self._dataloaderlist)):
            self._samplerlist[i].set_epoch(_iter_epoch_prob)
        iters_prob = [iter(ds) for ds in self._dataloaderlist]

        # Phase 2 (forward) iterator — only in fusion mode
        if self._is_fusion:
            _dataloader_len_fwd = len(self._dataloaderlist_forward[0])
            _cap_batches_fwd = _dataloader_len_fwd > MAX_BATCHES_PER_EPOCH
            _iter_epoch_fwd = 0
            for i in range(len(self._dataloaderlist_forward)):
                self._samplerlist_forward[i].set_epoch(_iter_epoch_fwd)
            iters_fwd = [iter(ds) for ds in self._dataloaderlist_forward]

        self._model.train()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(0, self.total_epochs):

            # --- Fusion phase selection ---
            in_forward_phase = self._is_fusion and curr_epoch >= self._prob_epochs
            if self._is_fusion and curr_epoch == self._prob_epochs:
                log.info(
                    f"=== FUSION TRAINING: switching to forward-sampling phase at epoch {curr_epoch} "
                    f"(prob phase: 0-{self._prob_epochs - 1}, forward phase: {self._prob_epochs}-{self.total_epochs - 1}) ==="
                )

            if in_forward_phase:
                active_iters = iters_fwd
                active_samplers = self._samplerlist_forward
                active_dataloaderlist = self._dataloaderlist_forward
                _dataloader_len = _dataloader_len_fwd
                _cap_batches = _cap_batches_fwd
            else:
                active_iters = iters_prob
                active_samplers = self._samplerlist
                active_dataloaderlist = self._dataloaderlist
                _dataloader_len = _dataloader_len_prob
                _cap_batches = _cap_batches_prob

            self._optimizer.zero_grad()

            if not NOTRAIN:
                # Each epoch consumes at most MAX_BATCHES_PER_EPOCH batches from
                # the persistent iterator.  When the iterator is exhausted it is
                # reset (and reshuffled via set_epoch) before pulling the next batch.
                n_batches = MAX_BATCHES_PER_EPOCH if _cap_batches else _dataloader_len

                for idx in range(n_batches):
                    try:
                        batch = next(active_iters[0])
                    except StopIteration:
                        if in_forward_phase:
                            _iter_epoch_fwd += 1
                            active_samplers[0].set_epoch(_iter_epoch_fwd)
                        else:
                            _iter_epoch_prob += 1
                            active_samplers[0].set_epoch(_iter_epoch_prob)
                        active_iters[0] = iter(active_dataloaderlist[0])
                        batch = next(active_iters[0])

                    if (
                        self.max_steps_per_epoch is not None
                        and (idx // self._gradient_accumulation_steps)
                        == self.max_steps_per_epoch
                    ):
                        break

                    # Start tracking CUDA memory for active steps for just the first epoch
                    if (
                        curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history()

                    utils.batch_to_device(batch, self._device)

                    # Calculate the number of unmasked tokens in the current batch
                    # and increment the total number of tokens seen in the step
                    current_num_tokens = (
                        batch["labels"] != self._loss_fn.ignore_index
                    ).sum()
                    num_tokens += current_num_tokens

                    # Loss is normalized by default so we multiply by the number of tokens
                    # This way we can normalize by the total number of tokens if we're accumulating gradients
                    current_loss, metrics = self._loss_step(batch, self._model)
                    current_loss = current_loss * current_num_tokens
                    current_loss.backward()
                    self._optimizer.step()
                    peak_memory = torch.cuda.max_memory_allocated()
                    log.info(f"PEAK MEMORY: Peak GPU memory allocated so far: {peak_memory / (1024**2):.2f} MB")
                    self._optimizer.zero_grad()

                    saved_losses.append(current_loss.detach().cpu().numpy())
                    self.curr_acc.append(metrics[0])
                    self.curr_ll.append(metrics[2])
                    if np.sum(np.isnan(saved_losses)) == 0:
                        np.savetxt(self.filename +'_innerloss.csv', np.array(saved_losses))
                        

                    # Reset running stats for the next step
                    num_tokens = 0
                    ##### END OF INNER LOOPS
                torch.cuda.empty_cache()  

                ##### OUTER LOOP BEGINS HERE
                # Stop tracking CUDA memory now that active steps are complete
                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history(enabled=None)


            ##### EVAL #####
            best_loss_v = self._run_epoch_validation(curr_epoch, best_loss_v)

        # Log final training summary (checkpoint already saved when loss improved)
        log.info(f"Training complete. Best validation loss achieved: {best_loss_v:.4f}")
        log.info(f"Best model saved to epoch_0 folder (overrides previous checkpoints)")



    def cleanup(self) -> None:
        self._metric_logger.close()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneRecipeSingleDevice", cfg=cfg)

    global MODEL
    global DATA_PROPERTY
    global NOTRAIN
    global LENGTH_CUTOFF
    global metaeval_steps
    global RUN_PRUNING
    global tokenizer
    global DATASET

    is_pyro_fusion = cfg.get("pyro_fusion_epochs", None) is not None and cfg.get("pyro_train_json", None) is not None and cfg.get("forward_dataset", None) is not None
    is_fusion = cfg.get("prob_epochs", None) is not None and cfg.get("forward_epochs", None) is not None
    is_pyro = cfg.get("pyro_train_json", None) is not None
    DATASET = 'pyro_fusion' if is_pyro_fusion else ('fusion' if is_fusion else ('pyro' if is_pyro else str(cfg.dataset._component_).split('_')[1]))

    if 'llama' in str(cfg.model._component_):
        MODEL = 'llama'
        model_path = "<DATA_ROOT>/resources/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
    elif 'qwen' in str(cfg.model._component_):
        MODEL = 'qwen'
        model_path = "<DATA_ROOT>/resources/qwen/Qwen2-7B-Instruct"

    DATA_PROPERTY = '' 
    NOTRAIN = False
    LENGTH_CUTOFF = 1500
    metaeval_steps = cfg.metaeval_steps
    RUN_PRUNING = False
    loss_mode = cfg.get("loss_mode", "distribution")
    loss_mode_suffix = "_mean_only" if loss_mode == "mean_only" else "_distribution"
    sft_suffix = "_sft"
    if "lora" in str(cfg.model._component_):
        tune_suffix = f"_lora_r{cfg.model.lora_rank}"
    else:
        tune_suffix = "_full"
    if not NOTRAIN:
        filename = f'{MODEL}-{DATASET}{DATA_PROPERTY}-normal-5t10neg5-stepeval{metaeval_steps}_{RUN_PRUNING}{loss_mode_suffix}{sft_suffix}{tune_suffix}'
    else:
        filename = f'{MODEL}-{DATASET}{DATA_PROPERTY}-notrain-1t10neg5-stepeval{metaeval_steps}_{RUN_PRUNING}{loss_mode_suffix}{sft_suffix}{tune_suffix}'

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    global log
    global FILENAME 
    FILENAME = filename + f'-seed{cfg.seed}'
    log = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{FILENAME}.log', encoding='utf-8', level=logging.DEBUG)
    log.debug('This message should go to the log file')
    log.info('So should this')
    log.warning('And this, too')


    recipe = FullFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
