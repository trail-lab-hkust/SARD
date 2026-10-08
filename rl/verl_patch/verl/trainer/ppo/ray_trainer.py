# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import shutil
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import subprocess
import shutil

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask, masked_mean, postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kl_mask = response_mask
    if "off_policy_mask" in data.batch.keys():
        kl_mask = response_mask * (~data.batch["off_policy_mask"].bool()).to(response_mask.dtype)
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * kl_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=kl_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        project_dir=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.project_dir = project_dir

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        dataset_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # Get lambda parameters from config
            lambda_init = self.config.algorithm.get('lambda_init', 0.8)
            lambda_final = self.config.algorithm.get('lambda_final', 0.0)
            # Calculate total_steps if not already set
            if not hasattr(self, 'total_training_steps'):
                self.total_training_steps = self.config.trainer.total_epochs * len(self.train_dataloader)
            result = self.val_reward_fn(
                test_batch, return_dict=True,
                global_step=self.global_steps,
                total_steps=self.total_training_steps,
                lambda_init=lambda_init,
                lambda_final=lambda_final
            )
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            dataset_lst.append(test_batch.non_tensor_batch.get("dataset", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        datasets = np.concatenate(dataset_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        dataset2var2metric2val = process_validation_metrics(datasets, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        for dataset_name, var2metric2val in dataset2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/dataset/{dataset_name}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        export_lora_only = bool(self.config.trainer.get("export_lora_only", False))
        is_lora_training = self.config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        if export_lora_only and is_lora_training:
            target_dir = os.path.join(self.config.trainer.default_local_dir, f"checkpoint-{self.global_steps}")
            target_lora_dir = os.path.join(target_dir, "lora_adapter")

            os.makedirs(target_dir, exist_ok=True)
            self.actor_rollout_wg.save_lora_adapter(target_dir)

            adapter_config_path = os.path.join(target_lora_dir, "adapter_config.json")
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            target_modules = adapter_config.get("target_modules")
            if isinstance(target_modules, list) and target_modules == list("all-linear"):
                adapter_config["target_modules"] = "all-linear"
            adapter_config["base_model_name_or_path"] = self.config.actor_rollout_ref.model.path
            with open(adapter_config_path, "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, ensure_ascii=False, indent=4)

            print(f"Exported LoRA-only checkpoint to {target_lora_dir}")
            return

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        ################### transform model ckpt ###################
        print('transformering the checkpoint to hf format...')
        try:
            target_dir = os.path.join(self.config.trainer.default_local_dir, f'checkpoint-{self.global_steps}')
            subprocess.run(
                ["bash", f"{self.project_dir}/verl/scripts/merge.sh", actor_local_path, self.config.actor_rollout_ref.model.path, target_dir],
                check=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"transforming error: {e}")
        # remove the previously saved actor checkpoint
        try:
            shutil.rmtree(local_global_step_folder)  # 递归删除目录及其内容
            print(f"{local_global_step_folder} deleted!")
            if os.path.exists(local_latest_checkpointed_iteration):
                os.remove(local_latest_checkpointed_iteration)
                print(f"{local_latest_checkpointed_iteration} deleted!")
        except OSError as e:
            print(f"删除失败: {e}")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _guided_grpo_cfg(self):
        return self.config.algorithm.get("guided_grpo", {}) or {}

    def _guided_grpo_enabled(self):
        return bool(self._guided_grpo_cfg().get("enable", False))

    def _guided_direct_teacher_fallback_enabled(self):
        return self._guided_grpo_enabled() and bool(self._guided_grpo_cfg().get("direct_teacher_fallback", False))

    def _guided_train_strategy_groups_enabled(self):
        cfg = self._guided_grpo_cfg()
        if "train_auxiliary_groups" in cfg:
            return bool(cfg.get("train_auxiliary_groups"))
        return bool(cfg.get("train_strategy_groups", True))

    def _guided_auxiliary_group_loss_weight(self):
        cfg = self._guided_grpo_cfg()
        if "auxiliary_group_loss_weight" in cfg:
            return float(cfg.get("auxiliary_group_loss_weight"))
        return 1.0

    def _guided_bridge_success_to_base_enabled(self):
        cfg = self._guided_grpo_cfg()
        return bool(cfg.get("bridge_success_to_base", True))

    def _guided_teacher_fallback_aux_group(self):
        cfg = self._guided_grpo_cfg()
        if "teacher_fallback_aux_group" in cfg:
            return str(cfg.get("teacher_fallback_aux_group") or "none")
        return "current_strategy" if self._guided_train_strategy_groups_enabled() else "none"

    def _guided_repair_cfg(self):
        return self._guided_grpo_cfg().get("repair", {}) or {}

    def _guided_repair_enabled(self):
        repair_cfg = self._guided_repair_cfg()
        max_rounds = int(repair_cfg.get("max_rounds", self._guided_grpo_cfg().get("max_repair_rounds", 0)) or 0)
        return bool(repair_cfg.get("enable", False)) and max_rounds > 0

    def _guided_repair_max_rounds(self):
        repair_cfg = self._guided_repair_cfg()
        return int(repair_cfg.get("max_rounds", self._guided_grpo_cfg().get("max_repair_rounds", 0)) or 0)

    def _guided_question_dump_cfg(self):
        return self._guided_grpo_cfg().get("batch_dump", {}) or {}

    def _guided_advantage_bucket_cfg(self):
        return self._guided_grpo_cfg().get("advantage_bucket", {}) or {}

    def _guided_advantage_bucket_enabled(self):
        return self._guided_grpo_enabled() and bool(self._guided_advantage_bucket_cfg().get("enable", False))

    def _guided_question_dump_enabled(self):
        return bool(self._guided_question_dump_cfg().get("enable", False))

    def _guided_question_dump_should_write(self):
        if not self._guided_question_dump_enabled():
            return False
        every_n_steps = int(self._guided_question_dump_cfg().get("every_n_steps", 1) or 1)
        return every_n_steps > 0 and int(self.global_steps) % every_n_steps == 0

    def _guided_question_dump_dir(self):
        default_dir = os.path.join(str(self.config.trainer.default_local_dir), "guided_question_dumps")
        return str(self._guided_question_dump_cfg().get("dump_dir", default_dir) or default_dir)

    @staticmethod
    def _jsonable(value):
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, np.ndarray):
            return [RayPPOTrainer._jsonable(item) for item in value.tolist()]
        if isinstance(value, torch.Tensor):
            return RayPPOTrainer._jsonable(value.detach().cpu().tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): RayPPOTrainer._jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [RayPPOTrainer._jsonable(item) for item in value]
        return value

    @staticmethod
    def _as_messages(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            return None
        messages = []
        for message in value:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                return None
            messages.append({"role": str(message["role"]), "content": str(message["content"])})
        return messages

    @staticmethod
    def _last_user_content(messages):
        messages = RayPPOTrainer._as_messages(messages)
        if not messages:
            return ""
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return messages[-1].get("content", "")

    @staticmethod
    def _truncate_text(text, max_chars):
        text = "" if text is None else str(text)
        if max_chars is None:
            return text
        max_chars = int(max_chars)
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        keep_head = max_chars // 2
        keep_tail = max_chars - keep_head
        return text[:keep_head] + "\n...[truncated]...\n" + text[-keep_tail:]

    @staticmethod
    def _find_json_object_span(text: str, start_pos: int = 0):
        start = text.find("{", start_pos)
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return start, idx + 1
        return None

    @staticmethod
    def _extract_strategy_card_from_messages(messages):
        user_content = RayPPOTrainer._last_user_content(messages)
        marker = "Structured ranking criterion:"
        marker_pos = user_content.find(marker)
        search_pos = marker_pos + len(marker) if marker_pos >= 0 else 0
        span = RayPPOTrainer._find_json_object_span(user_content, search_pos)
        if span is None:
            return {}
        try:
            return json.loads(user_content[span[0] : span[1]])
        except Exception:
            return {}

    @staticmethod
    def _extract_strategy_guidance_from_messages(messages):
        user_content = RayPPOTrainer._last_user_content(messages)
        match = re.search(r"<strategy_guidance>(.*?)</strategy_guidance>", user_content or "", flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _render_strategy_guidance(strategy_card):
        if not isinstance(strategy_card, dict):
            return str(strategy_card or "")

        def lines_for(key):
            value = strategy_card.get(key, [])
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                value = [str(value)]
            return "\n".join(f"- {item}" for item in value if str(item).strip())

        procedure = strategy_card.get("ranking_procedure", [])
        if isinstance(procedure, str):
            procedure = [procedure]
        elif not isinstance(procedure, (list, tuple)):
            procedure = [str(procedure)]
        procedure_text = "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(procedure) if str(item).strip())

        sections = [
            "Use the following ranking strategy as a decision guideline. Apply it to decide which passages are relevant, which passages should be omitted as irrelevant, and how the relevant passages should be ordered.",
            "\nCore relevance objective:\n" + str(strategy_card.get("objective", "")).strip(),
            "\nWhen to apply:\n" + str(strategy_card.get("when_to_apply", "")).strip(),
            "\nQuery signals:\n" + lines_for("query_signals"),
            "\nRetention rule:\n" + str(strategy_card.get("retention_rule", "")).strip(),
            "\nEvidence to promote:\n" + lines_for("evidence_to_promote"),
            "\nEvidence to demote or omit:\n" + lines_for("evidence_to_demote"),
            "\nRanking procedure:\n" + procedure_text,
            "\nTie-breaking rule:\n" + str(strategy_card.get("tie_breaking_rule", "")).strip(),
            "\nFallback behavior:\n" + str(strategy_card.get("fallback_behavior", "")).strip(),
            "\nAvoid when:\n" + lines_for("avoid_when"),
        ]
        return "\n".join(section for section in sections if section.strip())

    @staticmethod
    def _replace_strategy_card_in_messages(messages, new_strategy_card):
        messages = RayPPOTrainer._as_messages(messages)
        if not messages:
            raise ValueError("Cannot repair strategy prompt without chat messages.")

        rendered_card = json.dumps(RayPPOTrainer._jsonable(new_strategy_card), ensure_ascii=False, indent=2)
        marker = "Structured ranking criterion:"
        replaced = deepcopy(messages)
        for idx in range(len(replaced) - 1, -1, -1):
            if replaced[idx].get("role") != "user":
                continue
            content = replaced[idx].get("content", "")
            marker_pos = content.find(marker)
            if marker_pos < 0:
                continue
            span = RayPPOTrainer._find_json_object_span(content, marker_pos + len(marker))
            if span is None:
                raise ValueError("Cannot find strategy JSON block in strategy prompt.")
            replaced[idx]["content"] = content[: span[0]] + rendered_card + content[span[1] :]
            return replaced

        rendered_guidance = RayPPOTrainer._render_strategy_guidance(new_strategy_card)
        for idx in range(len(replaced) - 1, -1, -1):
            if replaced[idx].get("role") != "user":
                continue
            content = replaced[idx].get("content", "")
            pattern = re.compile(r"<strategy_guidance>.*?</strategy_guidance>", flags=re.DOTALL | re.IGNORECASE)
            if pattern.search(content):
                replaced[idx]["content"] = pattern.sub(
                    lambda _: f"<strategy_guidance>\n{rendered_guidance}\n</strategy_guidance>",
                    content,
                    count=1,
                )
                return replaced
        raise ValueError("Cannot find a supported strategy prompt block to repair.")

    def _tokenize_text_messages_for_generation(self, messages):
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            truncation = self.config.data.truncation
            max_length = self.config.data.max_prompt_length
            if truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-max_length:]
            elif truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:max_length]
            elif truncation == "middle":
                left_half = max_length // 2
                right_half = max_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_length}.")
        return input_ids[0], attention_mask[0], position_ids[0], raw_prompt_ids

    def _build_gen_batch_from_messages(self, messages_list):
        input_ids_lst = []
        attention_mask_lst = []
        position_ids_lst = []
        raw_prompt_ids_lst = []
        for messages in messages_list:
            input_ids, attention_mask, position_ids, raw_prompt_ids = self._tokenize_text_messages_for_generation(messages)
            input_ids_lst.append(input_ids)
            attention_mask_lst.append(attention_mask)
            position_ids_lst.append(position_ids)
            raw_prompt_ids_lst.append(raw_prompt_ids)

        return DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_lst, dim=0),
                "attention_mask": torch.stack(attention_mask_lst, dim=0),
                "position_ids": torch.stack(position_ids_lst, dim=0),
            },
            non_tensors={"raw_prompt_ids": np.array(raw_prompt_ids_lst, dtype=object)},
        )

    def _decode_response_text(self, response_ids):
        response_ids = response_ids.detach().cpu().tolist()
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            response_ids = [token_id for token_id in response_ids if token_id != pad_token_id]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    def _candidate_strategies_for_prompt(self, meta_batch: DataProto, local_idx: int):
        if "extra_info" not in meta_batch.non_tensor_batch:
            return []
        extra_info = meta_batch.non_tensor_batch["extra_info"][local_idx]
        if not isinstance(extra_info, dict):
            return []
        return self._jsonable(extra_info.get("strategy_candidates", []))

    @staticmethod
    def _strategy_card_equal(left, right):
        return json.dumps(RayPPOTrainer._jsonable(left), sort_keys=True, ensure_ascii=False) == json.dumps(
            RayPPOTrainer._jsonable(right), sort_keys=True, ensure_ascii=False
        )

    def _initial_repair_strategy_state(self, meta_batch: DataProto, local_idx: int):
        original = self._candidate_strategies_for_prompt(meta_batch, local_idx)
        library = []
        for idx, candidate in enumerate(original):
            if not isinstance(candidate, dict):
                continue
            card = candidate.get("strategy_prompt_structured")
            if not isinstance(card, dict):
                continue
            library.append(
                {
                    "candidate_key": f"c{idx + 1}",
                    "status": "current" if idx == 0 else "candidate",
                    "strategy_name": candidate.get("strategy_name", f"candidate_{idx + 1}"),
                    "strategy": self._jsonable(card),
                }
            )
        current = library[0]["strategy"] if library else {}
        return current, library

    def _advance_repair_strategy_library(self, library, decision):
        library = self._jsonable(library or [])
        new_strategy = self._jsonable(decision.get("new_strategy", {}))
        action = decision.get("action")
        selected_key = decision.get("selected_candidate_key", decision.get("selected_strategy_id"))
        current = next((item for item in library if item.get("status") == "current"), None)
        for item in library:
            if item.get("status") == "current":
                item["status"] = "chosen"

        selected = next((item for item in library if item.get("candidate_key") == selected_key), None)
        if action == "select_candidate_strategy":
            if selected is None:
                raise ValueError("select_candidate_strategy requires a selected library entry.")
            selected["status"] = "current"
            return self._jsonable(selected.get("strategy")), library

        if action in {"rewrite_library_strategy", "create_new_strategy"}:
            next_idx = 1 + sum(1 for item in library if str(item.get("candidate_key", "")).startswith("r"))
            library.append(
                {
                    "candidate_key": f"r{next_idx}",
                    "status": "current",
                    "strategy_name": new_strategy.get("strategy_name", f"repair_{next_idx}") if isinstance(new_strategy, dict) else f"repair_{next_idx}",
                    "strategy": self._jsonable(new_strategy),
                }
            )
            return self._jsonable(new_strategy), library

        raise ValueError(f"Unsupported repair action {action!r}.")

    @staticmethod
    def _match_strategy_identity(current_strategy_card, candidate_strategies):
        if not isinstance(current_strategy_card, dict):
            return {}
        current_objective = current_strategy_card.get("objective")
        current_procedure = RayPPOTrainer._jsonable(current_strategy_card.get("ranking_procedure", []))
        for candidate in candidate_strategies or []:
            if not isinstance(candidate, dict):
                continue
            candidate_card = candidate.get("strategy_prompt_structured") or {}
            if not isinstance(candidate_card, dict):
                continue
            objective_matches = current_objective and candidate_card.get("objective") == current_objective
            procedure_matches = current_procedure and RayPPOTrainer._jsonable(candidate_card.get("ranking_procedure", [])) == current_procedure
            if objective_matches or procedure_matches:
                return {
                    "strategy_id": candidate.get("strategy_id"),
                    "strategy_name": candidate.get("strategy_name"),
                }
        return {}

    def _instance_id_for_prompt(self, meta_batch: DataProto, local_idx: int):
        if "extra_info" in meta_batch.non_tensor_batch:
            extra_info = meta_batch.non_tensor_batch["extra_info"][local_idx]
            if isinstance(extra_info, dict):
                return str(extra_info.get("global_instance_id", extra_info.get("index", local_idx)))
        if "index" in meta_batch.non_tensor_batch:
            return str(meta_batch.non_tensor_batch["index"][local_idx])
        return str(local_idx)

    def _dataset_for_prompt(self, meta_batch: DataProto, local_idx: int):
        if "dataset" in meta_batch.non_tensor_batch:
            return str(meta_batch.non_tensor_batch["dataset"][local_idx])
        instance_id = self._instance_id_for_prompt(meta_batch, local_idx)
        return instance_id.split("_")[0]

    def _raw_prompt_user_instruction(self, meta_batch: DataProto, local_idx: int, strategy_messages):
        if "raw_prompt" in meta_batch.non_tensor_batch:
            raw_messages = self._as_messages(meta_batch.non_tensor_batch["raw_prompt"][local_idx])
            if raw_messages:
                return self._last_user_content(raw_messages)
        return self._last_user_content(strategy_messages)

    def _build_repair_request(
        self,
        meta_batch: DataProto,
        local_idx: int,
        repair_round: int,
        max_rounds: int,
        strategy_messages,
        strategy_group_batch: DataProto,
        strategy_best_idx: int,
        current_strategy_card=None,
        candidate_strategy_library=None,
    ):
        repair_cfg = self._guided_repair_cfg()
        problem_max_chars = repair_cfg.get("max_problem_chars", None)
        rollout_max_chars = repair_cfg.get("max_rollout_chars", None)

        candidate_strategies = (
            self._jsonable(candidate_strategy_library)
            if candidate_strategy_library is not None
            else self._candidate_strategies_for_prompt(meta_batch, local_idx)
        )
        if current_strategy_card is None:
            current_strategy_card = self._extract_strategy_card_from_messages(strategy_messages)
            if not current_strategy_card:
                strategy_guidance = self._extract_strategy_guidance_from_messages(strategy_messages)
                if strategy_guidance:
                    current_strategy_card = {"strategy_guidance": strategy_guidance}
        best_response = strategy_group_batch.batch["responses"][int(strategy_best_idx)]
        best_response_text = self._decode_response_text(best_response)
        data_source = None
        if "data_source" in meta_batch.non_tensor_batch:
            data_source = str(meta_batch.non_tensor_batch["data_source"][local_idx])

        return {
            "data_source": data_source,
            "ranking_problem": {
                "base_user_instruction": self._truncate_text(
                    self._raw_prompt_user_instruction(meta_batch, local_idx, strategy_messages),
                    problem_max_chars,
                )
            },
            "current_strategy": self._jsonable(current_strategy_card),
            "candidate_strategy_library": candidate_strategies,
            "student_best_rollout_under_current_strategy": {
                "response": self._truncate_text(best_response_text, rollout_max_chars),
            },
        }

    def _call_repair_llm_batch(self, requests):
        from verl.trainer.ppo.guided_repair_llm import RepairLLMResult, run_repair_batch

        try:
            return run_repair_batch(requests, self._guided_repair_cfg().get("llm", {}) or {})
        except Exception as exc:
            return [
                RepairLLMResult(
                    success=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    latency_s=0.0,
                    retry_count=0,
                )
                for _ in requests
            ]

    def _repair_dump_enabled(self):
        return bool(self._guided_repair_cfg().get("dump_records", False))

    def _repair_dump_dir(self):
        default_dir = os.path.join(str(self.config.trainer.default_local_dir), "guided_repair_dumps")
        return str(self._guided_repair_cfg().get("dump_dir", default_dir) or default_dir)

    def _repair_llm_payload_for_dump(self, request):
        try:
            from verl.trainer.ppo.guided_repair_llm import build_repair_llm_payload

            return self._jsonable(build_repair_llm_payload(request, self._guided_repair_cfg().get("llm", {}) or {}))
        except Exception as exc:
            return {
                "payload_build_error": type(exc).__name__,
                "payload_build_error_message": str(exc),
            }

    def _make_repair_dump_record(self, meta_batch: DataProto, local_idx: int, repair_round: int, request: dict, result):
        llm_result = {
            "success": bool(getattr(result, "success", False)),
            "decision": self._jsonable(getattr(result, "decision", None)),
            "raw_response": getattr(result, "raw_response", None),
            "error_type": getattr(result, "error_type", None),
            "error_message": getattr(result, "error_message", None),
            "latency_s": float(getattr(result, "latency_s", 0.0) or 0.0),
            "retry_count": int(getattr(result, "retry_count", 0) or 0),
            "attempts": self._jsonable(getattr(result, "attempts", None) or []),
        }
        return {
            "schema_version": "guided_grpo_repair_dump_v1",
            "metadata": {
                "timestamp_unix": time.time(),
                "global_step": int(self.global_steps),
                "repair_round": int(repair_round),
                "batch_failed_local_idx": int(local_idx),
                "instance_id": self._instance_id_for_prompt(meta_batch, local_idx),
                "dataset": self._dataset_for_prompt(meta_batch, local_idx),
            },
            "llm_input": {
                "request": self._jsonable(request),
                "payload": self._repair_llm_payload_for_dump(request),
            },
            "llm_output": llm_result,
            "application": {},
        }

    def _write_repair_dump_records(self, records):
        if not records or not self._repair_dump_enabled():
            return None
        dump_dir = self._repair_dump_dir()
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"repair_step_{int(self.global_steps):06d}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(self._jsonable(record), ensure_ascii=False) + "\n")
        return path

    def _data_index_for_prompt(self, meta_batch: DataProto, local_idx: int):
        if "index" in meta_batch.non_tensor_batch:
            return self._jsonable(meta_batch.non_tensor_batch["index"][local_idx])
        if "extra_info" in meta_batch.non_tensor_batch:
            extra_info = meta_batch.non_tensor_batch["extra_info"][local_idx]
            if isinstance(extra_info, dict) and "index" in extra_info:
                return self._jsonable(extra_info["index"])
        return int(local_idx)

    def _messages_for_dump(self, meta_batch: DataProto, local_idx: int, key: str, fallback_group: Optional[DataProto] = None):
        if key in meta_batch.non_tensor_batch:
            messages = self._as_messages(meta_batch.non_tensor_batch[key][local_idx])
            if messages is not None:
                return self._jsonable(messages)
        if fallback_group is not None and fallback_group.batch is not None and "prompts" in fallback_group.batch.keys() and len(fallback_group) > 0:
            prompt_text = self.tokenizer.decode(fallback_group.batch["prompts"][0].detach().cpu().tolist(), skip_special_tokens=True)
            return [{"role": "raw_prompt_text", "content": prompt_text}]
        return None

    def _init_guided_question_dump_records(self, meta_batch: DataProto, route_version: str, rollout_n: int, success_threshold: float):
        if not self._guided_question_dump_should_write():
            return None
        records = []
        for local_idx in range(len(meta_batch)):
            records.append(
                {
                    "schema_version": "guided_grpo_question_dump_v1",
                    "metadata": {
                        "timestamp_unix": time.time(),
                        "global_step": int(self.global_steps),
                        "route_version": route_version,
                        "rollout_n": int(rollout_n),
                        "success_threshold": float(success_threshold),
                        "batch_local_idx": int(local_idx),
                        "instance_id": self._instance_id_for_prompt(meta_batch, local_idx),
                        "dataset": self._dataset_for_prompt(meta_batch, local_idx),
                        "data_index": self._data_index_for_prompt(meta_batch, local_idx),
                    },
                    "outcome": {
                        "category": None,
                        "repair_success_round": None,
                        "teacher_fallback_reason": None,
                    },
                    "stages": [],
                }
            )
        return records

    def _set_question_dump_outcome(self, records, prompt_idx: int, category: str, repair_success_round=None, teacher_fallback_reason=None):
        if not records:
            return
        outcome = records[int(prompt_idx)]["outcome"]
        outcome["category"] = category
        outcome["repair_success_round"] = None if repair_success_round is None else int(repair_success_round)
        outcome["teacher_fallback_reason"] = teacher_fallback_reason

    def _append_question_dump_stage(
        self,
        records,
        prompt_idx: int,
        stage_name: str,
        group_batch: DataProto,
        scores,
        input_messages=None,
        repair_round=None,
        llm_decision=None,
    ):
        if not records:
            return

        if torch.is_tensor(scores):
            score_values = scores.detach().cpu().view(-1).tolist()
        else:
            score_values = np.array(scores).reshape(-1).tolist()
        score_values = [float(score) for score in score_values]

        best_rollout_idx = None
        best_score = None
        if score_values:
            best_rollout_idx = int(np.argmax(score_values))
            best_score = float(score_values[best_rollout_idx])

        threshold = float(records[int(prompt_idx)]["metadata"]["success_threshold"])
        rollouts = []
        response_count = len(group_batch)
        for rollout_idx in range(response_count):
            score = score_values[rollout_idx] if rollout_idx < len(score_values) else None
            rollouts.append(
                {
                    "rollout_idx": int(rollout_idx),
                    "text": self._decode_response_text(group_batch.batch["responses"][rollout_idx]),
                    "score": score,
                    "is_best": best_rollout_idx is not None and int(rollout_idx) == int(best_rollout_idx),
                }
            )

        stage = {
            "stage": stage_name,
            "round": None if repair_round is None else int(repair_round),
            "input_messages": self._jsonable(input_messages),
            "success": bool(best_score is not None and best_score > threshold),
            "best_rollout_idx": best_rollout_idx,
            "best_score": best_score,
            "rollouts": rollouts,
        }
        if llm_decision is not None:
            stage["llm_decision"] = self._jsonable(llm_decision)
        records[int(prompt_idx)]["stages"].append(stage)

    def _write_guided_question_dump_records(self, records):
        if not records:
            return 0
        dump_dir = self._guided_question_dump_dir()
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"step_{int(self.global_steps):06d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(self._jsonable(record), ensure_ascii=False) + "\n")
        print(f"Dumped guided question records to {path}")
        return len(records)

    def _dump_normal_grpo_question_records(self, batch: DataProto, rollout_n: int, scores):
        if self._guided_grpo_enabled() or not self._guided_question_dump_should_write():
            return 0
        if len(batch) % rollout_n != 0:
            return 0

        if torch.is_tensor(scores):
            scores = scores.detach().cpu().view(-1)
        else:
            scores = torch.tensor(np.array(scores).reshape(-1), dtype=torch.float32)

        records = []
        num_groups = len(batch) // rollout_n
        threshold = float(self._guided_grpo_cfg().get("success_threshold", 0.0))
        for group_idx in range(num_groups):
            start = group_idx * rollout_n
            group = self._single_group(batch, group_idx, rollout_n)
            records.append(
                {
                    "schema_version": "guided_grpo_question_dump_v1",
                    "metadata": {
                        "timestamp_unix": time.time(),
                        "global_step": int(self.global_steps),
                        "route_version": "normal_grpo",
                        "rollout_n": int(rollout_n),
                        "success_threshold": threshold,
                        "batch_local_idx": int(group_idx),
                        "instance_id": self._instance_id_for_prompt(batch, start),
                        "dataset": self._dataset_for_prompt(batch, start),
                        "data_index": self._data_index_for_prompt(batch, start),
                    },
                    "outcome": {
                        "category": "normal_grpo",
                        "repair_success_round": None,
                        "teacher_fallback_reason": None,
                    },
                    "stages": [],
                }
            )
            input_messages = self._messages_for_dump(batch, start, "raw_prompt", fallback_group=group)
            self._append_question_dump_stage(
                records,
                group_idx,
                "base",
                group,
                scores[start : start + rollout_n],
                input_messages=input_messages,
            )
        return self._write_guided_question_dump_records(records)

    def _reward_lambda_kwargs(self):
        return {
            "lambda_init": self.config.algorithm.get("lambda_init", 0.8),
            "lambda_final": self.config.algorithm.get("lambda_final", 0.0),
        }

    def _compute_routing_scores(self, batch: DataProto):
        lambda_kwargs = self._reward_lambda_kwargs()
        reward_tensor, reward_extra_infos = compute_reward(
            batch,
            self.reward_fn,
            global_step=self.global_steps,
            total_steps=self.total_training_steps,
            **lambda_kwargs,
        )
        routing_metric = str(self._guided_grpo_cfg().get("routing_metric", "score") or "score")
        if routing_metric != "score":
            metric_key = routing_metric
            if metric_key not in reward_extra_infos and routing_metric == "ndcg":
                ndcg_keys = [key for key in reward_extra_infos if str(key).startswith("ndcg@")]
                if ndcg_keys:
                    metric_key = sorted(ndcg_keys)[0]
            if metric_key not in reward_extra_infos:
                raise KeyError(f"Guided GRPO routing_metric={routing_metric!r} not found in reward extra info keys: {list(reward_extra_infos.keys())}")
            return torch.tensor(reward_extra_infos[metric_key], dtype=reward_tensor.dtype, device=reward_tensor.device)
        return reward_tensor.sum(-1)

    def _bucket_guided_advantage_rewards(self, batch: DataProto):
        cfg = self._guided_advantage_bucket_cfg()
        bucket_size = float(cfg.get("size", 0.05) or 0.05)
        if bucket_size <= 0:
            raise ValueError("algorithm.guided_grpo.advantage_bucket.size must be positive.")

        mode = str(cfg.get("mode", "ceil") or "ceil").lower()
        keep_negative_raw = bool(cfg.get("keep_negative_raw", True))
        cap_max = float(cfg.get("cap_max", 1.0))
        eps = float(cfg.get("epsilon", 1e-12))

        raw_rewards = batch.batch["token_level_rewards"]
        raw_scores = raw_rewards.sum(dim=-1)
        bucket_scores = raw_scores.clone()
        bucket_mask = raw_scores >= 0 if keep_negative_raw else torch.ones_like(raw_scores, dtype=torch.bool)
        values = raw_scores[bucket_mask]
        if values.numel() > 0:
            if mode == "ceil":
                bucketed_values = torch.ceil((values - eps) / bucket_size) * bucket_size
            elif mode == "floor":
                bucketed_values = torch.floor((values + eps) / bucket_size) * bucket_size
            elif mode == "round":
                bucketed_values = torch.round(values / bucket_size) * bucket_size
            else:
                raise ValueError(f"Unsupported guided_grpo advantage bucket mode: {mode}")
            bucket_scores[bucket_mask] = torch.clamp(bucketed_values, min=0.0, max=cap_max)

        scale = torch.ones_like(raw_scores)
        nonzero = raw_scores.abs() > eps
        scale[nonzero] = bucket_scores[nonzero] / raw_scores[nonzero]
        bucket_rewards = raw_rewards * scale.unsqueeze(-1)

        changed = (bucket_scores - raw_scores).abs() > max(eps, 1e-8)
        metrics = {
            "guided_grpo/adv_bucket_enabled": 1,
            "guided_grpo/adv_bucket_size": bucket_size,
            "guided_grpo/adv_bucket_changed_frac": float(changed.float().mean().detach().cpu().item()) if changed.numel() else 0.0,
            "guided_grpo/adv_bucket_raw_score_mean": float(raw_scores.mean().detach().cpu().item()) if raw_scores.numel() else 0.0,
            "guided_grpo/adv_bucket_score_mean": float(bucket_scores.mean().detach().cpu().item()) if bucket_scores.numel() else 0.0,
            "guided_grpo/adv_bucket_score_min": float(bucket_scores.min().detach().cpu().item()) if bucket_scores.numel() else 0.0,
            "guided_grpo/adv_bucket_score_max": float(bucket_scores.max().detach().cpu().item()) if bucket_scores.numel() else 0.0,
        }

        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if rollout_n > 0 and bucket_scores.numel() % rollout_n == 0:
            grouped = bucket_scores.view(-1, rollout_n)
            same_group = torch.isclose(grouped.max(dim=1).values, grouped.min(dim=1).values, atol=1e-8, rtol=0.0)
            all_top_group = torch.isclose(grouped.min(dim=1).values, torch.tensor(cap_max, device=grouped.device, dtype=grouped.dtype), atol=1e-8, rtol=0.0)
            metrics.update(
                {
                    "guided_grpo/adv_bucket_group_count": int(grouped.shape[0]),
                    "guided_grpo/adv_bucket_all_same_group_count": int(same_group.sum().detach().cpu().item()),
                    "guided_grpo/adv_bucket_all_same_group_frac": float(same_group.float().mean().detach().cpu().item()) if same_group.numel() else 0.0,
                    "guided_grpo/adv_bucket_all_top_group_count": int(all_top_group.sum().detach().cpu().item()),
                    "guided_grpo/adv_bucket_all_top_group_frac": float(all_top_group.float().mean().detach().cpu().item()) if all_top_group.numel() else 0.0,
                    "guided_grpo/adv_bucket_zero_adv_group_count": int(same_group.sum().detach().cpu().item()),
                }
            )

        return bucket_rewards, metrics

    @staticmethod
    def _flat_group_indices(prompt_indices, repeat_times):
        flat_indices = []
        for prompt_idx in prompt_indices:
            start = int(prompt_idx) * repeat_times
            flat_indices.extend(range(start, start + repeat_times))
        return flat_indices

    @staticmethod
    def _training_batch_from_rollout(meta_batch: DataProto, rollout_output: DataProto, repeat_times: int):
        training_batch = meta_batch.repeat(repeat_times=repeat_times, interleave=True).union(rollout_output)
        RayPPOTrainer._drop_rollout_transient_non_tensor_keys(training_batch)
        return training_batch

    @staticmethod
    def _drop_rollout_transient_non_tensor_keys(batch: DataProto):
        if batch.non_tensor_batch is None:
            return
        for key in ("raw_prompt_ids", "tools_kwargs"):
            batch.non_tensor_batch.pop(key, None)

    @staticmethod
    def _assign_group_uids(batch: DataProto, repeat_times: int):
        if len(batch) % repeat_times != 0:
            raise ValueError(f"Guided GRPO batch size {len(batch)} is not divisible by rollout.n={repeat_times}.")
        uids = []
        for _ in range(len(batch) // repeat_times):
            group_uid = str(uuid.uuid4())
            uids.extend([group_uid] * repeat_times)
        batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)

    @staticmethod
    def _ensure_off_policy_mask(batch: DataProto):
        if batch.batch is not None and "off_policy_mask" not in batch.batch.keys():
            batch.batch["off_policy_mask"] = torch.zeros_like(batch.batch["responses"], dtype=torch.bool)

    @staticmethod
    def _set_actor_loss_weight(batch: DataProto, weight: float):
        batch.batch["actor_loss_weight"] = torch.full(
            (len(batch),),
            float(weight),
            dtype=torch.float32,
            device=batch.batch["responses"].device,
        )

    @staticmethod
    def _sample_non_best_index(repeat_times: int, best_idx: int):
        candidates = [idx for idx in range(repeat_times) if idx != int(best_idx)]
        if not candidates:
            return int(best_idx)
        return int(np.random.choice(candidates))

    def _build_strategy_gen_batch(self, meta_batch: DataProto, prompt_indices):
        required_batch_keys = {
            "strategy_input_ids": "input_ids",
            "strategy_attention_mask": "attention_mask",
            "strategy_position_ids": "position_ids",
        }
        missing = [key for key in required_batch_keys if key not in meta_batch.batch.keys()]
        if missing:
            raise KeyError(f"Guided GRPO requires tokenized strategy prompt fields, missing: {missing}")
        if "strategy_raw_prompt_ids" not in meta_batch.non_tensor_batch:
            raise KeyError("Guided GRPO requires `strategy_raw_prompt_ids` in non_tensor_batch.")

        selected = meta_batch[prompt_indices]
        tensors = {target_key: selected.batch[source_key] for source_key, target_key in required_batch_keys.items()}
        non_tensors = {"raw_prompt_ids": selected.non_tensor_batch["strategy_raw_prompt_ids"]}
        if "tools_kwargs" in selected.non_tensor_batch:
            non_tensors["tools_kwargs"] = selected.non_tensor_batch["tools_kwargs"]
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    def _generate_sequences_with_prompt_padding(self, gen_batch: DataProto, rollout_n: int):
        prompt_count = len(gen_batch)
        if prompt_count == 0:
            return None

        gen_batch_padded, _ = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        if not self.async_rollout_mode:
            output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        else:
            self.async_rollout_manager.wake_up()
            output_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)
            self.async_rollout_manager.sleep()

        keep_size = prompt_count * rollout_n
        return output_padded[:keep_size]

    def _encode_teacher_response(self, response_text):
        if response_text is None:
            raise ValueError("Guided GRPO teacher fallback requires a non-empty teacher response.")
        if not isinstance(response_text, str):
            response_text = str(response_text)

        max_response_length = self.config.data.max_response_length
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        if max_response_length <= 0:
            raise ValueError("data.max_response_length must be positive.")

        if eos_token_id is not None:
            if len(response_ids) < max_response_length:
                if not response_ids or response_ids[-1] != eos_token_id:
                    response_ids.append(eos_token_id)
            elif response_ids and response_ids[-1] != eos_token_id:
                response_ids[max_response_length - 1] = eos_token_id

        response_ids = response_ids[:max_response_length]
        response_ids = response_ids + [pad_token_id] * (max_response_length - len(response_ids))
        return torch.tensor(response_ids, dtype=torch.long)

    def _replace_response(self, batch: DataProto, row_idx: int, response_ids: torch.Tensor, mark_off_policy: bool = False):
        if batch.batch["position_ids"].dim() != 2:
            raise NotImplementedError("Guided GRPO response replacement currently supports text-only 2D position_ids.")

        device = batch.batch["responses"].device
        response_ids = response_ids.to(device=device, dtype=batch.batch["responses"].dtype).clone()
        prompt_length = batch.batch["prompts"].shape[-1]
        prompt_ids = batch.batch["prompts"][row_idx].clone()
        prompt_mask = batch.batch["attention_mask"][row_idx, :prompt_length].clone()
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
        response_mask = get_response_mask(response_ids.unsqueeze(0), eos_token=eos_token_id, dtype=prompt_mask.dtype)[0]

        batch.batch["responses"][row_idx] = response_ids
        batch.batch["input_ids"][row_idx] = torch.cat([prompt_ids, response_ids], dim=-1)
        batch.batch["attention_mask"][row_idx] = torch.cat([prompt_mask, response_mask], dim=-1)
        batch.batch["position_ids"][row_idx] = compute_position_id_with_mask(batch.batch["attention_mask"][row_idx].unsqueeze(0))[0]
        if mark_off_policy:
            self._ensure_off_policy_mask(batch)
            batch.batch["off_policy_mask"][row_idx] = response_mask.bool()

    def _teacher_response_for_prompt(self, meta_batch: DataProto, local_idx: int):
        teacher_key = self.config.data.get("teacher_response_key", "teacher_response")
        if "teacher_response" in meta_batch.non_tensor_batch:
            return meta_batch.non_tensor_batch["teacher_response"][local_idx]
        if teacher_key in meta_batch.non_tensor_batch:
            return meta_batch.non_tensor_batch[teacher_key][local_idx]
        raise KeyError("Guided GRPO teacher fallback requires `teacher_response` in the dataset.")

    @staticmethod
    def _drop_rollout_log_probs(batch: DataProto):
        if batch.batch is not None and "rollout_log_probs" in batch.batch.keys():
            batch.batch.pop("rollout_log_probs")

    @staticmethod
    def _single_group(batch: DataProto, group_idx: int, repeat_times: int):
        start = int(group_idx) * repeat_times
        return batch[list(range(start, start + repeat_times))]

    def _append_teacher_fallback_groups(
        self,
        parts,
        base_failed_batch: DataProto,
        failed_meta_batch: DataProto,
        local_idx: int,
        strategy_group_batch: DataProto,
        base_best_idx: int,
        strategy_best_idx: int,
        rollout_n: int,
        train_strategy_group: bool = True,
        teacher_fallback_aux_group: str = "current_strategy",
        initial_strategy_group_batch: DataProto | None = None,
        initial_strategy_best_idx: int | None = None,
        best_strategy_group_batch: DataProto | None = None,
        best_strategy_best_idx: int | None = None,
    ):
        teacher_response = self._encode_teacher_response(self._teacher_response_for_prompt(failed_meta_batch, local_idx))

        base_replace_idx = int(local_idx) * rollout_n + self._sample_non_best_index(rollout_n, int(base_best_idx))
        self._replace_response(base_failed_batch, base_replace_idx, teacher_response, mark_off_policy=True)

        parts.append(base_failed_batch[self._flat_group_indices([local_idx], rollout_n)])
        if train_strategy_group and teacher_fallback_aux_group != "none":
            if teacher_fallback_aux_group == "initial_strategy":
                strategy_group = initial_strategy_group_batch
                strategy_best_idx = initial_strategy_best_idx
            elif teacher_fallback_aux_group == "best_strategy":
                strategy_group = best_strategy_group_batch
                strategy_best_idx = best_strategy_best_idx
            elif teacher_fallback_aux_group == "current_strategy":
                strategy_group = strategy_group_batch
            else:
                raise ValueError(f"Unsupported teacher_fallback_aux_group={teacher_fallback_aux_group!r}.")
            if strategy_group is None or strategy_best_idx is None:
                raise ValueError("Teacher fallback auxiliary group requires a strategy group and best index.")
            strategy_replace_idx = self._sample_non_best_index(rollout_n, int(strategy_best_idx))
            self._replace_response(strategy_group, strategy_replace_idx, teacher_response, mark_off_policy=True)
            parts.append(strategy_group)

    def _build_guided_grpo_direct_teacher_fallback_batch(
        self,
        parts,
        base_failed_batch: DataProto,
        failed_meta_batch: DataProto,
        base_best_indices,
        rollout_n: int,
        route_metrics,
        question_dump_records=None,
        failed_prompt_indices=None,
    ):
        failed_prompt_indices = failed_prompt_indices or list(range(len(failed_meta_batch)))

        for local_idx in range(len(failed_prompt_indices)):
            base_replace_idx = local_idx * rollout_n + self._sample_non_best_index(rollout_n, base_best_indices[local_idx])
            teacher_response = self._encode_teacher_response(self._teacher_response_for_prompt(failed_meta_batch, local_idx))
            self._replace_response(base_failed_batch, base_replace_idx, teacher_response, mark_off_policy=True)
            self._set_question_dump_outcome(
                question_dump_records,
                failed_prompt_indices[local_idx],
                "direct_teacher_fallback",
                teacher_fallback_reason="base_failed",
            )

        if question_dump_records:
            fallback_scores = self._compute_routing_scores(base_failed_batch).view(len(failed_prompt_indices), rollout_n)
            for local_idx, prompt_idx in enumerate(failed_prompt_indices):
                group = self._single_group(base_failed_batch, local_idx, rollout_n)
                input_messages = self._messages_for_dump(failed_meta_batch, local_idx, "raw_prompt", fallback_group=group)
                self._append_question_dump_stage(
                    question_dump_records,
                    prompt_idx,
                    "base_after_direct_teacher_fallback",
                    group,
                    fallback_scores[local_idx],
                    input_messages=input_messages,
                )

        parts.append(base_failed_batch)
        final_batch = DataProto.concat(parts)
        self._drop_rollout_log_probs(final_batch)
        self._assign_group_uids(final_batch, rollout_n)
        route_metrics.update(
            {
                "guided_grpo/strategy_success_count": 0,
                "guided_grpo/teacher_fallback_count": len(failed_prompt_indices),
                "guided_grpo/direct_teacher_fallback_count": len(failed_prompt_indices),
                "guided_grpo/final_group_count": len(final_batch) // rollout_n,
                "guided_grpo/train_strategy_groups": int(self._guided_train_strategy_groups_enabled()),
                "guided_grpo/auxiliary_group_loss_weight": self._guided_auxiliary_group_loss_weight(),
                "guided_grpo/teacher_fallback_aux_group": self._guided_teacher_fallback_aux_group(),
            }
        )
        route_metrics["guided_grpo/question_dump_record_count"] = self._write_guided_question_dump_records(question_dump_records)
        return final_batch, route_metrics

    def _build_guided_grpo_training_batch_with_repair(
        self,
        parts,
        base_failed_batch: DataProto,
        failed_meta_batch: DataProto,
        strategy_training_batch: DataProto,
        strategy_best,
        strategy_success,
        base_best_indices,
        rollout_n: int,
        route_metrics,
        question_dump_records=None,
        failed_prompt_indices=None,
    ):
        repair_cfg = self._guided_repair_cfg()
        max_rounds = self._guided_repair_max_rounds()
        failed_prompt_indices = failed_prompt_indices or list(range(len(failed_meta_batch)))
        train_strategy_groups = self._guided_train_strategy_groups_enabled()
        bridge_success_to_base = self._guided_bridge_success_to_base_enabled()
        if not bridge_success_to_base and not train_strategy_groups:
            raise ValueError(
                "algorithm.guided_grpo.bridge_success_to_base=False requires "
                "algorithm.guided_grpo.train_auxiliary_groups=True; otherwise successful guided groups are not trained."
            )
        teacher_fallback_aux_group = self._guided_teacher_fallback_aux_group()
        strategy_best_indices = strategy_best.indices.cpu().tolist()
        initial_success_locals = torch.nonzero(strategy_success, as_tuple=False).view(-1).cpu().tolist()
        active_locals = torch.nonzero(~strategy_success, as_tuple=False).view(-1).cpu().tolist()
        for local_idx in initial_success_locals:
            self._set_question_dump_outcome(question_dump_records, failed_prompt_indices[local_idx], "initial_strategy_success")

        repair_metrics = {
            "guided_grpo/repair_enabled": 1,
            "guided_grpo/repair_attempted_count": len(active_locals),
            "guided_grpo/repair_success_count": 0,
            "guided_grpo/repair_failed_count": 0,
            "guided_grpo/repair_skipped_count": 0,
            "guided_grpo/repair_llm_failure_count": 0,
            "guided_grpo/repair_timeout_count": 0,
            "guided_grpo/repair_invalid_json_count": 0,
            "guided_grpo/repair_action_rewrite_count": 0,
            "guided_grpo/repair_action_select_count": 0,
            "guided_grpo/repair_action_create_count": 0,
            "guided_grpo/repair_latency_s_mean": 0.0,
            "guided_grpo/repair_rounds_mean": 0.0,
            "guided_grpo/repair_rounds_max": 0,
            "guided_grpo/repair_dump_record_count": 0,
        }
        repair_latencies = []
        repair_success_rounds = []
        fallback_items = []
        fallback_reasons = {}
        repair_dump_records = []
        repair_dump_records_by_round = {}
        repair_decisions_by_round = {}
        current_strategy_cards = {}
        working_libraries = {}
        initial_strategy_groups = {}
        initial_strategy_best_indices = {}
        best_strategy_groups = {}
        best_strategy_best_indices = {}
        best_strategy_scores = {}
        best_strategy_rounds = {}
        for local_idx in range(len(failed_meta_batch)):
            current_strategy_cards[local_idx], working_libraries[local_idx] = self._initial_repair_strategy_state(failed_meta_batch, local_idx)
            initial_strategy_groups[local_idx] = self._single_group(strategy_training_batch, local_idx, rollout_n)
            initial_strategy_best_indices[local_idx] = strategy_best_indices[local_idx]
            best_strategy_groups[local_idx] = initial_strategy_groups[local_idx]
            best_strategy_best_indices[local_idx] = strategy_best_indices[local_idx]
            best_strategy_scores[local_idx] = float(strategy_best.values[local_idx].detach().cpu().item())
            best_strategy_rounds[local_idx] = None

        for local_idx in initial_success_locals:
            if bridge_success_to_base:
                base_replace_idx = local_idx * rollout_n + self._sample_non_best_index(rollout_n, base_best_indices[local_idx])
                donor_idx = local_idx * rollout_n + strategy_best_indices[local_idx]
                donor_response = strategy_training_batch.batch["responses"][donor_idx]
                self._replace_response(base_failed_batch, base_replace_idx, donor_response, mark_off_policy=True)

        if initial_success_locals:
            if bridge_success_to_base:
                parts.append(base_failed_batch[self._flat_group_indices(initial_success_locals, rollout_n)])
            if train_strategy_groups:
                parts.append(strategy_training_batch[self._flat_group_indices(initial_success_locals, rollout_n)])

        if active_locals:
            current_batch = strategy_training_batch[self._flat_group_indices(active_locals, rollout_n)]
            current_best_indices = [strategy_best_indices[local_idx] for local_idx in active_locals]
            strategy_raw_prompts = failed_meta_batch.non_tensor_batch.get("strategy_raw_prompt", None)
            current_messages = {}
            kept_locals = []
            kept_best_indices = []
            kept_group_positions = []
            for pos, local_idx in enumerate(active_locals):
                messages = None
                if strategy_raw_prompts is not None:
                    messages = self._as_messages(strategy_raw_prompts[local_idx])
                if messages is None:
                    fallback_items.append(
                        (
                            local_idx,
                            self._single_group(current_batch, pos, rollout_n),
                            base_best_indices[local_idx],
                            current_best_indices[pos],
                        )
                    )
                    fallback_reasons[local_idx] = "missing_strategy_prompt"
                    repair_metrics["guided_grpo/repair_llm_failure_count"] += 1
                    continue
                current_messages[local_idx] = messages
                kept_locals.append(local_idx)
                kept_best_indices.append(current_best_indices[pos])
                kept_group_positions.append(pos)

            if kept_locals:
                current_batch = DataProto.concat([self._single_group(current_batch, pos, rollout_n) for pos in kept_group_positions])
                active_locals = kept_locals
                current_best_indices = kept_best_indices
            else:
                active_locals = []
                current_best_indices = []
                current_batch = None
        else:
            current_batch = None
            current_best_indices = []
            current_messages = {}

        max_repairs_per_batch = repair_cfg.get("max_repairs_per_batch", None)
        if max_repairs_per_batch is not None:
            max_repairs_per_batch = max(0, int(max_repairs_per_batch))

        for repair_round in range(1, max_rounds + 1):
            if not active_locals:
                break

            repair_count = len(active_locals)
            if max_repairs_per_batch is not None:
                repair_count = min(repair_count, max_repairs_per_batch)

            repair_locals = active_locals[:repair_count]
            skipped_locals = active_locals[repair_count:]
            if skipped_locals:
                for skip_pos, local_idx in enumerate(skipped_locals, start=repair_count):
                    fallback_items.append(
                        (
                            local_idx,
                            self._single_group(current_batch, skip_pos, rollout_n),
                            base_best_indices[local_idx],
                            current_best_indices[skip_pos],
                        )
                    )
                    fallback_reasons[local_idx] = "max_repairs_per_batch_skipped"
                repair_metrics["guided_grpo/repair_skipped_count"] += len(skipped_locals)

            requests = []
            for pos, local_idx in enumerate(repair_locals):
                strategy_group = self._single_group(current_batch, pos, rollout_n)
                requests.append(
                    self._build_repair_request(
                        failed_meta_batch,
                        local_idx,
                        repair_round,
                        max_rounds,
                        current_messages[local_idx],
                        strategy_group,
                        current_best_indices[pos],
                        current_strategy_card=current_strategy_cards.get(local_idx),
                        candidate_strategy_library=working_libraries.get(local_idx),
                    )
                )

            llm_results = self._call_repair_llm_batch(requests)
            repaired_messages = []
            repaired_locals = []
            repaired_old_positions = []
            for pos, (local_idx, request, result) in enumerate(zip(repair_locals, requests, llm_results)):
                dump_record = None
                if self._repair_dump_enabled():
                    dump_record = self._make_repair_dump_record(failed_meta_batch, local_idx, repair_round, request, result)
                    repair_dump_records.append(dump_record)
                    repair_dump_records_by_round[(local_idx, repair_round)] = dump_record

                repair_latencies.append(float(getattr(result, "latency_s", 0.0) or 0.0))
                if not getattr(result, "success", False):
                    error_type = getattr(result, "error_type", "") or ""
                    if error_type == "timeout":
                        repair_metrics["guided_grpo/repair_timeout_count"] += 1
                    elif error_type == "invalid_json_or_schema":
                        repair_metrics["guided_grpo/repair_invalid_json_count"] += 1
                    else:
                        repair_metrics["guided_grpo/repair_llm_failure_count"] += 1
                    fallback_items.append(
                        (
                            local_idx,
                            self._single_group(current_batch, pos, rollout_n),
                            base_best_indices[local_idx],
                            current_best_indices[pos],
                        )
                    )
                    fallback_reasons[local_idx] = f"llm_failure:{error_type or 'unknown'}"
                    if dump_record is not None:
                        dump_record["application"].update(
                            {
                                "prompt_replacement_success": False,
                                "outcome_after_llm": "teacher_fallback_due_to_llm_failure",
                            }
                        )
                    continue

                decision = result.decision
                action = decision.get("action")
                if action == "rewrite_library_strategy":
                    repair_metrics["guided_grpo/repair_action_rewrite_count"] += 1
                elif action == "select_candidate_strategy":
                    repair_metrics["guided_grpo/repair_action_select_count"] += 1
                elif action == "create_new_strategy":
                    repair_metrics["guided_grpo/repair_action_create_count"] += 1

                try:
                    next_messages = self._replace_strategy_card_in_messages(current_messages[local_idx], decision["new_strategy"])
                except Exception:
                    repair_metrics["guided_grpo/repair_llm_failure_count"] += 1
                    fallback_items.append(
                        (
                            local_idx,
                            self._single_group(current_batch, pos, rollout_n),
                            base_best_indices[local_idx],
                            current_best_indices[pos],
                        )
                    )
                    fallback_reasons[local_idx] = "prompt_replacement_failure"
                    if dump_record is not None:
                        dump_record["application"].update(
                            {
                                "prompt_replacement_success": False,
                                "outcome_after_llm": "teacher_fallback_due_to_prompt_replacement_failure",
                            }
                        )
                    continue

                repaired_messages.append(next_messages)
                repaired_locals.append(local_idx)
                repaired_old_positions.append(pos)
                repair_decisions_by_round[(local_idx, repair_round)] = self._jsonable(decision)
                current_strategy_cards[local_idx], working_libraries[local_idx] = self._advance_repair_strategy_library(
                    working_libraries.get(local_idx),
                    decision,
                )
                if dump_record is not None:
                    dump_record["application"].update(
                        {
                            "prompt_replacement_success": True,
                            "rendered_strategy_prompt_messages": self._jsonable(next_messages),
                        }
                    )

            if not repaired_locals:
                active_locals = []
                current_batch = None
                current_best_indices = []
                current_messages = {}
                break

            repaired_gen_batch = self._build_gen_batch_from_messages(repaired_messages)
            repaired_rollout_output = self._generate_sequences_with_prompt_padding(repaired_gen_batch, rollout_n)
            repaired_meta_batch = failed_meta_batch[repaired_locals]
            repaired_training_batch = self._training_batch_from_rollout(repaired_meta_batch, repaired_rollout_output, rollout_n)
            self._ensure_off_policy_mask(repaired_training_batch)
            self._set_actor_loss_weight(repaired_training_batch, self._guided_auxiliary_group_loss_weight())
            repaired_scores = self._compute_routing_scores(repaired_training_batch).view(len(repaired_locals), rollout_n)
            repaired_best = torch.max(repaired_scores, dim=1)
            repaired_success = repaired_best.values > float(self._guided_grpo_cfg().get("success_threshold", 0.0))
            repaired_best_indices = repaired_best.indices.cpu().tolist()

            next_active_locals = []
            next_active_messages = {}
            next_active_groups = []
            next_active_best_indices = []
            for repaired_pos, local_idx in enumerate(repaired_locals):
                strategy_group = self._single_group(repaired_training_batch, repaired_pos, rollout_n)
                dump_record = repair_dump_records_by_round.get((local_idx, repair_round))
                repaired_best_score = float(repaired_best.values[repaired_pos].detach().cpu().item())
                repaired_best_idx = int(repaired_best_indices[repaired_pos])
                repaired_is_success = bool(repaired_success[repaired_pos].item())
                if repaired_best_score > best_strategy_scores[local_idx]:
                    best_strategy_groups[local_idx] = strategy_group
                    best_strategy_best_indices[local_idx] = repaired_best_idx
                    best_strategy_scores[local_idx] = repaired_best_score
                    best_strategy_rounds[local_idx] = repair_round
                self._append_question_dump_stage(
                    question_dump_records,
                    failed_prompt_indices[local_idx],
                    "repair_strategy",
                    strategy_group,
                    repaired_scores[repaired_pos],
                    input_messages=repaired_messages[repaired_pos],
                    repair_round=repair_round,
                    llm_decision=repair_decisions_by_round.get((local_idx, repair_round)),
                )
                if dump_record is not None:
                    dump_record["application"].update(
                        {
                            "repaired_rollout_best_score": repaired_best_score,
                            "repaired_rollout_best_index": repaired_best_idx,
                            "repaired_rollout_success": repaired_is_success,
                            "outcome_after_rollout": "repair_success" if repaired_is_success else "continue_repair",
                        }
                    )
                if bool(repaired_success[repaired_pos].item()):
                    if bridge_success_to_base:
                        base_replace_idx = local_idx * rollout_n + self._sample_non_best_index(rollout_n, base_best_indices[local_idx])
                        donor_response = strategy_group.batch["responses"][repaired_best_indices[repaired_pos]]
                        self._replace_response(base_failed_batch, base_replace_idx, donor_response, mark_off_policy=True)
                        parts.append(base_failed_batch[self._flat_group_indices([local_idx], rollout_n)])
                    if train_strategy_groups:
                        parts.append(strategy_group)
                    self._set_question_dump_outcome(
                        question_dump_records,
                        failed_prompt_indices[local_idx],
                        "repair_success",
                        repair_success_round=repair_round,
                    )
                    repair_metrics["guided_grpo/repair_success_count"] += 1
                    repair_success_rounds.append(repair_round)
                else:
                    next_active_locals.append(local_idx)
                    next_active_messages[local_idx] = repaired_messages[repaired_pos]
                    next_active_groups.append(strategy_group)
                    next_active_best_indices.append(repaired_best_indices[repaired_pos])

            active_locals = next_active_locals
            current_messages = next_active_messages
            current_best_indices = next_active_best_indices
            current_batch = DataProto.concat(next_active_groups) if next_active_groups else None

        if active_locals and current_batch is not None:
            for pos, local_idx in enumerate(active_locals):
                fallback_items.append(
                    (
                        local_idx,
                        self._single_group(current_batch, pos, rollout_n),
                        base_best_indices[local_idx],
                        current_best_indices[pos],
                    )
                )
                fallback_reasons[local_idx] = "repair_exhausted"

        teacher_fallback_count = 0
        for local_idx, strategy_group, base_best_idx, strategy_best_idx in fallback_items:
            self._set_question_dump_outcome(
                question_dump_records,
                failed_prompt_indices[local_idx],
                "teacher_fallback",
                teacher_fallback_reason=fallback_reasons.get(local_idx, "repair_failed"),
            )
            if question_dump_records and teacher_fallback_aux_group == "best_strategy":
                question_dump_records[failed_prompt_indices[local_idx]]["outcome"].update(
                    {
                        "teacher_fallback_aux_group": "best_strategy",
                        "best_strategy_source_stage": "initial_strategy" if best_strategy_rounds.get(local_idx) is None else "repair_strategy",
                        "best_strategy_source_round": best_strategy_rounds.get(local_idx),
                        "best_strategy_best_score": best_strategy_scores.get(local_idx),
                    }
                )
            self._append_teacher_fallback_groups(
                parts,
                base_failed_batch,
                failed_meta_batch,
                local_idx,
                strategy_group,
                base_best_idx,
                strategy_best_idx,
                rollout_n,
                train_strategy_group=train_strategy_groups,
                teacher_fallback_aux_group=teacher_fallback_aux_group,
                initial_strategy_group_batch=initial_strategy_groups.get(local_idx),
                initial_strategy_best_idx=initial_strategy_best_indices.get(local_idx),
                best_strategy_group_batch=best_strategy_groups.get(local_idx),
                best_strategy_best_idx=best_strategy_best_indices.get(local_idx),
            )
            teacher_fallback_count += 1

        repair_metrics["guided_grpo/repair_failed_count"] = teacher_fallback_count
        if repair_latencies:
            repair_metrics["guided_grpo/repair_latency_s_mean"] = float(np.mean(repair_latencies))
        if repair_success_rounds:
            repair_metrics["guided_grpo/repair_rounds_mean"] = float(np.mean(repair_success_rounds))
            repair_metrics["guided_grpo/repair_rounds_max"] = int(max(repair_success_rounds))
        repair_metrics["guided_grpo/repair_dump_record_count"] = len(repair_dump_records) if self._repair_dump_enabled() else 0
        self._write_repair_dump_records(repair_dump_records)

        final_batch = DataProto.concat(parts)
        self._drop_rollout_log_probs(final_batch)
        self._assign_group_uids(final_batch, rollout_n)
        route_metrics.update(
            {
                "guided_grpo/strategy_success_count": len(initial_success_locals),
                "guided_grpo/teacher_fallback_count": teacher_fallback_count,
                "guided_grpo/final_group_count": len(final_batch) // rollout_n,
                "guided_grpo/train_strategy_groups": int(train_strategy_groups),
                "guided_grpo/auxiliary_group_loss_weight": self._guided_auxiliary_group_loss_weight(),
                "guided_grpo/bridge_success_to_base": int(bridge_success_to_base),
                "guided_grpo/teacher_fallback_aux_group": teacher_fallback_aux_group,
            }
        )
        route_metrics.update(repair_metrics)
        route_metrics["guided_grpo/question_dump_record_count"] = self._write_guided_question_dump_records(question_dump_records)
        return final_batch, route_metrics

    def _build_guided_grpo_training_batch(self, meta_batch: DataProto, base_rollout_output: DataProto):
        rollout_n = self.config.actor_rollout_ref.rollout.n
        cfg = self._guided_grpo_cfg()
        train_strategy_groups = self._guided_train_strategy_groups_enabled()
        bridge_success_to_base = self._guided_bridge_success_to_base_enabled()
        if not bridge_success_to_base and not train_strategy_groups:
            raise ValueError(
                "algorithm.guided_grpo.bridge_success_to_base=False requires "
                "algorithm.guided_grpo.train_auxiliary_groups=True; otherwise successful guided groups are not trained."
            )
        success_threshold = float(cfg.get("success_threshold", 0.0))
        if self._guided_direct_teacher_fallback_enabled():
            route_version = "direct_teacher_fallback"
        else:
            route_version = "v1_repair" if self._guided_repair_enabled() else "v0"
        if not train_strategy_groups:
            route_version = f"{route_version}_baseonly"
        if not bridge_success_to_base:
            route_version = f"{route_version}_nobridge"

        base_training_batch = self._training_batch_from_rollout(meta_batch, base_rollout_output, rollout_n)
        self._ensure_off_policy_mask(base_training_batch)
        self._set_actor_loss_weight(base_training_batch, 1.0)
        base_scores = self._compute_routing_scores(base_training_batch)
        num_prompts = len(meta_batch)
        base_scores = base_scores.view(num_prompts, rollout_n)
        base_best = torch.max(base_scores, dim=1)
        base_success = base_best.values > success_threshold
        question_dump_records = self._init_guided_question_dump_records(meta_batch, route_version, rollout_n, success_threshold)
        if question_dump_records:
            for prompt_idx in range(num_prompts):
                group = self._single_group(base_training_batch, prompt_idx, rollout_n)
                input_messages = self._messages_for_dump(meta_batch, prompt_idx, "raw_prompt", fallback_group=group)
                self._append_question_dump_stage(
                    question_dump_records,
                    prompt_idx,
                    "base",
                    group,
                    base_scores[prompt_idx],
                    input_messages=input_messages,
                )

        success_prompt_indices = torch.nonzero(base_success, as_tuple=False).view(-1).cpu().tolist()
        failed_prompt_indices = torch.nonzero(~base_success, as_tuple=False).view(-1).cpu().tolist()
        for prompt_idx in success_prompt_indices:
            self._set_question_dump_outcome(question_dump_records, prompt_idx, "base_success")
        route_metrics = {
            "guided_grpo/base_success_count": len(success_prompt_indices),
            "guided_grpo/base_failed_count": len(failed_prompt_indices),
            "guided_grpo/success_threshold": success_threshold,
        }

        if not failed_prompt_indices:
            self._drop_rollout_log_probs(base_training_batch)
            self._assign_group_uids(base_training_batch, rollout_n)
            route_metrics.update(
                {
                    "guided_grpo/strategy_success_count": 0,
                    "guided_grpo/teacher_fallback_count": 0,
                    "guided_grpo/final_group_count": num_prompts,
                    "guided_grpo/train_strategy_groups": int(train_strategy_groups),
                    "guided_grpo/auxiliary_group_loss_weight": self._guided_auxiliary_group_loss_weight(),
                    "guided_grpo/bridge_success_to_base": int(bridge_success_to_base),
                    "guided_grpo/teacher_fallback_aux_group": self._guided_teacher_fallback_aux_group(),
                }
            )
            route_metrics["guided_grpo/question_dump_record_count"] = self._write_guided_question_dump_records(question_dump_records)
            return base_training_batch, route_metrics

        parts = []
        if success_prompt_indices:
            parts.append(base_training_batch[self._flat_group_indices(success_prompt_indices, rollout_n)])

        failed_flat_indices = self._flat_group_indices(failed_prompt_indices, rollout_n)
        base_failed_batch = base_training_batch[failed_flat_indices]
        failed_meta_batch = meta_batch[failed_prompt_indices]
        base_best_indices = base_best.indices[failed_prompt_indices].cpu().tolist()

        if self._guided_direct_teacher_fallback_enabled():
            return self._build_guided_grpo_direct_teacher_fallback_batch(
                parts=parts,
                base_failed_batch=base_failed_batch,
                failed_meta_batch=failed_meta_batch,
                base_best_indices=base_best_indices,
                rollout_n=rollout_n,
                route_metrics=route_metrics,
                question_dump_records=question_dump_records,
                failed_prompt_indices=failed_prompt_indices,
            )

        strategy_gen_batch = self._build_strategy_gen_batch(failed_meta_batch, list(range(len(failed_prompt_indices))))
        strategy_rollout_output = self._generate_sequences_with_prompt_padding(strategy_gen_batch, rollout_n)
        strategy_training_batch = self._training_batch_from_rollout(failed_meta_batch, strategy_rollout_output, rollout_n)
        self._ensure_off_policy_mask(strategy_training_batch)
        self._set_actor_loss_weight(strategy_training_batch, self._guided_auxiliary_group_loss_weight())
        strategy_scores = self._compute_routing_scores(strategy_training_batch).view(len(failed_prompt_indices), rollout_n)
        strategy_best = torch.max(strategy_scores, dim=1)
        strategy_success = strategy_best.values > success_threshold
        strategy_best_indices = strategy_best.indices.cpu().tolist()
        if question_dump_records:
            for local_idx, prompt_idx in enumerate(failed_prompt_indices):
                group = self._single_group(strategy_training_batch, local_idx, rollout_n)
                input_messages = self._messages_for_dump(failed_meta_batch, local_idx, "strategy_raw_prompt", fallback_group=group)
                self._append_question_dump_stage(
                    question_dump_records,
                    prompt_idx,
                    "initial_strategy",
                    group,
                    strategy_scores[local_idx],
                    input_messages=input_messages,
                )

        if self._guided_repair_enabled():
            return self._build_guided_grpo_training_batch_with_repair(
                parts=parts,
                base_failed_batch=base_failed_batch,
                failed_meta_batch=failed_meta_batch,
                strategy_training_batch=strategy_training_batch,
                strategy_best=strategy_best,
                strategy_success=strategy_success,
                base_best_indices=base_best_indices,
                rollout_n=rollout_n,
                route_metrics=route_metrics,
                question_dump_records=question_dump_records,
                failed_prompt_indices=failed_prompt_indices,
            )

        teacher_fallback_count = 0
        teacher_fallback_aux_group = self._guided_teacher_fallback_aux_group()
        for local_idx in range(len(failed_prompt_indices)):
            base_replace_idx = local_idx * rollout_n + self._sample_non_best_index(rollout_n, base_best_indices[local_idx])
            strategy_replace_idx = local_idx * rollout_n + self._sample_non_best_index(rollout_n, strategy_best_indices[local_idx])
            if bool(strategy_success[local_idx].item()):
                self._set_question_dump_outcome(question_dump_records, failed_prompt_indices[local_idx], "strategy_success")
                if bridge_success_to_base:
                    donor_idx = local_idx * rollout_n + strategy_best_indices[local_idx]
                    donor_response = strategy_training_batch.batch["responses"][donor_idx]
                    self._replace_response(base_failed_batch, base_replace_idx, donor_response, mark_off_policy=True)
            else:
                self._set_question_dump_outcome(
                    question_dump_records,
                    failed_prompt_indices[local_idx],
                    "teacher_fallback",
                    teacher_fallback_reason="initial_strategy_failed",
                )
                teacher_fallback_count += 1
                teacher_response = self._encode_teacher_response(self._teacher_response_for_prompt(failed_meta_batch, local_idx))
                self._replace_response(base_failed_batch, base_replace_idx, teacher_response, mark_off_policy=True)
                if train_strategy_groups and teacher_fallback_aux_group != "none":
                    self._replace_response(strategy_training_batch, strategy_replace_idx, teacher_response, mark_off_policy=True)

        bridge_or_fallback_locals = (
            list(range(len(failed_prompt_indices)))
            if bridge_success_to_base
            else torch.nonzero(~strategy_success, as_tuple=False).view(-1).cpu().tolist()
        )
        if bridge_or_fallback_locals:
            parts.append(base_failed_batch[self._flat_group_indices(bridge_or_fallback_locals, rollout_n)])
        if train_strategy_groups:
            strategy_group_locals = list(range(len(failed_prompt_indices))) if teacher_fallback_aux_group != "none" else torch.nonzero(
                strategy_success, as_tuple=False
            ).view(-1).cpu().tolist()
            if strategy_group_locals:
                parts.append(strategy_training_batch[self._flat_group_indices(strategy_group_locals, rollout_n)])
        final_batch = DataProto.concat(parts)
        self._drop_rollout_log_probs(final_batch)
        self._assign_group_uids(final_batch, rollout_n)
        route_metrics.update(
            {
                "guided_grpo/strategy_success_count": int(torch.sum(strategy_success).item()),
                "guided_grpo/teacher_fallback_count": teacher_fallback_count,
                "guided_grpo/final_group_count": len(final_batch) // rollout_n,
                "guided_grpo/train_strategy_groups": int(train_strategy_groups),
                "guided_grpo/auxiliary_group_loss_weight": self._guided_auxiliary_group_loss_weight(),
                "guided_grpo/bridge_success_to_base": int(bridge_success_to_base),
                "guided_grpo/teacher_fallback_aux_group": self._guided_teacher_fallback_aux_group(),
            }
        )
        route_metrics["guided_grpo/question_dump_record_count"] = self._write_guided_question_dump_records(question_dump_records)
        return final_batch, route_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                keep_raw_prompt_for_guided_dump = self._guided_question_dump_enabled()
                if "raw_prompt" in batch.non_tensor_batch and not (self._guided_repair_enabled() or keep_raw_prompt_for_guided_dump):
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if self._guided_grpo_enabled():
                        batch, guided_grpo_metrics = self._build_guided_grpo_training_batch(batch, gen_batch_output)
                        metrics.update(guided_grpo_metrics)
                    else:
                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            # Get lambda parameters from config
                            lambda_init = self.config.algorithm.get('lambda_init', 0.8)
                            lambda_final = self.config.algorithm.get('lambda_final', 0.0)
                            # Calculate total_steps if not already set
                            if not hasattr(self, 'total_training_steps'):
                                self.total_training_steps = self.config.trainer.total_epochs * len(self.train_dataloader)
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn,
                                global_step=self.global_steps,
                                total_steps=self.total_training_steps,
                                lambda_init=lambda_init,
                                lambda_final=lambda_final
                            )

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            for key, values in reward_extra_infos_dict.items():
                                try:
                                    arr = np.asarray(values, dtype=np.float64)
                                except (TypeError, ValueError):
                                    continue
                                if arr.size == 0:
                                    continue
                                finite = arr[np.isfinite(arr)]
                                if finite.size == 0:
                                    continue
                                metric_key = re.sub(r"[^0-9A-Za-z_/@.-]+", "_", str(key))
                                metrics[f"train_reward_extra/{metric_key}_mean"] = float(np.mean(finite))
                        if not self._guided_grpo_enabled():
                            dump_count = self._dump_normal_grpo_question_records(
                                batch,
                                int(self.config.actor_rollout_ref.rollout.n),
                                reward_tensor.sum(-1),
                            )
                            if dump_count:
                                metrics["guided_grpo/question_dump_record_count"] = dump_count

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        advantage_reward_backup = None
                        if self._guided_advantage_bucket_enabled():
                            if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
                                raise ValueError("Guided GRPO advantage bucket is only supported with algorithm.adv_estimator=grpo.")
                            advantage_reward_backup = batch.batch["token_level_rewards"]
                            bucket_rewards, bucket_metrics = self._bucket_guided_advantage_rewards(batch)
                            batch.batch["token_level_rewards"] = bucket_rewards
                            metrics.update(bucket_metrics)

                        try:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            )
                        finally:
                            if advantage_reward_backup is not None:
                                batch.batch["token_level_rewards"] = advantage_reward_backup

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            batch.meta_info["off_policy_gamma"] = float(self._guided_grpo_cfg().get("off_policy_gamma", 0.1))
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
