# SPDX-License-Identifier: Apache-2.0
"""Common pool-id contract for GLM index allocation, production, and attention."""
from vllm_ascend import envs


def indexer_pool_id_enabled(index_kpool=4, logical_block_size=None):
    if not envs.VLLM_ASCEND_ENABLE_INDEXER_POOL_ID or index_kpool != 4:
        return False
    from vllm.config import get_current_vllm_config
    try:
        config = get_current_vllm_config()
    except (AssertionError, RuntimeError):
        config = None
    if config is not None:
        hf = config.model_config.hf_text_config
        if getattr(hf, "index_kpool", None) != 4:
            return False
        if logical_block_size is None:
            logical_block_size = config.cache_config.block_size
        from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
        if get_current_hardware_profile().device_adaptor_family is not DeviceAdaptorFamily.STANDARD:
            return False
    return (logical_block_size or 128) % 4 == 0


def sparse_index_block_size(indices, cache_block_size):
    return 4 if indexer_pool_id_enabled(logical_block_size=cache_block_size) else 1
