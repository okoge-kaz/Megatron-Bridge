# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os

import pytest

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.gemma import (
    Gemma2ModelProvider2B,
    Gemma2ModelProvider9B,
    Gemma2ModelProvider27B,
)
from tests.functional_tests.utils import compare_provider_configs


HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER = {
    "google/gemma-2-2b": Gemma2ModelProvider2B,
    "google/gemma-2-9b": Gemma2ModelProvider9B,
    "google/gemma-2-27b": Gemma2ModelProvider27B,
}

ROOT_PATH: str = "/home/TestData/megatron_bridge/hf_home"

HF_MODEL_ID_PATH_TO_MODEL_PROVIDER = {
    os.path.join(ROOT_PATH, hf_model_id): provider_class
    for hf_model_id, provider_class in HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()
}


class TestGemma2ModelProviderMapping:
    """Test that bridge provider configs are equivalent to predefined provider configs."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_PATH_TO_MODEL_PROVIDER.items()))
    def test_bridge_vs_predefined_provider_config_equivalence(self, hf_model_id, provider_class):
        """Test that bridge converted provider config matches predefined provider config."""
        # Create bridge from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Create predefined provider
        predefined_provider = provider_class()

        # Compare configs
        compare_provider_configs(converted_provider, predefined_provider, hf_model_id)
