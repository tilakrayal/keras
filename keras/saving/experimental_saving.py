# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Keras python-based idempotent saving functions (experimental)."""
import json
import os
from keras.saving.saved_model import json_utils
from keras.utils import generic_utils
import tensorflow.compat.v2 as tf

_CONFIG_FILE = 'config.keras'


def load(dirpath):
  """Load a saved python model."""
  file_path = os.path.join(dirpath, _CONFIG_FILE)
  with tf.io.gfile.GFile(file_path, 'r') as f:
    config_json = f.read()
  config_dict = json.loads(config_json)
  model_class = generic_utils.get_custom_objects_by_name(
      config_dict['model_class'])
  model = model_class.from_config(config_dict['model_config'])
  return model


def save(model, dirpath):
  """Save a saved python model."""
  if not tf.io.gfile.exists(dirpath):
    tf.io.gfile.mkdir(dirpath)
  file_path = os.path.join(dirpath, _CONFIG_FILE)
  model_config_dict = model.get_config()
  class_registered_name = generic_utils.get_registered_name(model.__class__)
  assert class_registered_name is not None
  config_dict = {
      'model_class': class_registered_name,
      'model_config': model_config_dict
  }
  config_json = json.dumps(config_dict, cls=json_utils.Encoder)
  with tf.io.gfile.GFile(file_path, 'w') as f:
    f.write(config_json)
