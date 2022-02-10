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
"""Tests for Keras python-based idempotent saving functions (experimental)."""
import os
import sys

import keras
from keras.saving import experimental_saving
from keras.utils import io_utils
import numpy as np
import tensorflow.compat.v2 as tf


class NewSavingTest(tf.test.TestCase):

  def test_new_saving(self):
    train_step_message = 'This is my training step'
    temp_dir = os.path.join(self.get_temp_dir(), 'my_model')

    @keras.utils.generic_utils.register_keras_serializable('MyDense')
    class MyDense(keras.layers.Dense):

      def two(self):
        return 2

    @keras.utils.generic_utils.register_keras_serializable('CustomModelX')
    class CustomModelX(keras.Model):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1)

      def call(self, inputs):
        return self.dense1(inputs)

      def train_step(self, data):
        tf.print(train_step_message)
        x, y = data
        with tf.GradientTape() as tape:
          y_pred = self(x)
          loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {}

      def one(self):
        return 1

    subclassed_model = CustomModelX()
    subclassed_model.compile(
        optimizer='adam', loss=['mse', keras.losses.MeanSquaredError()])

    subclassed_model._save_new(temp_dir)

    x = np.random.random((100, 32))
    y = np.random.random((100, 1))
    subclassed_model.fit(x, y, epochs=1)
    subclassed_model._save_new(temp_dir)

    loaded_model = experimental_saving.load(temp_dir)

    io_utils.enable_interactive_logging()
    # `tf.print` writes to stderr.
    with self.captureWritesToStream(sys.stderr) as printed:
      loaded_model.fit(x, y, epochs=1)
      self.assertRegex(printed.contents(), train_step_message)

    # Check that the custom class does get used.
    self.assertIsInstance(loaded_model, CustomModelX)
    # Check that the custom method is available.
    self.assertEqual(loaded_model.one(), 1)
    self.assertEqual(loaded_model.dense1.two(), 2)

    self.assertIs(loaded_model.optimizer.__class__,
                  keras.optimizers.optimizer_v2.adam.Adam)
    self.assertIs(loaded_model.compiled_loss.__class__,
                  keras.engine.compile_utils.LossesContainer)
    self.assertIs(loaded_model.compiled_loss._losses[0].__class__,
                  keras.losses.LossFunctionWrapper)
    self.assertIs(loaded_model.compiled_loss._losses[1].__class__,
                  keras.losses.MeanSquaredError)


if __name__ == '__main__':
  tf.test.main()
