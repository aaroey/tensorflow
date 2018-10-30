# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class DefFunctionTest(test.TestCase):

  def testNoVariables(self):

    @def_function.function
    def fn(x):
      return 2 * x

    self.assertAllEqual(fn(constant_op.constant(4.0)), 8.0)

  def testFailIfVariablesAreCreatedMoreThanOnce(self):

    @def_function.function
    def fn(x):
      return variables.Variable(1.0) + x

    with self.assertRaises(ValueError):
      fn(1.0)

  def testFailIfVariablesAreCreatedMoreThanOnceNoWeakRef(self):
    state = []

    @def_function.function
    def fn(x):
      state.append(variables.Variable(1.0))
      return state[-1] + x

    with self.assertRaises(ValueError):
      fn(1.0)

  def testCorrectVariableCreation(self):

    state = []

    @def_function.function
    def fn(x):
      if not state:
        state.append(variables.Variable(2.0))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)
    self.assertAllEqual(fn(constant_op.constant(3.0)), 6.0)

  def testFunctionInitializer(self):

    state = []

    @def_function.function
    def fn(x):
      if not state:
        state.append(variables.Variable(lambda: 2.0))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)

  def testVariableInitializerNotConstant(self):

    state = []

    @def_function.function
    def fn(x):
      if not state:
        state.append(variables.Variable(2.0 * x))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)
    self.assertAllEqual(fn(constant_op.constant(3.0)), 6.0)

  def testLegacyGraphModeVariables(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      state = []

      @def_function.function
      def fn(x):
        if not state:
          state.append(variables.Variable(2.0))
        return state[0] * x

      result = fn(3.0)

      sess.run(variables.global_variables_initializer())
      self.assertAllEqual(sess.run(state[0]), 2.0)
      self.assertAllEqual(sess.run(result), 6.0)

  def testLegacyGraphModeVariablesNonTrivialInitializer(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      state = []

      @def_function.function
      def fn(x):
        if not state:
          two = constant_op.constant(2.0)
          four = two * two
          two_again = math_ops.sqrt(four)
          state.append(variables.Variable(two_again + four))
        return state[0] * x

      result = fn(3.0)

      sess.run(variables.global_variables_initializer())
      self.assertAllEqual(sess.run(state[0]), 6.0)
      self.assertAllEqual(sess.run(result), 18.0)

  def testLegacyGraphModeInputDependentInitializerFails(self):
    with ops.Graph().as_default():
      state = []

      @def_function.function
      def fn(x):
        if not state:
          state.append(variables.Variable(2.0 * x))
        return state[0] * x

      with self.assertRaises(ValueError):
        fn(constant_op.constant(3.0))

  def testMethod(self):

    class MyModel(object):

      def __init__(self):
        self.var = None

      @def_function.function
      def apply(self, x):
        if self.var is None:
          self.var = variables.Variable(2.0)
        return self.var * x

    m0 = MyModel()
    self.assertAllEqual(m0.apply(3.0), 6.0)
    # Calling twice to exercise that we do not recreate variables.
    m0.var.assign(3.0)
    self.assertAllEqual(m0.apply(3.0), 9.0)

    m1 = MyModel()
    self.assertAllEqual(m1.apply(3.0), 6.0)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
