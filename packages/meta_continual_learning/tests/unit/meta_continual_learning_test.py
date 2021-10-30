# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------


import unittest

import torch

from nupic.research.frameworks.meta_continual_learning.experiments import (
    MetaContinualLearningExperiment,
)
from nupic.research.frameworks.meta_continual_learning.maml_utils import clone_model
from nupic.research.frameworks.meta_continual_learning.models import OMLNetwork

# Retrieve function that updates params in place.
# This enables taking gradients of gradients.
update_params = MetaContinualLearningExperiment.update_params


class Quadratic(torch.nn.Module):
    """Quadratic layer: Computes W^T W x"""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([
            [0.94, 0.07],
            [0.40, 0.21]
        ]))

    def forward(self, x):
        """Compute W^T W x"""
        out = torch.matmul(self.weight, x)
        out = torch.matmul(self.weight.transpose(1, 0), out)
        return out


class GradsOfGradsTest(unittest.TestCase):
    """
    Perform tests for taking gradients of gradients. Specifically, this uses a
    quadratic layer as a test example since it yields non-trivial results (unlike a
    linear layer) and enables manually calculated expected values.
    """

    def setUp(self):
        # Input passed to quadratic function: a W^T W x
        self.left_input = torch.tensor([[1., 1.]], requires_grad=True)  # a
        self.right_input = torch.tensor([[0.32], [0.72]], requires_grad=True)  # x

        # Expected gradient of gradient for quadratic layer updated via SGD.
        self.expected_grad = torch.tensor([[0.3972, 0.6762], [0.2869, 0.4458]])

    def test_auto_grad_with_quadratic_function(self):
        """
        Test use of pytorch's autograd to keep track of gradients of gradients.

        This validates the hand derived solution of self.expected_grad.
        """

        # Use the predefined weight from the quadratic layer.
        weight = Quadratic().weight

        # First forward pass: L = a W^T W x
        a = self.left_input
        x = self.right_input
        loss = torch.matmul(weight, x)
        loss = torch.matmul(weight.transpose(1, 0), loss)
        loss = torch.matmul(a, loss)

        # First backward pass.
        loss.backward(retain_graph=True, create_graph=True)
        weight.grad

        # --------------
        # 1st Derivative
        # --------------

        # Compare the manually computed expected 2nd derivative with pytorch's autograd.
        # Using the identity from
        #    `https://en.wikipedia.org/wiki/Matrix_calculus#Scalar-by-matrix_identities`
        #
        # we have
        #    dL/dW = W · (x·a + (x·a)^T)
        #
        # where "·" is a matrix product.
        #
        with torch.no_grad():
            m = torch.matmul(x, a)
            m = m + m.transpose(1, 0)
            w_grad_expected = torch.matmul(weight, m)
        self.assertTrue(weight.grad.allclose(w_grad_expected, atol=1e-8))

        # Update weight
        lr = 0.1
        weight2 = weight - lr * weight.grad
        weight2.retain_grad()

        # Zero the gradient of the non-updated weight.
        weight.grad = None

        # Second forward pass: L_2 = a W_2^T W_2 x
        loss2 = torch.matmul(weight2, x)
        loss2 = torch.matmul(weight2.transpose(1, 0), loss2)
        loss2 = torch.matmul(a, loss2)
        loss2.backward()

        # --------------
        # 2nd Derivative
        # --------------

        # Compare the manually derived 2nd derivative with pytorch's autograd.
        # Using the chain rule
        #    dL_2/dW = (dL_2/dW_2) · (dW_2/dW)
        #
        # The term on the right hand side is analogous to dL/dW as above so that
        #    dL_2/dW_2 = W_2 · (x·a + (x·a)^T)
        #
        # For the second term, recall that W_2 = W - lr dL/dW = W - lr W·(x·a + (x·a)^T)
        # This gives,
        #    dW_2/dW  = I - lr(x·a + (x·a)^T)
        #
        # Putting these results together
        #    dL_2/dW = W_2 · (x·a + (x·a)^T) · (I - lr(x·a + (x·a)^T)
        #
        m = torch.matmul(x, a)
        m = m + m.transpose(1, 0)
        w2_grad_expected = torch.matmul(weight2, m)
        w_grad_expected = torch.matmul(w2_grad_expected, (torch.eye(2) - lr * m))
        self.assertTrue(weight.grad.allclose(w_grad_expected, atol=1e-8))
        self.assertTrue(weight2.grad.allclose(w2_grad_expected, atol=1e-8))

        # Compare pytorch's 2nd derivative with the one saved by this test class.
        self.assertTrue(weight.grad.allclose(self.expected_grad, atol=1e-4))

    def test_update_params_with_quadratic_layer(self):
        """
        Test use of update_params function.

        It should update the parameters in a way that enables taking
        a gradient of a gradient.
        """

        quad = Quadratic()
        quad_clone = clone_model(quad)
        x = self.right_input

        # First forward and backward pass: akin to the inner loop in meta-cl.
        out = quad_clone(x)
        loss = out.sum()
        update_params(quad_clone.named_parameters(), quad_clone, loss, lr=0.1)

        # Second forward and backward pass: akin to the outer loop in meta-cl.
        out2 = quad_clone(x)
        loss2 = out2.sum()
        loss2.backward()

        # Validate gradient on original weight parameter.
        self.assertTrue(torch.allclose(quad.weight.grad, self.expected_grad, atol=1e-4))


class CloneModelTest(unittest.TestCase):

    def test_clone_model_params_are_deepcopied(self):
        """
        Validate that the parameters cloned from the OMLNetwork have been deepcopied -
        the clone and original model should share no data between there params.
        """

        oml = OMLNetwork(num_classes=10)
        oml_data_ptrs = [p.data_ptr() for p in oml.parameters()]

        # The clone and original model should share no data between there params.
        oml_clone = clone_model(oml)
        clone_data_ptrs = [p.data_ptr() for p in oml_clone.parameters()]

        overlap_ptrs = set(clone_data_ptrs) & set(oml_data_ptrs)
        self.assertEqual(len(overlap_ptrs), 0)

        # Params from the property named_fast_params should solely belong to the clone.
        fast_data_ptrs = [p.data_ptr() for n, p in oml_clone.named_fast_params.items()]
        self.assertTrue(set(fast_data_ptrs) <= set(clone_data_ptrs))

    def test_copy_hooks_test(self):
        """
        Test whether all model hooks are persevered when `keep_hooks=True` is passed to
        `clone_model`.
        """

        # Define module and param hook, both forward and backward.
        class Hook:
            """Stateful hook"""
            def __init__(self):
                self.state = torch.tensor([0])

            def __call__(self, *args):
                self.state.add_(1)

        forward_hook = Hook()
        backward_hook = Hook()
        forward_pre_hook = Hook()
        weight_hook = Hook()

        # Save the state for later reference.
        s1 = forward_hook.state
        s2 = backward_hook.state
        s3 = forward_pre_hook.state
        s4 = weight_hook.state

        # Register these hooks.
        quad = torch.nn.Linear(2, 2)
        quad.register_forward_hook(forward_hook)
        quad.register_backward_hook(backward_hook)
        quad.register_forward_pre_hook(forward_pre_hook)
        quad.weight.register_hook(weight_hook)

        # Clone the model and copy none of it's hooks.
        cloned = clone_model(quad, keep_hooks=False)

        # Run one forward and backward pass.
        x = torch.rand(2 , 2)
        y = cloned(x)
        y.sum().backward()

        # Validate that none of the original state has been updated.
        self.assertEqual(s1.item(), 0)
        self.assertEqual(s2.item(), 0)
        self.assertEqual(s3.item(), 0)
        self.assertEqual(s4.item(), 1)  # keep_hooks only refers to module hooks

        # Clone the model and copy all of it's hooks.
        cloned = clone_model(quad, keep_hooks=True)

        # Run one forward and backward pass.
        x = torch.rand(2 , 2)
        y = cloned(x)
        y.sum().backward()

        # Validate that the original state has been updated.
        self.assertEqual(s1.item(), 1)
        self.assertEqual(s2.item(), 1)
        self.assertEqual(s3.item(), 1)
        self.assertEqual(s4.item(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
