"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from numpy.lib.index_tricks import AxisConcatenator

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy


import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)
        

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        a_pow = power_scalar(a, self.scalar-1)
        return out_grad * mul_scalar(a_pow, self.scalar) 


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b)

    # out_grad * x , out_grad * (- x/y^2)
    def gradient(self, out_grad, node):
        a, b = node.inputs
        a_grad = out_grad / b
        b_sq   = b * b
        res    = (a / b_sq) * -1
        b_grad = out_grad * res
        return a_grad, b_grad



def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return out_grad / self.scalar

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
          return array_api.swapaxes(a, a.ndim-2, a.ndim-1)
        else:
          return array_api.swapaxes(a, self.axes[0], self.axes[1])


    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)


    def gradient(self, out_grad, node):
        # Retrieve the shape of the original input associated with the provided node
        original_shape = node.inputs[0].shape
        
        # Initialize a list to mark dimensions to be summed over
        shrink_dims = [i if ori == cur else -1 for i, (ori, cur) in enumerate(zip(reversed(original_shape), reversed(self.shape)))]
        
        # Convert marked dimensions into a tuple, excluding those marked as -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        
        # Sum out_grad along the dimensions specified by shrink_dims
        summed_grad = out_grad.sum(shrink_dims)
        
        # Reshape the summed gradient to match the shape of the original input
        reshaped_grad = summed_grad.reshape(original_shape)
        
        return reshaped_grad

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        # Create a new list to store the modified shape of the gradient tensor
        new_shape = list(node.inputs[0].shape)

        # Determine the axes along which to reshape the gradient tensor
        # If self.axes is None, reshape along all axes, otherwise reshape along specified axes
        axes = range(len(new_shape)) if self.axes is None else self.axes

        # Set the corresponding dimensions in new_shape to 1 for each axis
        for axis in axes:
            new_shape[axis] = 1

        # Reshape the gradient tensor to have the modified shape
        reshaped_grad = out_grad.reshape(new_shape)

        # Broadcast the reshaped gradient tensor to match the shape of the original input
        broadcasted_grad = reshaped_grad.broadcast_to(node.inputs[0].shape)

        # Return the reshaped and broadcasted gradient tensor
        return broadcasted_grad

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs

        # Compute the gradients of the loss with respect to lhs and rhs matrices
        lgrad = matmul(out_grad, rhs.transpose())  # Gradient with respect to lhs
        rgrad = matmul(lhs.transpose(), out_grad)  # Gradient with respect to rhs

        # Ensure lgrad and rgrad have the same number of dimensions as lhs and rhs, respectively
        lgrad = lgrad.sum(axis=tuple(range(len(lgrad.shape) - len(lhs.shape)))) if len(lhs.shape) < len(lgrad.shape) else lgrad
        rgrad = rgrad.sum(axis=tuple(range(len(rgrad.shape) - len(rhs.shape)))) if len(rhs.shape) < len(rgrad.shape) else rgrad

        # Return the gradients with respect to lhs and rhs matrices
        return lgrad, rgrad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)

'''
class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
'''
