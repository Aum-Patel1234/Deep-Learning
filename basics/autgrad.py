import math
from re import S
from typing import List, Self


class Value:
    def __init__(self, data: float) -> None:
        self.data = data
        self.prev_nodes: List[Self] = []
        self.op: str | None = None
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data)

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        out.prev_nodes = [self, other]
        out.op = "+"

        return out

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data - other.data)

        def backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = backward
        out.prev_nodes = [self, other]
        out.op = "-"

        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data)

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        out.prev_nodes = [self, other]
        out.op = "*"
        return out

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data / other.data)

        def backward():
            self.grad += (1.0 / other.data) * out.grad
            other.grad += (-self.data) / (other.data**2) * out.grad

        out._backward = backward
        out.prev_nodes = [self, other]
        out.op = "/"
        return out

    def tanh(self):
        out = Value(math.tanh(self.data))

        def backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = backward
        out.prev_nodes = [self]
        out.op = "tanh"
        return out

    def relu(self):
        out = Value(max(self.data, 0))

        def backward():
            if self.data > 0:
                self.grad += out.grad

        out._backward = backward
        out.prev_nodes = [self]
        out.op = "relu"
        return out

    def __pow__(self, n: int):
        out = Value(pow(self.data, n))

        def backward():
            self.grad += (n * (self.data ** (n - 1))) * out.grad

        out._backward = backward
        out.prev_nodes = [self]
        out.op = "**"
        return out

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        other = Value(other)
        return other - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        other = Value(other)
        return other / self

    def topoSort(self) -> List[Self]:
        # TODO:
        topo: List[Self] = []
        return topo

    def backward(self):
        self.grad = 1.0
        topo = self.topoSort()
        print(f"topo sort start (len={len(topo)}) - ")
        for node in reversed(topo):
            node._backward()
            print(node)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"


if __name__ == "__main__":
    a = Value(9)
    b = Value(90)
    c = 10 + a
    d = b - a
    f = 10 * a
    e = a.relu()
    g = d**2
    h = 10 / a

    # c.grad = 1.0
    # c.backward()
    # print(c)
    # print(a)
    g.backward()
