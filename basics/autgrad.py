import math


class Value:
    def __init__(self, data: float) -> None:
        self.data = data
        self.prev_nodes = []
        self.op = None
        self.grad = 0.0
        self._backward = None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data)
        out.prev_nodes = [self, other]
        out.op = "+"
        return out

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data - other.data)
        out.prev_nodes = [self, other]
        out.op = "-"
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data)
        out.prev_nodes = [self, other]
        out.op = "*"
        return out

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data / other.data)
        out.prev_nodes = [self, other]
        out.op = "/"
        return out

    def tanh(self):
        out = Value(math.tanh(self.data))
        out.prev_nodes = [self]
        out.op = "tanh"
        return out

    def relu(self):
        out = Value(max(self.data, 0))
        out.prev_nodes = [self]
        out.op = "relu"
        return out

    def __pow__(self, n: int):
        out = Value(pow(self.data, n))
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

    # .def("backPropogate",
    def backward(self):
        pass

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"


if __name__ == "__main__":
    a = Value(9)
    b = a + 90
    c = 10 + a
    d = 10 - a
    e = 10 / a
    print(b)
    print(c)
    print(d)
    print(e)
