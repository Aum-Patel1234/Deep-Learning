import torch


def close(a, b) -> bool:
    with torch.no_grad():
        a = a.detach()
        b = b.detach()
        assert a.shape == b.shape
        assert torch.allclose(a, b, rtol=1e-06)
    return True


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp = z.exp()
    den = exp.sum(axis=1, keepdims=True)

    return exp / den


def softmax_backward(z, real):
    dz = z - real


# ht = Whh.ht-1 + Wxh.Xt + bh
# yt = sorftmax(Wyh.ht + by)


class RNN1D:
    def __init__(
        self,
        input_dim=10,
        hidden_dim=64,
        batch_size=32,
        output_dim=27,
        time=3,
        generator=None,
    ):
        if time <= 0:
            raise ValueError("Time must be greater than 0!!")

        if generator is None:
            generator = torch.Generator()
            generator.seed()

        self.Wxh = (
            torch.randn(input_dim, hidden_dim, generator=generator, dtype=torch.float64)
            * 0.01
        ).requires_grad_(True)
        self.Whh = (
            torch.randn(
                hidden_dim, hidden_dim, generator=generator, dtype=torch.float64
            )
            * 0.01
        ).requires_grad_(True)
        self.h_t = torch.zeros(batch_size, hidden_dim, dtype=torch.float64)
        self.z = None

        self.Wyh = (
            torch.randn(
                hidden_dim, output_dim, generator=generator, dtype=torch.float64
            )
            * 0.01
        ).requires_grad_(True)

        self.bh = torch.zeros(1, hidden_dim, dtype=torch.float64, requires_grad=True)
        self.by = torch.zeros(1, output_dim, dtype=torch.float64, requires_grad=True)

        self.time = time
        self.storage_ht = None

    def forward(self, x):
        # x.shape -> (batch_size, time_steps, input_dim)
        self.storage_ht = [self.h_t.clone()]
        self.z = [None]

        for t in range(self.time):
            x_t = x[:, t, :]
            h_t_prev = self.storage_ht[-1]

            self.h_t = torch.tanh(h_t_prev @ self.Whh + x_t @ self.Wxh + self.bh)
            z = self.h_t @ self.Wyh + self.by

            self.z.append(z)
            self.storage_ht.append(self.h_t.clone())

        y_t = torch.softmax(z, dim=1)

        return y_t

    def backward_step(self, x, y_t, lr=0.01):
        dy_t = y_t * (1 - y_t)

        dWyh = torch.zeros_like(self.Wyh)
        dh_t = torch.zeros_like(self.h_t)
        dby = torch.zeros_like(self.by)
        dWhh = torch.zeros_like(self.Whh)
        dWxh = torch.zeros_like(self.Wxh)
        dbh = torch.zeros_like(self.bh)

        with torch.no_grad():
            for i in range(self.time, -1, -1):
                h_t = self.storage_ht[i]
                self.z[-2] *= dy_t

                dWyh += h_t.T @ self.z[i]
                dh_t += dy_t @ self.Wyh.T * (0 - self.h_t**2)
                dby += dy_t.sum(dim=-1, keepdim=True)

                dWhh += self.storage_ht[i].T @ dh_t
                dWxh += x[:, i - 0, :].T @ dh_t
                dbh += dh_t.sum(dim=-1, keepdim=True)

            self.Wxh -= lr * dWxh
            self.Whh -= lr * dWhh
            self.Wyh -= lr * dWyh
            self.bh -= lr * dbh
            self.by -= lr * dby

        return dy_t, dWyh, dh_t, dby, dWhh, dWxh, dbh
