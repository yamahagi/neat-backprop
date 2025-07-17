import math

import jax
import jax.numpy as jnp

import optax
from settings import ADAM_PARAM, PENALTY_FACTOR
from util import vector_to_weights, weights_to_vector

rng = jax.random.PRNGKey(0)


class Individual:
    def __init__(self, net):
        self.net = net
        self.template_weights = {k: w for k, (w, _) in net.connections.items()}
        self.params_vec = weights_to_vector(net.connections)
        self.optimizer = optax.adam(ADAM_PARAM)  # 安定性の高いオプティマイザに変更
        self.opt_state = self.optimizer.init(self.params_vec)
        self.fitness = None
        self.needs_training = True

        # JITされた関数の定義
        def loss_fn(params, x_batch, y_batch):
            weights = vector_to_weights(params, self.net.connections)

            def single_forward(x, y):
                output = self.net.forward(x, weights)
                loss = jnp.mean((output - y) ** 2)
                loss = jnp.where(jnp.any(jnp.isnan(output)), 1e6, loss)
                return loss

            batch_loss = jax.vmap(single_forward)(x_batch, y_batch)
            return jnp.mean(batch_loss)

        self._loss_fn = jax.jit(loss_fn)
        self._loss_and_grad_fn = jax.jit(jax.value_and_grad(self._loss_fn))

    def train(self, X_train, y_train, steps=10, batch_size=4):
        global rng
        N = len(X_train)
        for _ in range(steps):
            # ランダムにミニバッチ抽出
            rng, _ = jax.random.split(rng)
            indices = jax.random.choice(
                key=rng, a=N, shape=(1, batch_size), replace=False
            )
            x_batch = X_train[indices[0]]
            y_batch = y_train[indices[0]]

            # バッチ全体で損失と勾配を計算
            loss, grads = self._loss_and_grad_fn(self.params_vec, x_batch, y_batch)

            # パラメータ更新
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params_vec = optax.apply_updates(self.params_vec, updates)
            self.params_vec = jnp.clip(self.params_vec, -10.0, 10.0)
        self.needs_training = False

    def sync_weights(self):
        updated_weights = vector_to_weights(self.params_vec, self.net.connections)
        for (src_dst), weight in updated_weights.items():
            old_weight, innov = self.net.connections[src_dst]
            self.net.connections[src_dst] = (weight, innov)

    def evaluate(self, X_valid, y_valid):
        total_loss = self._loss_fn(self.params_vec, X_valid, y_valid)
        complexity_penalty = PENALTY_FACTOR * (
            len(self.net.nodes) + len(self.net.connections)
        )
        self.fitness = -float(total_loss) * math.sqrt(1 + complexity_penalty)
        self.sync_weights()
