import jax
from jax import device_put
from jax.lib import xla_bridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset import generate_random_data, to_jax
from innovation import InnovationTracker
from population import Population
from settings import (
    CHOICE,
    GENERATION_SIZE,
    POPULATION_SIZE,
    TEST_SIZE,
    TRAIN_SIZE,
    TRAIN_STEPS,
)
from util import vector_to_weights
from visualize import plot_decision_boundary, visualize_network_graphviz

print("JAX backend:", xla_bridge.get_backend().platform)
print("Devices:", jax.devices())

train, test = generate_random_data(
    train_size=TRAIN_SIZE, test_size=TEST_SIZE, choice=CHOICE, noise=0.1
)
X_train, y_train = to_jax(train)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
X_train = device_put(X_train)
y_train = device_put(y_train)
X_valid = device_put(X_valid)
y_valid = device_put(y_valid)
X_test, y_test = to_jax(test)
X_test = device_put(X_test)
y_test = device_put(y_test)


tracker = InnovationTracker()
pop = Population(size=POPULATION_SIZE, input_size=2, output_size=1, tracker=tracker)

for gen in range(GENERATION_SIZE):
    pop.evolve(X_train, y_train, X_valid, y_valid, train_steps=TRAIN_STEPS)
    best = max(
        [i for i in pop.individuals if i.fitness is not None], key=lambda i: i.fitness
    )
    if gen % 10 == 0:
        print(
            f"[Gen {gen}] Best fitness: {best.fitness:.4f}, Species: {len(pop.species)}"
        )
        print(best.net.connections)
        print("\n== Best Network ==")
        w_final = vector_to_weights(best.params_vec, best.net.connections)
        y_pred = [1 if best.net.forward(xi, w_final) > 0.5 else 0 for xi in X_test]
        print(f"Current Test Accuracy Score: {accuracy_score(y_test, y_pred)}")

print("\n== Best Network ==")
w_final = vector_to_weights(best.params_vec, best.net.connections)
y_pred = [1 if best.net.forward(xi, w_final) > 0.5 else 0 for xi in X_test]
visualize_network_graphviz(best.net, filename=f"{CHOICE}_graphviz")
y_train_pred = [1 if best.net.forward(xi, w_final) > 0.5 else 0 for xi in X_train]
y_test_pred = [1 if best.net.forward(xi, w_final) > 0.5 else 0 for xi in X_test]
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)

# プロット実行
plot_decision_boundary(
    best.net,
    w_final,
    X_train,
    y_train,
    X_test,
    y_test,
    acc_train,
    acc_test,
    title=f"db{CHOICE}_graphviz",
)
