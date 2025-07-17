import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphviz import Digraph
from network import EvolvingNEATNetwork


def visualize_network(net: EvolvingNEATNetwork, title="NEAT Network"):
    G = nx.DiGraph()

    node_colors = {
        "input": "skyblue",
        "output": "lightgreen",
        "hidden": "lightcoral",
        "bias": "yellow",
    }

    # ノードの追加
    for nid, node in net.nodes.items():
        G.add_node(
            nid,
            label=f"{nid}\n{node.activation}",
            color=node_colors.get(node.type, "gray"),
        )

    # 接続の追加
    for (from_id, to_id), (weight, innov) in net.connections.items():
        G.add_edge(from_id, to_id, weight=round(weight, 2), label=str(round(weight, 2)))

    # レイアウト
    pos = nx.spring_layout(G, seed=42)

    # ノード描画
    node_colors_list = [G.nodes[n]["color"] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=1000)

    # エッジ描画
    nx.draw_networkx_edges(G, pos, arrows=True)

    # ラベル描画
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    edge_labels = {(u, v): f"{d['label']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="gray", font_size=7
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{title}.png")


def plot_decision_boundary(
    net,
    weights,
    X_train,
    y_train,
    X_test,
    y_test,
    accuracy_train,
    accuracy_test,
    title="",
):
    # グリッドを作成
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_min, x_max = -5.5, 5.5
    y_min, y_max = -5.5, 5.5
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # すべてのグリッド点で予測
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array([float(net.forward(x, weights)[0]) for x in grid_points])
    Z = preds.reshape(xx.shape)

    # 背景の決定境界（色）
    plt.clf()
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")

    # 学習データ点のプロット
    train_x = X_train[:, 0]
    train_y = X_train[:, 1]
    plt.scatter(
        train_x,
        train_y,
        c=y_train,
        cmap="coolwarm",
        s=30,
        edgecolors="k",
        alpha=0.7,
        label="train",
    )

    # テストデータ点のプロット
    test_x = X_test[:, 0]
    test_y = X_test[:, 1]
    plt.scatter(
        test_x,
        test_y,
        c=y_test,
        cmap="coolwarm",
        marker="o",
        s=30,
        alpha=0.6,
        edgecolors="none",
        label="test",
    )

    # 軸と精度
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.text(
        x_min + 0.2,
        y_min + 0.5,
        f"train accuracy = {accuracy_train * 100:.1f}%  test accuracy = {accuracy_test * 100:.1f}%",
        fontsize=10,
    )
    # plt.legend()
    plt.savefig(f"{title}.png")


def visualize_network_graphviz(net: EvolvingNEATNetwork, filename="neat_network"):
    dot = Digraph(format="png")
    dot.attr(rankdir="LR")  # 左→右にレイアウト

    # ノードの色と形を設定
    color_map = {
        "input": "skyblue",
        "output": "lightgreen",
        "hidden": "lightcoral",
        "bias": "gold",
    }

    shape_map = {
        "input": "ellipse",
        "output": "ellipse",
        "hidden": "circle",
        "bias": "diamond",
    }

    # ノード追加
    for nid, node in net.nodes.items():
        if node.type == "input":
            label = "input"
        elif node.type == "bias":
            label = "bias"
        elif node.type == "output":
            label = "output"
        else:
            label = f"{nid}\n{node.activation}"
        dot.node(
            str(nid),
            label=label,
            style="filled",
            fillcolor=color_map.get(node.type, "white"),
            shape=shape_map.get(node.type, "circle"),
        )

    # 接続追加
    for (from_id, to_id), (weight, innov) in net.connections.items():
        color = "black" if weight >= 0 else "red"
        label = f"{weight:.2f} / {innov}"
        dot.edge(str(from_id), str(to_id), label=label, color=color)

    dot.render(filename, view=True)
