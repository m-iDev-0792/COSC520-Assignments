import matplotlib.pyplot as plt
from matplotlib import animation, patches
import numpy as np
from kd_tree_2d import KDTree2D


def visualize_query_with_pruning(tree: KDTree2D, target, delay=800):
    """
    Visualize KDTree nearest neighbor search step by step with real pruning.
    - Light blue = region visited
    - Light gray = region pruned
    - Green star = current best NN
    - Red dot = query point
    """
    points = tree.points
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Plot static elements
    ax.scatter(points[:, 0], points[:, 1], c="gray", s=40, label="Points")
    ax.scatter([target[0]], [target[1]], c="red", s=100, label="Query")
    ax.legend(loc="upper right")

    # Draw KD-tree partitions
    def draw_splits(node, bounds):
        if node is None:
            return
        axis = node.axis
        x0, x1, y0, y1 = bounds
        px, py = tree.points[node.point_idx]
        if axis == 0:
            ax.plot([px, px], [y0, y1], "k--", lw=0.8)
            left_bounds = (x0, px, y0, y1)
            right_bounds = (px, x1, y0, y1)
        else:
            ax.plot([x0, x1], [py, py], "k--", lw=0.8)
            left_bounds = (x0, x1, y0, py)
            right_bounds = (x0, x1, py, y1)
        draw_splits(node.left, left_bounds)
        draw_splits(node.right, right_bounds)

    draw_splits(tree.root, (0, 1, 0, 1))

    # Helper: squared distance between a point and a bounding box
    def min_dist2_to_box(point, bounds):
        x0, x1, y0, y1 = bounds
        px, py = point
        dx = 0.0 if x0 <= px <= x1 else min(abs(px - x0), abs(px - x1))
        dy = 0.0 if y0 <= py <= y1 else min(abs(py - y0), abs(py - y1))
        return dx * dx + dy * dy

    # Recursive nearest neighbor with pruning, logging each step
    frames = []

    def dist2(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    def search(node, bounds, best_idx, best_d2):
        if node is None:
            return best_idx, best_d2

        # Record the step (visit start)
        frames.append(("visit", node, bounds, best_idx, best_d2))

        pt = tree.points[node.point_idx]
        d2 = dist2(pt, target)
        if d2 < best_d2:
            best_d2 = d2
            best_idx = node.point_idx

        axis = node.axis
        x0, x1, y0, y1 = bounds
        px, py = pt

        if axis == 0:
            left_bounds = (x0, px, y0, y1)
            right_bounds = (px, x1, y0, y1)
            val, pivot = target[0], px
        else:
            left_bounds = (x0, x1, y0, py)
            right_bounds = (x0, x1, py, y1)
            val, pivot = target[1], py

        near, far = (node.left, node.right) if val < pivot else (node.right, node.left)
        near_bounds, far_bounds = (left_bounds, right_bounds) if val < pivot else (right_bounds, left_bounds)

        # Visit near side first
        best_idx, best_d2 = search(near, near_bounds, best_idx, best_d2)

        # Check if we need to explore the far side
        min_d2_far = min_dist2_to_box(target, far_bounds)
        if min_d2_far <= best_d2:
            best_idx, best_d2 = search(far, far_bounds, best_idx, best_d2)
        else:
            # Record pruning event
            frames.append(("prune", None, far_bounds, best_idx, best_d2))

        # Record step after return
        frames.append(("return", node, bounds, best_idx, best_d2))
        return best_idx, best_d2

    # Run the search once to collect frames
    best_idx, best_d2 = search(tree.root, (0, 1, 0, 1), None, float("inf"))

    # --------- Animation state ----------
    active_patch = None
    best_marker = None

    def update(frame_idx):
        nonlocal active_patch, best_marker
        action, node, bounds, best_idx, best_d2 = frames[frame_idx]
        x0, x1, y0, y1 = bounds

        # Remove previous highlight
        if active_patch is not None:
            active_patch.remove()

        color = "lightskyblue" if action == "visit" else "lightgray"
        alpha = 0.3 if action == "visit" else 0.2

        # Draw current region
        active_patch = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=color, alpha=alpha, lw=0
        )
        ax.add_patch(active_patch)

        # Update best marker
        if best_idx is not None:
            if best_marker is not None:
                best_marker.remove()
            best_marker = ax.scatter(
                [tree.points[best_idx, 0]],
                [tree.points[best_idx, 1]],
                c="green", s=120, marker="*", label="Best NN"
            )

        return [active_patch, best_marker] if best_marker else [active_patch]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=delay, blit=False, repeat=False
    )
    plt.show()


# Example usage
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    pts = rng.random((25, 2))
    tree = KDTree2D(pts, split_method="variance")
    target = (0.4, 0.7)
    visualize_query_with_pruning(tree, target)
