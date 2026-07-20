from __future__ import annotations

import numpy as np

from two_drnn import adam_update_2drnn, backward_2drnn, create_net_2drnn, forward_2drnn, one_hot_label


def main() -> None:
    input_image = np.array(np.reshape(np.arange(1, 28 * 28 + 1), (28, 28), order="F"), dtype=float)
    m = 7
    p = 4
    f = 8
    h1 = 2
    h2 = 3
    net = create_net_2drnn(p, f, h1, h2, m)
    prob_d, pred_digit, h_final, patch_store, hidden_history = forward_2drnn(net, input_image)
    onehot = one_hot_label(0)
    grad_struct = backward_2drnn(net, input_image, onehot, prob_d, h_final, patch_store, hidden_history)
    net = adam_update_2drnn(net, grad_struct, 0.1)
    print("pred_digit:", pred_digit)
    print("prob_d:", prob_d.ravel())
    print("updated W_FeatL shape:", net.W_FeatL.shape)


if __name__ == "__main__":
    main()
