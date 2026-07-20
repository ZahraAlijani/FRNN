from __future__ import annotations

from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np

from two_drnn import (
    adam_update_2drnn,
    backward_2drnn,
    create_net_2drnn,
    forward_2drnn,
    margin_loss,
    matlab_label_to_digit,
    one_hot_label,
)


def _load_mat_array(data: dict, *candidate_keys: str) -> np.ndarray:
    for key in candidate_keys:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(f"None of the keys were found: {candidate_keys}")


def train_for_f(f: int, total_epoch: int = 100) -> dict[str, float | int | np.ndarray]:
    p = 7
    h1 = 10
    h2 = 1
    m_grid = 4
    learning_rate = 0.001
    batch_size = 1000
    total_training_samples = 60000

    net = create_net_2drnn(p, f, h1, h2, m_grid)

    train_data = loadmat("MNIST2.mat")
    x_train = _load_mat_array(train_data, "MNSIT_Train", "X_train")
    y_train_raw = matlab_label_to_digit(_load_mat_array(train_data, "MNSIT_TrainL", "Y_train", "Y_train_raw"))

    onehot_label_matrix = np.zeros((10, total_training_samples))
    for sample_idx, digit_value in enumerate(y_train_raw[:total_training_samples]):
        onehot_label_matrix[digit_value, sample_idx] = 1.0

    loss_log = np.zeros(total_epoch)
    accuracy_log = np.zeros(total_epoch)

    start_time = __import__("time").time()
    for epoch_count in range(total_epoch):
        shuffle_index = np.random.permutation(total_training_samples)
        shuffled_image_set = x_train[:, :, shuffle_index]
        shuffled_label_set = onehot_label_matrix[:, shuffle_index]

        cumulative_loss = 0.0
        correct_prediction_counter = 0
        total_batch_number = total_training_samples // batch_size

        for batch_idx in range(total_batch_number):
            start_pos = batch_idx * batch_size
            end_pos = (batch_idx + 1) * batch_size

            batch_images = shuffled_image_set[:, :, start_pos:end_pos]
            batch_labels = shuffled_label_set[:, start_pos:end_pos]

            for local_sample in range(batch_size):
                current_image = batch_images[:, :, local_sample]
                current_onehot_label = batch_labels[:, local_sample].reshape(10, 1)

                prob_d, pred_digit, h_final, patch_store, hidden_history = forward_2drnn(net, current_image)

                true_digit_label = int(np.argmax(current_onehot_label))
                if pred_digit == true_digit_label:
                    correct_prediction_counter += 1

                cumulative_loss += margin_loss(net, prob_d, current_onehot_label)
                gradient_output = backward_2drnn(
                    net,
                    current_image,
                    current_onehot_label,
                    prob_d,
                    h_final,
                    patch_store,
                    hidden_history,
                )
                net = adam_update_2drnn(net, gradient_output, learning_rate)

        average_epoch_loss = cumulative_loss / total_training_samples
        epoch_accuracy = correct_prediction_counter / total_training_samples
        loss_log[epoch_count] = average_epoch_loss
        accuracy_log[epoch_count] = epoch_accuracy
        print(f"f={f} | Epoch {epoch_count + 1} || Average Loss: {average_epoch_loss:.4f} || Accuracy: {epoch_accuracy:.4f}")

    run_time = (__import__("time").time() - start_time) / total_epoch

    test_data = loadmat("MNIST2Test.mat")
    x_test = _load_mat_array(test_data, "X_Test", "X_test")
    y_test_raw = matlab_label_to_digit(_load_mat_array(test_data, "Y_Test", "Y_test"))

    test_correct = 0
    total_test_loss = 0.0
    for sample_idx in range(10000):
        img = x_test[:, :, sample_idx]
        true_digit = int(y_test_raw[sample_idx])
        onehot_vec = one_hot_label(true_digit)

        prob_d, _, _, _, _ = forward_2drnn(net, img)
        total_test_loss += margin_loss(net, prob_d, onehot_vec)

        predicted_digit = int(np.argmax(prob_d))
        if predicted_digit == true_digit:
            test_correct += 1

    avg_test_loss = total_test_loss / 10000
    test_accuracy = test_correct / 10000
    param_n = (f * p) + (p * f) + (f * f) + (h1 * f) + (f * h2) + (h1 * h1) + (h2 * h2) + (h1 * h2) + (10 * h1) + (h2 * 1) + (10 * 1)

    net_name = f"RNN2D_H({h1}t{h2})m{m_grid}f{f}.mat"
    savemat(
        net_name,
        {
            "net": net,
            "loss_log": loss_log,
            "accuracy_log": accuracy_log,
            "epoch_count": total_epoch,
        },
    )

    print("Final Reports for Table")
    print(
        f"f={f} | Num. N: {param_n} | Loss (Tr) {average_epoch_loss:.4f} | Train acc {epoch_accuracy * 100:.4f} | Test acc {test_accuracy * 100:.4f} | T={run_time:.4f} (s)"
    )
    print("Copy for Table:\n")
    print(f"{f} &  {param_n} &  {average_epoch_loss:.4f} &  {epoch_accuracy * 100:.4f} & {test_accuracy * 100:.4f} & {run_time:.4f}  \n")

    return {
        "f": f,
        "param_n": param_n,
        "train_loss": average_epoch_loss,
        "train_accuracy": epoch_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": avg_test_loss,
        "run_time": run_time,
        "loss_log": loss_log,
        "accuracy_log": accuracy_log,
    }


def main() -> None:
    results = train_for_f(2, total_epoch=100)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(results["loss_log"], linewidth=1.2)
    plt.title("Epoch-wise Average Training Loss")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss Value")

    plt.subplot(2, 1, 2)
    plt.plot(results["accuracy_log"], color="red", linewidth=1.2)
    plt.title("Classification Accuracy over Training Epochs")
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy Ratio")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
