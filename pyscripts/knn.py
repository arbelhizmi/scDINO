import torch
import argparse
from argparse import Namespace
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import f1_score, accuracy_score

import utils

label_to_idx = {
    '"Lymphatic" endothelial cells': 0,
    'B-cell': 1,
    'Cholangiocytes': 2,
    'Dendritic cells': 3,
    'Hepatocyte': 4,
    'Kupffer': 5,
    'Liver sinusoidal endothelial cell': 6,
    'Stellate cell': 7,
    'T-cell': 8,
    'Vascular endothelial cell': 9
}


def save_and_print_metrics(metrics, output_filename: str, k: int, temperature: float) -> None:
    """
    Formats the metrics, writes them to a file, and prints them.

    Args:
        metrics (Tuple): Tuple containing (top1, top2, overall_accuracy, overall_f1, per_class_accuracy, per_class_f1).
        output_filename (str): Path to the output text file.
        k (int): Number of nearest neighbors.
        temperature (float): Temperature scaling factor.
    """
    top1, top2, overall_acc, overall_f1, per_acc, per_f1 = metrics

    # Format the metrics into a readable string
    formatted_text = format_metrics(
        top1=top1,
        top2=top2,
        overall_accuracy=overall_acc,
        overall_f1=overall_f1,
        per_class_accuracy=per_acc,
        per_class_f1=per_f1,
        k=k,
        temperature=temperature
    )

    # Write the formatted string to the output file
    try:
        with open(output_filename, 'a') as file:
            file.write(formatted_text)
    except Exception as e:
        print(f"Error writing to file {output_filename}: {e}")

    # Print the formatted metrics to the console
    print(formatted_text)


def format_metrics(
    top1: float,
    top2: float,
    overall_accuracy: float,
    overall_f1: float,
    per_class_accuracy: Dict[int, float],
    per_class_f1: Dict[int, float],
    k: int,
    temperature: float
) -> str:
    lines = []
    lines.append(f"KNN Classification Metrics (k={k}, temperature={temperature})")
    lines.append("=" * 60)
    lines.append(f"Top-1 Accuracy: {top1:.2f}%")
    lines.append(f"Top-2 Accuracy: {top2:.2f}%")
    lines.append(f"Overall Accuracy: {overall_accuracy:.2f}%")
    lines.append(f"Overall F1-Score: {overall_f1:.2f}%")
    lines.append("\nPer-Class Metrics:")
    lines.append("-" * 60)
    lines.append(f"{'Class':<40}{'Accuracy (%)':<15}{'F1-Score (%)':<15}")
    lines.append("-" * 60)

    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Sort the class indices for consistent ordering
    for class_idx in sorted(per_class_accuracy.keys()):
        label = idx_to_label.get(class_idx, f"Class {class_idx}")
        acc = per_class_accuracy[class_idx]
        f1 = per_class_f1[class_idx]
        lines.append(f"{label:<40}{acc:<15.2f}{f1:<15.2f}")

    lines.append("\n")  # Add space after each metric block
    return "\n".join(lines)

    # lines = []
    # lines.append(f"KNN Classification Metrics (k={k}, temperature={temperature})")
    # lines.append("=" * 50)
    # lines.append(f"Top-1 Accuracy: {top1:.2f}%")
    # lines.append(f"Top-2 Accuracy: {top2:.2f}%")
    # lines.append(f"Overall Accuracy: {overall_accuracy:.2f}%")
    # lines.append(f"Overall F1-Score: {overall_f1:.2f}%")
    # lines.append("\nPer-Class Metrics:")
    # lines.append("-" * 50)
    # lines.append(f"{'Class':<10}{'Accuracy (%)':<15}{'F1-Score (%)':<15}")
    # lines.append("-" * 50)
    # for class_idx in sorted(per_class_accuracy.keys()):
    #     acc = per_class_accuracy[class_idx]
    #     f1 = per_class_f1[class_idx]
    #     lines.append(f"{class_idx:<10}{acc:<15.2f}{f1:<15.2f}")
    # lines.append("\n")  # Add space after each metric block
    # return "\n".join(lines)


def load_features_and_labels(path_to_features_and_labels: str):
    features_tensor = torch.load(f'{path_to_features_and_labels}/features.pth')
    labels_tensor = torch.load(f'{path_to_features_and_labels}/labels.pth')

    return features_tensor, labels_tensor


def split_features_and_labels_to_train_test(features: torch.Tensor, labels: torch.Tensor, train_ratio: float):
    # features: [N, D] where N is the number of samples and D is the feature dimension
    # labels: [N] where N is the number of samples
    number_of_samples = features.size(0)  # Number of samples

    # Calculate split sizes
    train_size = int(train_ratio * number_of_samples)
    test_size = number_of_samples - train_size

    # Randomly split the dataset into training and testing
    train_indices, test_indices = torch.utils.data.random_split(
        range(number_of_samples), [train_size, test_size]
    )

    # Use indices to split features and labels
    train_features = features[train_indices.indices]
    test_features = features[test_indices.indices]
    train_labels = labels[train_indices.indices]
    test_labels = labels[test_indices.indices]

    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def calculate_knn_classification(
    arguments: Namespace, number_of_nearest_neighbours: int, temperature: float, features: torch.Tensor,
    labels: torch.Tensor, number_of_classes: int,
):
    # Split features to train and test according to ratio
    train_features, test_features, train_labels, test_labels = split_features_and_labels_to_train_test(
        features, labels, arguments.train_ratio,
    )

    top1, top2, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(number_of_nearest_neighbours, number_of_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(number_of_nearest_neighbours, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * number_of_nearest_neighbours, number_of_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temperature).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, number_of_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top2 = top2 + correct.narrow(1, 0, min(2, number_of_nearest_neighbours)).sum().item()  # dependent on number of classes
        total += targets.size(0)

    top1 = top1 * 100.0 / total
    top2 = top2 * 100.0 / total
    return top1, top2


@torch.no_grad()
def calculate_knn_classification_with_classes(
    arguments: Namespace, number_of_nearest_neighbours: int, temperature: float, features: torch.Tensor,
    labels: torch.Tensor, number_of_classes: int,
):
    train_features, test_features, train_labels, test_labels = split_features_and_labels_to_train_test(
        features, labels, arguments.train_ratio,
    )

    top1, top2, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(number_of_nearest_neighbours, number_of_classes).to(train_features.device)

    all_predictions = []
    all_targets = []

    for idx in range(0, num_test_images, imgs_per_chunk):
        # Get the features for test images
        features_batch = test_features[
                         idx: min((idx + imgs_per_chunk), num_test_images), :
                         ]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # Calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features_batch, train_features)
        distances, indices = similarity.topk(number_of_nearest_neighbours, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * number_of_nearest_neighbours, number_of_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(temperature).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, number_of_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # Top-1 and Top-2 predictions
        top1_preds = predictions[:, 0]
        top2_preds = predictions[:, :2] if number_of_nearest_neighbours >= 2 else predictions[:, :1]

        # Collect all top1 predictions and targets for overall metrics
        all_predictions.extend(top1_preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

        # Calculate top1 and top2 correct
        correct_top1 = top1_preds.eq(targets.data.view(-1)).sum().item()
        top1 += correct_top1

        if number_of_nearest_neighbours >= 2:
            correct_top2 = top2_preds.eq(targets.data.view(-1, 1)).any(dim=1).sum().item()
        else:
            correct_top2 = correct_top1  # If k < 2, top2 is same as top1
        top2 += correct_top2

        total += targets.size(0)

    # Calculate percentages
    top1_percentage = top1 * 100.0 / total
    top2_percentage = top2 * 100.0 / total

    # Compute overall accuracy and F1-score using sklearn
    overall_accuracy = accuracy_score(all_targets, all_predictions) * 100.0
    overall_f1 = f1_score(all_targets, all_predictions, average='weighted') * 100.0

    # Compute per-class accuracy
    per_class_accuracy = {}
    per_class_f1 = {}
    for class_idx in range(number_of_classes):
        # True Positives for the class
        tp = sum((pred == class_idx) and (true == class_idx) for pred, true in zip(all_predictions, all_targets))
        # Total actual instances of the class
        total_true = sum(true == class_idx for true in all_targets)
        if total_true > 0:
            per_class_accuracy[class_idx] = (tp / total_true) * 100.0
        else:
            per_class_accuracy[class_idx] = 0.0

    # Compute per-class F1-score using sklearn
    per_class_f1_scores = f1_score(all_targets, all_predictions, average=None)
    for class_idx in range(number_of_classes):
        per_class_f1[class_idx] = per_class_f1_scores[class_idx] * 100.0

    return top1_percentage, top2_percentage, overall_accuracy, overall_f1, per_class_accuracy, per_class_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('kNN')
    # computation settings
    parser.add_argument(
        "--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")'
    )
    parser.add_argument('--batch_size', default=1024, type=int, help='Per-GPU batch-size')

    # Dataset arguments
    parser.add_argument(
        '--liver_slices_directory_path', type=str, default='/home/labs/nyosef/jonasm/vizgen',
        help='path to the liver slices directory (e.g. like ../jonasm/vizgen directory)',
    )
    parser.add_argument(
        '--liver_slice_names', default=['Liver1Slice1', ], type=list, nargs='+',
        help='list of liver slice names as list of strs (like Liver1Slice1 etc.)',
    )
    parser.add_argument(
        '--transformed_coordinates_csv_path', type=str, default='/home/labs/nyosef/Collaboration/Arbel/vizgen_liver',
        help='path to the local liver slices directory (e.g. like ../Collaboration/Arbel/vizgen_liver directory)',
    )
    parser.add_argument(
        '--unified', type=utils.bool_flag, default=True,
        help="flag to indicate whether to use 'Hepatocyte' as a single cell type (all cell types which are versions of "
             "'Hepatocyte' are unified to a single cell type (for example, 'Hepatocyte, peri-portal' would just"
             "be 'Hepatocyte'"
    )
    parser.add_argument(
        '--selected_channels', default=[0, 1, 2, 3, 4], type=list,
        help="""list of channel indexes of the .tiff images which should be used to create the tensors."""
    )
    parser.add_argument(
        '--norm_per_channel', help="""2x tuple of mean and std per channel typically values between 0 and 1""",
        default=[(0.2981, 0.5047, 0.1918, 0.2809, 0.3800), (0.2232, 0.1659, 0.2403, 0.1938, 0.1956)], type=list,
    )
    parser.add_argument('--output_dir', default='./experiments', type=str, help='path to save logs and checkpoints.')
    parser.add_argument(
        "--train_ratio", default=0.8, type=float, help="when using scDINO full pipeline"
    )

    args = parser.parse_args()

    # Define which top-k to measure (for example, if [1, 2] it means find top-1, top-2)
    top_k = [1, 2]

    k_nearest_neighbors = [5, 10, 20, 50, 100]
    temperature = 0.1
    results = {}

    # Loading features and labels path
    if args.unified:
        features_directory_name = f'features_and_labels/unified/features_and_labels_{args.checkpoint_key}'
        knn_directory_name = f'knn/unified/knn_results_{args.checkpoint_key}'
    else:
        features_directory_name = f'features_and_labels/not_unified/features_and_labels_{args.checkpoint_key}'
        knn_directory_name = f'knn/not_unified/knn_results_{args.checkpoint_key}'

    features_and_labels_path = Path(
        f'{args.transformed_coordinates_csv_path}/{args.liver_slice_names[0]}/{features_directory_name}'
    )
    features, labels = load_features_and_labels(str(features_and_labels_path))
    number_of_classes = len(labels.unique().tolist())

    local_rank = torch.cuda.current_device()
    device = torch.device(f'cuda:{local_rank}')
    features = features.to(device)
    labels = labels.to(device)

    # Define results saving path and output filename
    save_knn_results_path = Path(
        f'{args.transformed_coordinates_csv_path}/{args.liver_slice_names[0]}/{knn_directory_name}'
    )
    Path(save_knn_results_path).mkdir(parents=True, exist_ok=True)

    output_filename = save_knn_results_path / f'knn_top-{top_k[0]}_and_top-{top_k[1]}_accuracy_f1-score.txt'
    # if output_filename.exists():
    #     output_filename.unlink()
    # if output_filename.exists():
    #     print(f'Results for top-{top_k[0]} and top-{top_k[1]} already exists in:\n{output_filename}')
    #     sys.exit(0)

    # use_metrics = False
    use_metrics = True
    number_of_splits = 3

    if use_metrics:
        
        for split_index in range(number_of_splits):

            output_filename = save_knn_results_path / f'knn_accuracy_f1-score_split_{split_index}.txt'
            if output_filename.exists():
                output_filename.unlink()
        
            for k in k_nearest_neighbors:
                metrics = calculate_knn_classification_with_classes(
                    args, k, temperature, features, labels, number_of_classes,
                )
                top1, top2, overall_acc, overall_f1, per_acc, per_f1 = metrics

                # Save and print the metrics
                save_and_print_metrics(
                    metrics=metrics,
                    output_filename=str(output_filename),
                    k=k,
                    temperature=temperature
                )

                # print(f"{k}-NN classifier result: Top{top_k[0]}: {top1}, Top{top_k[1]}: {top2}")

                # results[k] = (top1, top2)
                # with open(output_filename, 'a') as file:
                #     file.write(f'{k}-NN classifier result: Top{top_k[0]}: {top1}, Top{top_k[1]}: {top2}\n')

    else:
        for k in k_nearest_neighbors:
            top_1_accuracy, top_2_accuracy = calculate_knn_classification(
                args, k, temperature, features, labels, number_of_classes,
            )
            text_message = f'{k}-NN classifier result: Top{top_k[0]}: {top_1_accuracy}, Top{top_k[1]}: {top_2_accuracy}'
            print(text_message)

            results[k] = (top_1_accuracy, top_2_accuracy)
            with open(output_filename, 'a') as file:
                file.write(f'{text_message}\n')
