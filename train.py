import argparse
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime, timezone

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

logger = logging.getLogger(__name__)


def load_data(data_dir: str) -> dict:
    """
    Load datasets from specified directory and return data loaders.

    :param data_dir: Path to the directory containing training data.
    :return: A dictionary containing data loaders for train, valid, and test sets,
             along with class labels.
    """
    logger.info("Loading data.")
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    modified_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_dataset = datasets.ImageFolder(data_dir, transform=modified_transforms)
    train_dataset = datasets.ImageFolder(train_dir, transform=modified_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=modified_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=modified_transforms)

    batch_size = 32

    data_loaders = torch.utils.data.DataLoader(
        image_dataset, batch_size=batch_size, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)

    return {
        "data_loaders": data_loaders,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "labels": cat_to_name,
        "class_to_idx": train_dataset.class_to_idx,
    }


def build_model(
    arch: str, hidden_units: int, learning_rate: float, labels: list
) -> tuple:
    """
    Build and configure a neural network model based on specified architecture.

    :param arch: Model architecture ('vgg16' or 'resnet').
    :param hidden_units: Number of units in hidden layer of the classifier.
    :param learning_rate: Learning rate for the optimizer.
    :param labels: List of class labels for the dataset.
    :return: A tuple containing the constructed model, criterion, optimizer, and input size of the classifier.
    """
    logger.info("Building model.")

    # Load a pre-trained model based on architecture
    if arch.startswith("vgg"):
        model = models.__dict__[arch](pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch.startswith("resnet"):
        model = models.__dict__[arch](pretrained=True)
        classifier_input_size = model.fc.in_features
    else:
        raise ValueError(
            f"Unsupported architecture '{arch}'. Please choose 'vgg' or 'resnet'."
        )

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(labels)

    # Define classifier based on architecture
    if arch.startswith("vgg"):
        classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(classifier_input_size, hidden_units)),
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(hidden_units, num_classes)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )
        model.classifier = classifier
    elif arch.startswith("resnet"):
        classifier = nn.Linear(classifier_input_size, num_classes)
        model.fc = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Succesfully built model.")
    return model, criterion, optimizer, classifier_input_size


def test_network(model, criterion, test_loader, device):
    """
    Function to evaluate the model on test/validation data.

    :param model: The neural network model to be evaluated.
    :param criterion: The loss function used for evaluation.
    :param test_loader: DataLoader providing the test/validation data.
    :param device: Device to run the evaluation on (e.g., "cpu" or "cuda").
    :return: Tuple of (average test loss, accuracy)
    """
    logger.info("Testing model.")
    model.eval()
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(outputs)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    # Return average test loss and accuracy
    return test_loss / len(test_loader), accuracy / len(test_loader)


def train_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    device: str,
) -> None:
    """
    This function trains an image classification model and validates it on the validation set during each epoch.

    :param model: The neural network model to be trained.
    :param criterion: The loss function used for training.
    :param optimizer: The optimizer used for updating the model parameters.
    :param epochs: Number of epochs (iterations over the entire dataset) for training.
    :param train_loader: DataLoader providing the training data.
    :param valid_loader: DataLoader providing the validation data.
    :param device: Device to run the training on (e.g., "cpu" or "cuda").
    """
    logger.info("Training model.")
    model.to(device)
    model.train()

    datetime_now_utc = datetime.now(tz=timezone.utc)
    first_loop = True

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0.0

        # Training loop
        for _, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate average training time per batch
            if first_loop:
                first_loop = False
                batch_time = datetime.now(tz=timezone.utc) - datetime_now_utc
                batches_per_epoch = len(train_loader)
                estimated_time_per_epoch = batch_time * batches_per_epoch
                total_training_time = estimated_time_per_epoch * epochs
                eta_utc = datetime_now_utc + total_training_time

                # Log estimated time of completion
                logger.info(
                    f"Estimated time of completion (UTC): {eta_utc:%Y-%m-%d %H:%M:%S}"
                )

        # Calculate average training loss
        train_loss /= len(train_loader)

        # Validation loop
        if epoch % 1 == 0:
            valid_loss, accuracy = test_network(model, criterion, valid_loader, device)

            # Logging
            logger.info(
                f"Epoch {epoch}/{epochs}, "
                f"Train Loss: {train_loss:.6f}, "
                f"Valid Loss: {valid_loss:.6f}, "
                f"Accuracy: {accuracy:.4f}"
            )

    logger.info("Training complete!")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    classifier_input_size: int,
    labels: list,
    arch: str,
    class_to_idx: dict,
    save_dir: str,
) -> None:
    """
    Save the trained model checkpoint.

    :param model: The trained neural network model.
    :param train_loader: DataLoader providing the training data.
    :param optimizer: The optimizer used for training.
    :param classifier_input_size: Input size of the classifier.
    :param labels: List of class labels.
    :param arch: Model architecture used for training.
    :param save_dir: Directory path where checkpoints will be saved.

    :return: None
    """
    logger.info("Saving model.")
    model.class_to_idx = class_to_idx

    checkpoint = {
        "class_to_idx": model.class_to_idx,
        "classifier": model.classifier,
        "features": model.features if hasattr(model, "features") else None,
        "input_size": classifier_input_size,
        "optimizer": optimizer.state_dict(),
        "output_size": len(labels),
        "state_dict": model.state_dict(),
        "transfer_model": arch,
    }

    # Construct the checkpoint file path using save_dir
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    torch.save(checkpoint, checkpoint_path)


def main() -> None:
    """
    Main function for training a neural network on a dataset.

    :return: None
    """
    # Define argument parser
    parser = argparse.ArgumentParser(description="Train a new network on a dataset")
    parser.add_argument(
        "data_directory",
        metavar="data_dir",
        type=str,
        help="Path to the directory containing training data",
    )
    parser.add_argument(
        "--save_dir",
        dest="save_directory",
        metavar="save_dir",
        type=str,
        default=".",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--arch",
        dest="arch",
        metavar="arch",
        type=str,
        default="vgg16",
        help="Model architecture to use for training (vgg16 or resnet)",
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        metavar="learning_rate",
        type=float,
        default=0.002,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--hidden_units",
        dest="hidden_units",
        metavar="hidden_units",
        type=int,
        default=4096,
        help="Number of units in hidden layer",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        metavar="epochs",
        type=int,
        default=5,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        help="Use GPU for training if available",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    try:
        data = load_data(args.data_directory)
        train_loader = data["train_loader"]
        valid_loader = data["valid_loader"]
        labels = data["labels"]
        class_to_idx = data["class_to_idx"]
    except FileNotFoundError:
        logger.error(
            f"Directory '{args.data_directory}' not found or does not contain required data."
        )
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Build the model
    try:
        model, criterion, optimizer, input_size = build_model(
            args.arch, args.hidden_units, args.learning_rate, labels
        )
    except ValueError as ve:
        logger.error(str(ve))
        return

    # Train the model
    try:
        train_model(
            model, criterion, optimizer, args.epochs, train_loader, valid_loader, device
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

    # Save the trained model checkpoint
    try:
        save_checkpoint(
            model,
            optimizer,
            input_size,
            labels,
            args.arch,
            class_to_idx,
            args.save_directory,
        )
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return


if __name__ == "__main__":
    main()
