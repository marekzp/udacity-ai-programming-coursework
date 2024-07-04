import argparse
import json
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


def load_checkpoint(filepath: str, device: str) -> torch.nn.Module:
    """
    Loads a checkpoint and rebuilds the model.

    :param filepath: Path to the checkpoint file.
    :param device: Device to load the model onto (e.g., "cpu" or "cuda").
    :return: Rebuilt model loaded from the checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=torch.device(device))

    # Determine the model architecture based on the checkpoint
    arch = checkpoint["transfer_model"]
    if arch.startswith("vgg"):
        model = models.__dict__[arch](pretrained=True)
        model.classifier = checkpoint["classifier"]
        model.features = checkpoint["features"]
    elif arch.startswith("resnet"):
        model = models.__dict__[arch](pretrained=True)
        model.fc = checkpoint["classifier"]
    else:
        raise ValueError(f"Unsupported architecture '{arch}' found in the checkpoint.")

    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image_path: str) -> torch.Tensor:
    """
    Processes a PIL image for use in a PyTorch model.

    :param image_path: Path to the image file.
    :return: Processed image tensor.
    """
    im = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor_image = transform(im).float()

    return tensor_image


def imshow(image: torch.Tensor, ax=None):
    """
    Displays a PyTorch tensor as an image.

    :param image: Image tensor to display.
    :param ax: Optional axes to plot on.
    :param title: Optional title for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Undo normalization and convert to numpy
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.axis("off")


def predict(
    image_path: str, model: torch.nn.Module, device: str, topk: int = 5
) -> tuple:
    """
    Predicts the top-K classes of an image using a trained deep learning model.

    :param image_path: Path to the image file.
    :param model: Trained PyTorch model for prediction.
    :param device: Device to run prediction on (e.g., "cpu" or "cuda").
    :param topk: Number of top classes to predict.
    :return: Tuple containing lists of top-K probabilities and classes.
    """
    model.eval()

    # Process the image
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities, classes = torch.topk(torch.softmax(output, dim=1), topk)

    probabilities, classes = torch.exp(output).topk(topk)

    return probabilities[0].tolist(), classes[0].add(1).tolist()


def main():
    """
    Main function for predicting image classes using a trained neural network model.
    """
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image using a trained neural network"
    )
    parser.add_argument(
        "image_path",
        metavar="image_path",
        type=str,
        help="Path to the image file for prediction",
    )
    parser.add_argument(
        "checkpoint", metavar="checkpoint", type=str, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--top_k",
        dest="top_k",
        metavar="top_k",
        type=int,
        default=5,
        help="Number of top-K classes to predict",
    )
    parser.add_argument(
        "--category_names",
        dest="category_names",
        metavar="category_names",
        type=str,
        default="cat_to_name.json",
        help="Path to a JSON file mapping categories to real names",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_true",
        help="Use GPU for inference if available",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load the category names
    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)

    # Load the checkpoint
    try:
        model = load_checkpoint(args.checkpoint, device)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return

    # Predict the class probabilities
    try:
        probabilities, classes = predict(args.image_path, model, device, args.top_k)
    except Exception as e:
        logger.error(f"Error predicting image: {e}")
        return

    # Map classes to category names
    class_names = [cat_to_name[str(cls)] for cls in classes]

    # Display the image and prediction results
    try:
        Image.open(args.image_path)
        _, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=1, nrows=2)
        imshow(
            process_image(args.image_path),
            ax=ax1,
        )
        ax2.barh(class_names, probabilities, color="red")
        ax2.set_xlabel("Probability")
        ax2.set_title(f"Top {args.top_k} Predictions")
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error displaying image and predictions: {e}")
        return


if __name__ == "__main__":
    main()
