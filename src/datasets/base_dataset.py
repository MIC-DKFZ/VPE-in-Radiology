import json
import math
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class BaseImageDataset(Dataset):
    """
    Base class for image datasets with annotation capabilities.

    Args:
        transform: Optional transform to apply to images
        image_annotation_type: Type of annotation ('bbox', 'circle', 'arrow', 'crop', None)
        color: Color for annotations (if applicable)
        line_width: Line width for annotations (if applicable)
        text_annotation_suffix: Suffix to append to text annotations
    """

    def __init__(
        self,
        transform=None,
        image_annotation_type=None,
        color=None,
        line_width=None,
        text_annotation_suffix="",
    ):
        config = self.load_class_config()
        metadata_file = Path(config["metadata_file"])
        images_dir = Path(config["images_dir"])

        assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist"
        assert images_dir.exists(), f"Images directory {images_dir} does not exist"

        self.df = self.load_metadata(metadata_file)
        self.images_dir = images_dir
        self.transform = transform
        self.annotation_type = image_annotation_type
        self.color = color
        self.line_width = line_width
        self.text_annotation_suffix = text_annotation_suffix

    def load_class_config(self):
        config_path = Path(os.environ.get("CONFIG", "default_config.json"))
        assert config_path.exists(), f"Config file {config_path} does not exist"
        with config_path.open("r") as f:
            config = json.load(f)
        class_config = config[self.__class__.__name__]
        return class_config

    @classmethod
    def load_metadata(cls, metadata_file):
        """
        Load metadata from a metadata file.

        Args:
            metadata_file: Path to the metadata file containing metadata

        Returns:
            DataFrame with metadata
        """
        return pd.read_csv(metadata_file)

    @property
    def class_labels(self):
        """
        Get unique class labels from the dataset.

        Returns:
            List of unique class labels
        """
        raise NotImplementedError(
            "Subclasses must implement class_labels property to return unique labels"
        )

    def generate_text_prompt(self):
        """
        Generate text prompts for the dataset based on class labels.

        Returns:
            List of text prompts for each class label
        """
        prompts = [
            f"A chest x-ray with signs of {c}{self.text_annotation_suffix}."
            for c in self.class_labels
        ]
        print(f"Generated {len(prompts)} text prompts: {prompts}")
        return prompts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Default implementation that should be overridden by subclasses
        to handle dataset-specific image loading and annotation.
        """
        raise NotImplementedError("Subclasses must implement __getitem__")

    def get_box_coordinates(self, image, image_path, item_info, label):
        """
        Get bounding box coordinates for the given image and label.
        Should be implemented by subclasses based on dataset-specific annotation format.

        Args:
            image: PIL Image with target size
            image_path: Path to the image file
            item_info: Dictionary with annotation information
            label: Label for which to find bounding box

        Returns:
            Tuple (x1, y1, x2, y2) of box coordinates
        """
        raise NotImplementedError("Subclasses must implement get_box_coordinates")

    def annotate_image(
        self,
        image_path,
        item_info,
        label,
        annotation_type=None,
        color=None,
        line_width=None,
    ):
        """
        Annotate image based on annotation_type.

        Args:
            image: PIL Image to annotate
            item_info: Dictionary with additional information about the image
            label: Label for which to find annotations
            annotation_type: Type of annotation ('bbox', 'circle', 'arrow', 'crop')
            color: Color for annotations
            line_width: Line width for annotations

        Returns:
            Annotated PIL Image
        """
        # Use instance attributes if specific parameters not provided
        annotation_type = annotation_type or self.annotation_type
        color = color or self.color
        line_width = line_width or self.line_width or 1

        image = self.load_image(image_path)
        if annotation_type is None:
            return image

        if annotation_type == "crop":
            # Use full resolution image for cropping
            image = self.load_image(image_path, resize=None)
            x1, y1, x2, y2 = self.get_box_coordinates(
                image, image_path, item_info, label
            )
            w, h = image.size
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            crop_size = max(0.5 * (x2 - x1), 0.5 * (y2 - y1), 0.2 * w, 0.2 * h)
            x1 = max(0, int(center_x - crop_size))
            x2 = min(w, int(center_x + crop_size))
            y1 = max(0, int(center_y - crop_size))
            y2 = min(h, int(center_y + crop_size))
            image = image.crop((x1, y1, x2, y2))
            return image

        elif annotation_type in ["bbox", "circle", "arrow"]:
            try:
                box_coordinates = self.get_box_coordinates(
                    image, image_path, item_info, label
                )
                x1, y1, x2, y2 = box_coordinates
                draw = ImageDraw.Draw(image)

                if annotation_type == "bbox":
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                elif annotation_type == "circle":
                    draw.ellipse([x1, y1, x2, y2], outline=color, width=line_width)
                elif annotation_type == "arrow":
                    # Compute centers and desired arrow length
                    w, h = image.size
                    cx_img, cy_img = w / 2, h / 2
                    cx_box, cy_box = (x1 + x2) / 2, (y1 + y2) / 2
                    desired_arrow_length = 0.25 * min(w, h)
                    dx, dy = cx_box - cx_img, cy_box - cy_img
                    dist = math.sqrt(dx * dx + dy * dy)
                    arrow_end_x, arrow_end_y = cx_box, cy_box

                    if dist == 0:
                        arrow_start_x, arrow_start_y = (
                            cx_box,
                            cy_box + 5,
                        )  # minimal offset
                    else:
                        ux, uy = dx / dist, dy / dist
                        arrow_start_x = arrow_end_x - ux * desired_arrow_length
                        arrow_start_y = arrow_end_y - uy * desired_arrow_length

                    draw.line(
                        [(arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y)],
                        fill=color,
                        width=line_width,
                    )

                    # Draw arrowhead
                    arrow_head_size = 10
                    line_dx = arrow_end_x - arrow_start_x
                    line_dy = arrow_end_y - arrow_start_y
                    line_len = math.sqrt(line_dx**2 + line_dy**2)
                    if line_len > 0:
                        ux_line, uy_line = line_dx / line_len, line_dy / line_len
                        perp_x, perp_y = -uy_line, ux_line
                        arrow_head_points = [
                            (arrow_end_x, arrow_end_y),
                            (
                                arrow_end_x
                                - ux_line * arrow_head_size
                                + perp_x * (arrow_head_size / 2),
                                arrow_end_y
                                - uy_line * arrow_head_size
                                + perp_y * (arrow_head_size / 2),
                            ),
                            (
                                arrow_end_x
                                - ux_line * arrow_head_size
                                - perp_x * (arrow_head_size / 2),
                                arrow_end_y
                                - uy_line * arrow_head_size
                                - perp_y * (arrow_head_size / 2),
                            ),
                        ]
                        draw.polygon(arrow_head_points, fill=color)
            except Exception as e:
                print(f"Error annotating image: {e}")

            return image
        else:
            raise ValueError(f"Invalid annotation type: {annotation_type}")

    def load_image(self, image_path, resize=256):
        """
        Loads an image, normalizes it to [0, 255], converts to RGB,
        and optionally resizes it.
        """
        image = np.array(Image.open(image_path))
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        if resize:
            image = image.resize((resize, resize))
        return image
