
from datasets.base_dataset import BaseImageDataset


class NIH14Dataset(BaseImageDataset):
    """
    Dataset class for NIH14 images

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
        super().__init__(
            transform=transform,
            image_annotation_type=image_annotation_type,
            text_annotation_suffix=text_annotation_suffix,
            color=color,
            line_width=line_width,
        )

    @property
    def class_labels(self):
        return sorted(self.df["Finding Label"].unique().tolist())

    def __getitem__(self, idx):
        """
        Get an image and its label from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple of (image_id, image, label)
        """
        row = self.df.iloc[idx]
        image_id = row["Image Index"]
        label = row["Finding Label"]

        # Get the image path
        # FIXME: this is not ideal but the chunk strucutre cannot be guessed
        # and the images are stored in subdirectories like images_001, images_002, etc.
        # so we need to iterate through them to find the correct image
        image_path = image_id

        item_info = {
            "x": float(row["Bbox [x"]),
            "y": float(row["y"]),
            "w": float(row["w"]),
            "h": float(row["h]"]),
        }
        image = self.annotate_image(
            image_path=image_path,
            item_info=item_info,
            label=label,
            annotation_type=self.annotation_type,
            color=self.color,
            line_width=self.line_width,
        )

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        return image_id, image, label

    def get_box_coordinates(self, image, image_path, item_info, label):
        """
        Get bounding box coordinates for NIH14 findings.

        Args:
            image: PIL Image with target size
            image_path: Path to the image file
            item_info: Dictionary with annotation information
            label: Label for which to find bounding box

        Returns:
            Tuple (x1, y1, x2, y2) of box coordinates
        """
        # load original image size
        orig_image = self.load_image(image_path, resize=None)
        w, h = orig_image.size
        t_h, t_w = image.size
        w_ratio, h_ratio = w / t_w, h / t_h

        bx, by, bw, bh = item_info["x"], item_info["y"], item_info["w"], item_info["h"]

        # scale box coordinates to target_image size
        x1 = int(bx / w_ratio)
        y1 = int(by / h_ratio)
        x2 = int((bx + bw) / w_ratio)
        y2 = int((by + bh) / h_ratio)

        return (x1, y1, x2, y2)

    def load_image(self, image_path, resize=256):
        # image_path is actually the image ID, e.g. "00000001_000.png"
        # iterate through all directories image_dir / f"images_{0XX}" / f"{image_id}.png" and find the image
        _image_path = None
        for i in range(1, 13):
            _image_path = self.images_dir / f"images_{i:03d}" / "images" / image_path
            if _image_path.exists():
                break

        return super().load_image(_image_path, resize=resize)
