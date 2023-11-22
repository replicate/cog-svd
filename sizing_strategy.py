from PIL import Image

MAX_DIMENSION = 1024


class SizingStrategy:
    def __init__(self):
        pass

    def get_dimensions(self, image):
        return image.size

    def resize_image(self, image, width, height):
        return image.resize((width, height)) if image is not None else None

    def open_image(self, image_path):
        return Image.open(str(image_path)) if image_path is not None else None

    def divisible_by_64(self, image):
        width, height = image.size
        if height % 64 != 0 or width % 64 != 0:
            width, height = map(lambda x: x - x % 64, (width, height))
            print(
                f"WARNING: Your image is of size {height}x{width} which is not divisible by 64. We are resizing to {height}x{width}!"
            )
        return width, height

    def apply(
        self,
        sizing_strategy,
        image=None,
    ):
        image = self.open_image(image)

        width, height = self.get_dimensions(image)

        if sizing_strategy == "max_width_1024":
            print("Using max width 1024")
            width = MAX_DIMENSION
            height = int(width / (width / height))
            if height % 64 != 0:
                height = height - (height % 64)
        else:
            print("Using image dimensions")
            width, height = self.divisible_by_64(image)

        resized_image = self.resize_image(
            image,
            width,
            height,
        )

        print(f"Using dimensions {width}x{height}")
        return width, height, resized_image
