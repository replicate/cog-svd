from PIL import Image

MAX_W_DIMENSION = 1024
MAX_H_DIMENSION = 576


class SizingStrategy:
    def __init__(self):
        pass

    def maintain_aspect_ratio(self, width, height):
        aspect_ratio = width / height

        if aspect_ratio >= 1:  # Width is the limiting factor
            new_width = min(width, MAX_W_DIMENSION)
            new_height = int(new_width / aspect_ratio)
        else:  # Height is the limiting factor
            new_height = min(height, MAX_H_DIMENSION)
            new_width = int(new_height * aspect_ratio)

        # Ensure neither dimension exceeds the maximum
        if new_height > MAX_H_DIMENSION:
            new_height = MAX_H_DIMENSION
            new_width = int(new_height * aspect_ratio)

        if new_width > MAX_W_DIMENSION:
            new_width = MAX_W_DIMENSION
            new_height = int(new_width / aspect_ratio)

        # Adjust to be divisible by 64
        new_width -= new_width % 64
        new_height -= new_height % 64

        return new_width, new_height

    def resize_and_crop(self, width, height, image):
        # Determine which dimension is less constraining
        scale_factor_w = MAX_W_DIMENSION / width
        scale_factor_h = MAX_H_DIMENSION / height

        print(f"Scale factor w: {scale_factor_w}, Scale factor h: {scale_factor_h}")

        # Scale up/down based on the less constraining dimension
        if scale_factor_w < scale_factor_h:
            # Height is less constraining
            new_height = MAX_H_DIMENSION
            new_width = int(width * scale_factor_h)
        else:
            # Width is less constraining
            new_width = MAX_W_DIMENSION
            new_height = int(height * scale_factor_w)

        print(f"New width: {new_width}, New height: {new_height}")

        # Resize the image
        resized_image = self.resize_image(image, new_width, new_height)

        # Calculate cropping dimensions
        left = max((new_width - MAX_W_DIMENSION) / 2, 0)
        top = max((new_height - MAX_H_DIMENSION) / 2, 0)
        right = left + MAX_W_DIMENSION
        bottom = top + MAX_H_DIMENSION

        print(f"Left: {left}, Top: {top}, Right: {right}, Bottom: {bottom}")

        # Crop the image to 1024x576
        cropped_image = resized_image.crop((left, top, right, bottom))

        print("Resized and cropped dimensions: 1024x576")
        return cropped_image

    def get_dimensions(self, image):
        return image.size

    def resize_image(self, image, width, height):
        return image.resize((width, height)) if image is not None else None

    def open_image(self, image_path):
        return Image.open(str(image_path)) if image_path is not None else None

    def divisible_by_64(self, image):
        width, height = image.size
        print(f"Original dimensions: {width}x{height}")
        if height % 64 != 0 or width % 64 != 0:
            width, height = map(lambda x: x - x % 64, (width, height))
            print(
                f"WARNING: Your image is not divisible by 64 – resizing to {width}x{height}"
            )
        return width, height

    def apply(
        self,
        sizing_strategy,
        image=None,
    ):
        image = self.open_image(image)
        width, height = self.get_dimensions(image)

        if sizing_strategy == "crop_to_16_9":
            print("Resizing and cropping to 16:9")
            return self.resize_and_crop(width, height, image)
        elif sizing_strategy == "maintain_aspect_ratio":
            print("Resizing but keeping aspect ratio")
            width, height = self.maintain_aspect_ratio(width, height)
        else:
            print("Using image dimensions")
            width, height = self.divisible_by_64(image)

        resized_image = self.resize_image(
            image,
            width,
            height,
        )

        print(f"Using dimensions {width}x{height}")
        return resized_image
