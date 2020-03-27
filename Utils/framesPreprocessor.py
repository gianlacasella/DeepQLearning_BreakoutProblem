from tensorflow import image, resize_images

# Transforms frames from 210x160x3 Tensors to 84x84x1 Tensor (84x84 grayscale picture)
class Preprocessor:
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width

    def preprocessFrame(self, frame):
        # To grayscale
        self.returning_tensor = tensorflow.image.rgb_to_grayscale(frame)
        # Cropping
        self.returning_tensor = tensorflow.image.crop_to_bounding_box(self.returning_tensor, 34, 0, 160, 160)
        # Resizing to target_heightxtarget_width with nearest neighbor method
        self.returning_tensor = tensorflow.resize_images(self.returning_tensor, [self.target_height, self.target_width],
                                                         method=tensorflow.ResizeMethod.NEAREST_NEIGHBOR)
        return self.returning_tensor
