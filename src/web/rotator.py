from scipy import ndimage
import tensorflow as tf
import cv2

class Rotator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
    def __call__(self, img):
        work_img = cv2.resize(img, (256, 256)) / 255
        pred = self.model.predict(work_img[None])
        img = ndimage.rotate(img, -int(pred), cval=255)
        return img


if __name__ == "__main__":
    rotator = Rotator(f"rotator_6.2")

    for img_path in list((data_dir / 'test').glob('*/*')):
        img = cv2.imread(str(img_path))
        img = rotator(img)

        plt.imshow(img)
        plt.title('my picture')
        plt.show()