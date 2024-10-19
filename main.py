import cv2
import numpy as np
import os
from pathlib import Path

def load_image(image_path):
    return cv2.imread(image_path)

def resize_image(image, scale_factor):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

def rotate_image(image, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    return cv2.warpAffine(image, matrix, (width, height))

def translate_image(image, x, y):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, matrix, (width, height))

def smooth_image(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def zoom_image(image, zoom_factor):
    h, w = image.shape[:2]
    if zoom_factor > 1:
        crop_h, crop_w = int(h / zoom_factor), int(w / zoom_factor)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        cropped = image[top:top+crop_h, left:left+crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        target_h, target_w = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[top:top+target_h, left:left+target_w] = resized
        return result

def pad_image(image, top, bottom, left, right, color=(0, 0, 0)):
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def vertical_crop(image, position):
    h, w = image.shape[:2]
    crop_height = int(h * 0.6)
    
    if position == 'top':
        return image[:crop_height, :]
    elif position == 'middle':
        start = (h - crop_height) // 2
        return image[start:start+crop_height, :]
    elif position == 'bottom':
        return image[-crop_height:, :]
    else:
        return image

def augment_image(image, output_dir, num_augmentations=100):
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]
    
    for i in range(num_augmentations):
        position = np.random.choice(['top', 'middle', 'bottom', 'full'])
        cropped_image = vertical_crop(image, position) if position != 'full' else image
        
        zoom_factor = np.random.uniform(0.5, 2.5)
        zoomed_image = zoom_image(cropped_image, zoom_factor)
        
        max_translate = int(min(h, w) * 0.2)
        tx = np.random.randint(-max_translate, max_translate)
        ty = np.random.randint(-max_translate, max_translate)
        positioned_image = translate_image(zoomed_image, tx, ty)
        
        pad_top = max(0, -ty)
        pad_bottom = max(0, ty + zoomed_image.shape[0] - h)
        pad_left = max(0, -tx)
        pad_right = max(0, tx + zoomed_image.shape[1] - w)
        padded_image = pad_image(positioned_image, pad_top, pad_bottom, pad_left, pad_right)
        
        rotation_angle = np.random.uniform(-45, 45)
        rotated_image = rotate_image(padded_image, rotation_angle)
        
        if np.random.choice([True, False]):
            kernel_size = np.random.choice([3, 5, 7])
            processed_image = smooth_image(rotated_image, kernel_size)
        else:
            processed_image = sharpen_image(rotated_image)
        
        output_path = os.path.join(output_dir, f"augmented_{i+1}_{position}.jpg")
        cv2.imwrite(output_path, processed_image)
        
    print(f"{num_augmentations} augmented images have been generated and saved in {output_dir}")

def main():
    input_image_path = "./img.png" 
    output_directory = "./outputs/"
    
    image = load_image(input_image_path)
    if image is None:
        print(f"Error: Unable to load the input image from {input_image_path}")
        return
    
    augment_image(image, output_directory)

if __name__ == "__main__":
    main()