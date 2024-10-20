import os
import random
from PIL import Image, ImageEnhance, ImageOps
import itertools

input_folder = './inputs/'
output_folder = './outputs/'

os.makedirs(output_folder, exist_ok=True)

def flip_image(image):
    return ImageOps.mirror(image)

def rotate_image(image):
    angle = random.randint(-30, 30)
    return image.rotate(angle, expand=True)

def adjust_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)

def adjust_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)

def adjust_color(image):
    enhancer = ImageEnhance.Color(image)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)

def adjust_sharpness(image):
    enhancer = ImageEnhance.Sharpness(image)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)

def zoom_image(image, zoom_factor):
    w, h = image.size

    # zoom in factor
    if zoom_factor > 1:
        crop_w, crop_h = int(w / zoom_factor), int(h / zoom_factor)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((w, h), Image.LANCZOS)
    # zoom out factor
    else:
        target_w, target_h = int(w * zoom_factor), int(h * zoom_factor)
        resized = image.resize((target_w, target_h), Image.LANCZOS)
        new_img = Image.new('RGB', (w, h), (0, 0, 0))
        paste_left = (w - target_w) // 2
        paste_top = (h - target_h) // 2
        new_img.paste(resized, (paste_left, paste_top))
        return new_img

def translate_image(image, dx, dy):
    return ImageOps.expand(image, border=(dx, dy, -dx, -dy), fill=0)

def vertical_crop(image, position):
    w, h = image.size
    crop_height = int(h * 0.6)
    if position == 'top':
        return image.crop((0, 0, w, crop_height))
    elif position == 'middle':
        start = (h - crop_height) // 2
        return image.crop((0, start, w, start + crop_height))
    elif position == 'bottom':
        return image.crop((0, h - crop_height, w, h))
    else:
        return image

augmentations = {
    'flip': flip_image,
    'rotate': rotate_image,
    'brightness': adjust_brightness,
    'contrast': adjust_contrast,
    'color': adjust_color,
    'sharpness': adjust_sharpness,
    'zoom': lambda img: zoom_image(img, random.uniform(0.5, 2.5)),
    'translate': lambda img: translate_image(img, random.randint(-50, 50), random.randint(-50, 50)),
    'vertical_crop': lambda img: vertical_crop(img, random.choice(['top', 'middle', 'bottom', 'full']))
}

def generate_unique_combinations(num_augmentations):
    all_combinations = list(itertools.chain.from_iterable(
        itertools.combinations(augmentations.keys(), r) for r in range(1, len(augmentations) + 1)
    ))
    return random.sample(all_combinations, min(num_augmentations, len(all_combinations)))

min_images = input("Enter minimum number of images to generate: ")
max_images = input("Enter maximum number of images to generate: ")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        num_augmented_images = random.randint(int(min_images), int(max_images))
        unique_combinations = generate_unique_combinations(num_augmented_images)
        
        for i, combination in enumerate(unique_combinations):
            augmented_image = image.copy()
            
            for aug_name in combination:
                augmented_image = augmentations[aug_name](augmented_image)
            
            new_filename = f"{os.path.splitext(filename)[0]}_aug_{i+1}{os.path.splitext(filename)[1]}"
            augmented_image.save(os.path.join(output_folder, new_filename))
        
        print(f"Generated {len(unique_combinations)} unique augmented images for {filename}")

print("Done.")