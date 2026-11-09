import numpy as np
import matplotlib.pyplot as plt

def pad_image(image: np.ndarray, pad: int, mode: str) -> np.ndarray:
    # сюда потом вставишь реализацию
    pass

def box_blur(image: np.ndarray, kernel_size: int, mode: str) -> np.ndarray:
    # сюда потом вставишь реализацию
    pass

# ИСПРАВЛЕНО: было -> Image, стало -> np.ndarray
def make_test_image(size: int = 40) -> np.ndarray:
    import random
    random.seed(0)
    img = np.zeros((size, size))
    for i in range(size):
        row = np.zeros(size)
        for j in range(size):
            base = 220.0 if (i // 5 + j // 5) % 2 == 0 else 40.0
            noise = random.uniform(-15, 15)
            row[j] = max(0.0, min(255.0, base + noise))
        img[i] = row
    return img

def main():
    # ИСПРАВЛЕНО: импорты убраны из функции, они уже есть в начале файла
    image = make_test_image(size=40)

    # ОПЦИОНАЛЬНО: тут было несоответствие подписей и значений (k=3 в заголовке, но 9 в коде)
    # Если по заданию надо именно 9 — оставь 9. Если надо, чтобы подписи совпадали — поставь 3.
    blurred_k3 = box_blur(image, kernel_size=3, mode="edge")
    blurred_k9 = box_blur(image, kernel_size=9, mode="zero")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["исходное", "Blur k=3 (edge)", "Blur k=9 (zero)"]
    data = [image, blurred_k3, blurred_k9]

    for ax, title, d in zip(axes, titles, data):
        ax.imshow(d, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
