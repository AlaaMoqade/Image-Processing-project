import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and display the grayscale images
img_paths = [
    r'C:\Users\amjad\OneDrive\Desktop\opencvapp\im1-jpg.png',
    r'C:\Users\amjad\OneDrive\Desktop\opencvapp\lena.jpg',
    # Add more image paths as needed
]

def load_and_show_images(img_paths):
    images = []
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to load image at path {img_path}")
            continue
        images.append(img)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.show()
    return images

images = load_and_show_images(img_paths)

# Ensure that images were loaded successfully
if not images:
    print("No images were loaded. Please check the file paths.")
    exit()

# Manually mark expected edge locations on the images
manual_edges_list = []
colored_imgs = []

def mark_edge(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(manual_edges, (x, y), 1, 255, -1)
        cv2.circle(colored_img, (x, y), 1, (0, 255, 0), -1)
        cv2.imshow('Manual Edge Marking', colored_img)

for img in images:
    manual_edges = np.zeros_like(img, dtype=np.uint8)
    colored_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    manual_edges_list.append(manual_edges)
    colored_imgs.append(colored_img)

    cv2.imshow('Manual Edge Marking', colored_img)
    cv2.setMouseCallback('Manual Edge Marking', mark_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for manual_edges in manual_edges_list:
    plt.imshow(manual_edges, cmap='gray')
    plt.title('Manually Marked Edges')
    plt.axis('off')
    plt.show()

# Apply Edge Detection Operators
def apply_operator(gray, kernel_x, kernel_y):
    grad_x = cv2.filter2D(gray, cv2.CV_16S, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_16S, kernel_y)
    return cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0))

kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=int)
kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=int)
kernel_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
kernel_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)

# Apply edge detection to all images
edges_roberts_list = []
edges_sobel_list = []
edges_prewitt_list = []

for img in images:
    edges_roberts = apply_operator(img, kernel_roberts_x, kernel_roberts_y)
    edges_sobel = apply_operator(img, kernel_sobel_x, kernel_sobel_y)
    edges_prewitt = apply_operator(img, kernel_prewitt_x, kernel_prewitt_y)
    
    edges_roberts_list.append(edges_roberts)
    edges_sobel_list.append(edges_sobel)
    edges_prewitt_list.append(edges_prewitt)

# Threshold Adjustment
def threshold_edges(edges, threshold=100):
    _, thresh_edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    return thresh_edges

# Compare and Evaluate Results
def compare_and_evaluate(detected, manual):
    comparison = np.hstack((manual, detected))
    difference = cv2.absdiff(manual, detected)
    score = np.sum(difference) / 255
    return comparison, score

# Function to evaluate different thresholds
def evaluate_thresholds(edges, manual, threshold_values):
    best_score = float('inf')
    best_thresh = None
    best_edges = None

    for threshold in threshold_values:
        thresh_edges = threshold_edges(edges, threshold)
        _, score = compare_and_evaluate(thresh_edges, manual)
        if score < best_score:
            best_score = score
            best_thresh = threshold
            best_edges = thresh_edges

    return best_edges, best_thresh, best_score

# Evaluate and compare results for all images
for i, img in enumerate(images):
    edges_roberts_thresh, best_thresh_roberts, score_roberts = evaluate_thresholds(edges_roberts_list[i], manual_edges_list[i], range(50, 151, 10))
    edges_sobel_thresh, best_thresh_sobel, score_sobel = evaluate_thresholds(edges_sobel_list[i], manual_edges_list[i], range(50, 151, 10))
    edges_prewitt_thresh, best_thresh_prewitt, score_prewitt = evaluate_thresholds(edges_prewitt_list[i], manual_edges_list[i], range(50, 151, 10))
    
    comp_roberts, _ = compare_and_evaluate(edges_roberts_thresh, manual_edges_list[i])
    comp_sobel, _ = compare_and_evaluate(edges_sobel_thresh, manual_edges_list[i])
    comp_prewitt, _ = compare_and_evaluate(edges_prewitt_thresh, manual_edges_list[i])
    
    print(f"Image {i+1} Evaluation Scores (Lower is better):")
    print(f"Roberts: {score_roberts} (Best Threshold: {best_thresh_roberts})")
    print(f"Sobel: {score_sobel} (Best Threshold: {best_thresh_sobel})")
    print(f"Prewitt: {score_prewitt} (Best Threshold: {best_thresh_prewitt})")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(comp_roberts, cmap='gray'), plt.title(f'Roberts Comparison\nBest Threshold: {best_thresh_roberts}')
    plt.subplot(132), plt.imshow(comp_sobel, cmap='gray'), plt.title(f'Sobel Comparison\nBest Threshold: {best_thresh_sobel}')
    plt.subplot(133), plt.imshow(comp_prewitt, cmap='gray'), plt.title(f'Prewitt Comparison\nBest Threshold: {best_thresh_prewitt}')
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(edges_roberts_thresh, cmap='gray'), plt.title(f'Roberts Edges\nBest Threshold: {best_thresh_roberts}')
    plt.subplot(132), plt.imshow(edges_sobel_thresh, cmap='gray'), plt.title(f'Sobel Edges\nBest Threshold: {best_thresh_sobel}')
    plt.subplot(133), plt.imshow(edges_prewitt_thresh, cmap='gray'), plt.title(f'Prewitt Edges\nBest Threshold: {best_thresh_prewitt}')
    plt.axis('off')
    plt.show()
