# views.py

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
from .forms import ImageUploadForm
import os
from django.conf import settings

def home(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            file_url = fs.url(filename)

            image_path = os.path.join(settings.MEDIA_ROOT, filename)
            image = cv2.imread(image_path)
            if image is None:
                return render(request, 'index.html', {'form': form, 'error': 'Could not read image.'})

            # Apply Gaussian blur
            blurred_image = apply_gaussian_blur(image)

            # Edge detection
            edges = edge_detection(blurred_image)

            # Non-maximum suppression
            nms_edges = non_maximum_suppression(edges)

            # Vectorize edges
            contours = vectorize_edges(nms_edges)

            # Detect number plate
            number_plate_image = detect_number_plate(image, contours)

            processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_' + filename)
            cv2.imwrite(processed_image_path, number_plate_image)

            context = {
                'form': form,
                'file_url': file_url,
                'processed_file_url': fs.url('processed_' + filename),
            }
            return render(request, 'index.html', context)
    else:
        form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})

def apply_gaussian_blur(image, ksize=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, ksize, sigma)

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def non_maximum_suppression(edges):
    if hasattr(cv2, 'ximgproc'):
        return cv2.ximgproc.thinning(edges, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    else:
        return edges

def vectorize_edges(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_number_plate(image, contours):
    number_plate_image = image.copy()

    best_contour = None
    max_area = 0

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        width = rect[1][0]
        height = rect[1][1]
        aspect_ratio = width / height if height > 0 else 0
        area = width * height

        if 2 < aspect_ratio < 6 and 1000 < area < 15000:
            if area > max_area:
                best_contour = box
                max_area = area

    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        cv2.drawContours(image, [best_contour], 0, (0, 255, 0), 2)
        number_plate_image = four_point_transform(image, best_contour)

    return number_plate_image

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 3) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
