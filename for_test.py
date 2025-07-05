#num1
img = cv2.imread('assets/logo.jpg', 1)
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite('new_img.jpg', img)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



import cv2
import random

img = cv2.imread('assets/logo.jpg', -1)

# Change first 100 rows to random pixels
#נעבור על כל הפיקסלים ונשנה לרנדמלי. דרך לייצר רעש בתמונה
for i in range(100):
	for j in range(img.shape[1]):
		img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

# Copy part of image
#העתקה והדבקה מקטע של התמונה
tag = img[500:700, 600:900]
img[100:300, 650:950] = tag

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import pandas as pd
import scipy.cluster.hierarchy as sch

# === קריאת תמונה והמרה לגווני אפור ===
# קרא תמונה והפוך אותה לגווני אפור
img = imageio.imread('your_image.jpg')  # ← החלף בשם קובץ התמונה שלך
if img.ndim == 3:
    img = img.mean(axis=2).astype(np.uint8)

# הצג את התמונה המקורית
plt.figure()
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

# === בינאריזציה לפי סף ===
def binary_segmentation(im, threshold=128):
    # מחזירה תמונה בשחור-לבן לפי ערך סף
    return (im > threshold).astype(np.uint8) * 255

bin_img = binary_segmentation(img)

# הצגת תמונה בינארית
plt.figure()
plt.title("Binary Segmentation")
plt.imshow(bin_img, cmap='gray')
plt.axis('off')

# === כיווץ עמודות לפי ממוצע (Squeeze) ===
def squeeze_image(im, factor):
    # כל factor עמודות ממוצעות לעמודה אחת
    new_n = im.shape[0]
    new_m = im.shape[1] // factor
    new_img = np.zeros((new_n, new_m))
    for j in range(new_m):
        cols = im[:, j*factor:(j+1)*factor]
        new_img[:, j] = cols.mean(axis=1)
    return new_img.astype(np.uint8)

squeezed = squeeze_image(img, 4)

# הצגת תמונה מכווצת
plt.figure()
plt.title("Squeezed Image")
plt.imshow(squeezed, cmap='gray')
plt.axis('off')

# === Morphology: פעולות על שכנות של פיקסל ===
def get_neighborhood(im, x, y, dx=1, dy=1):
    # מחזירה את הסביבה של פיקסל במרחק dx,dy
    x1 = max(x - dx, 0)
    x2 = min(x + dx + 1, im.shape[1])
    y1 = max(y - dy, 0)
    y2 = min(y + dy + 1, im.shape[0])
    return im[y1:y2, x1:x2]

def morph_by_neighborhood(im, operator, dx=1, dy=1):
    # מחילה פונקציה על כל סביבה בתמונה
    new_im = np.zeros_like(im)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            nbr = get_neighborhood(im, x, y, dx, dy)
            new_im[y, x] = operator(nbr)
    return new_im.astype(np.uint8)

# דילול (Erosion) = מינימום
erosion = morph_by_neighborhood(bin_img, np.min)
plt.figure()
plt.title("Erosion")
plt.imshow(erosion, cmap='gray')
plt.axis('off')

# הרחבה (Dilation) = מקסימום
dilation = morph_by_neighborhood(bin_img, np.max)
plt.figure()
plt.title("Dilation")
plt.imshow(dilation, cmap='gray')
plt.axis('off')

# === ניקוי רעש עם ממוצע וחציון ===
def denoise_mean(im, dx=1, dy=1):
    # מנקה רעש לפי ממוצע של סביבה
    return morph_by_neighborhood(im, lambda x: np.mean(x), dx, dy)

def denoise_median(im, dx=1, dy=1):
    # מנקה רעש לפי חציון של סביבה
    return morph_by_neighborhood(im, lambda x: np.median(x), dx, dy)

plt.figure()
plt.title("Mean Denoised")
plt.imshow(denoise_mean(img), cmap='gray')
plt.axis('off')

plt.figure()
plt.title("Median Denoised")
plt.imshow(denoise_median(img), cmap='gray')
plt.axis('off')

# === גרדיאנט – הפרש בין פיקסל למי שמעליו ===
def image_gradient(im):
    zero_row = np.zeros((1, im.shape[1]), dtype=np.int32)
    up_shifted = np.vstack((im[1:], zero_row))
    diff = np.abs(np.int32(up_shifted) - np.int32(im))
    return diff.astype(np.uint8)

gradient = image_gradient(img)
plt.figure()
plt.title("Gradient (Vertical)")
plt.imshow(gradient, cmap='gray')
plt.axis('off')

# הגברת הגרדיאנט
brightened = np.minimum(gradient * 5, 255).astype(np.uint8)
plt.figure()
plt.title("Brightened Gradient")
plt.imshow(brightened, cmap='gray')
plt.axis('off')

# === חלק 2: ניתוח טבלאות ציונים (Lecture 13) ===

# יצירת DataFrame לדוגמה
grades_df = pd.DataFrame({
    'Student': ['Yael', 'Nadav', 'Michal', 'Shoshana', 'Danielle',
                'Omer', 'Yarden', 'Avi', 'Roy', 'Tal'],
    'Math': [94, 65, 58, 80, 90, 85, 92, 55, 92, 88],
    'History': [95, 70, 60, 78, 90, 81, 87, 53, 90, 86],
    'Biology': [60, 92, 45, 89, 93, 91, 54, 48, 91, 58]
})

students = grades_df['Student'].values
grades = grades_df.drop(columns='Student').values

# כמה נכשלות יש
print("Total Fails:", np.sum(grades < 60))

# נכשלות לכל סטודנט
print("Fails per student:")
for i, name in enumerate(students):
    count = np.sum(grades[:, i] < 60) if grades.shape[1] > i else 0
    print(f"{name}: {count}")

# Boxplot
plt.figure()
plt.boxplot(grades, labels=grades_df.columns[1:])
plt.title("Grades Boxplot")
plt.ylabel("Grade")
plt.show()

# Heatmap
plt.figure()
plt.imshow(grades, cmap='hot', aspect='auto')
plt.title("Grades Heatmap")
plt.colorbar()
plt.yticks(ticks=range(len(students)), labels=students)
plt.xticks(ticks=range(grades.shape[1]), labels=grades_df.columns[1:])
plt.tight_layout()
plt.show()

# Dendrogram (clustering)
Z = sch.linkage(grades.T, method='ward')
plt.figure()
sch.dendrogram(Z, labels=grades_df.columns[1:])
plt.title("Dendrogram (Courses)")
plt.show()

