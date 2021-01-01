#CREATION OF BACK LESS TITLE
import cv2
import numpy as np
img = cv2.imread('/Users/amanalisiddiqui/Documents/acm/logo.png')
img[img != 0] = 255 # change everything to white where pixel is not black
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
mask = 255 - mask
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
result = img.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:,:,3] = mask
cv2.imwrite('/Users/amanalisiddiqui/Documents/acm/title cut.png', result)

#CREATION OF BACK LESS DESIGN
#CREATES LOGOXX

def remove_background(img):
    # == Parameters =======================================================================
    BLUR = 5
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 100
    MASK_DILATE_ITER = 20
    MASK_ERODE_ITER = 20
    MASK_COLOR = (0.0, 0.0, 0.0)  # In BGR format

    # == Processing =======================================================================

    # -- Read image -----------------------------------------------------------------------
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))
    # cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype('uint8')  # Convert back to 8-bit
    masked=255+masked
    return masked
    #cv2.imwrite('/Users/amanalisiddiqui/Documents/acm/logoxx.png', masked)

remove_background('/Users/amanalisiddiqui/Documents/acm/logo.png')


img = remove_background('/Users/amanalisiddiqui/Documents/acm/logo.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
mask = 255 - mask
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
result = img.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:,:,3] = mask
cv2.imwrite('/Users/amanalisiddiqui/Documents/acm/logo cut.png', result)

#SUPERIMPOSE
from PIL import Image
img1 = Image.open("/Users/amanalisiddiqui/Documents/acm/logo cut.png")
img2 = Image.open("/Users/amanalisiddiqui/Documents/acm/title cut.png")
background = Image.open("/Users/amanalisiddiqui/Documents/acm/landscape-photography-tips-yosemite-valley-feature.jpg")
#i1=cv2.resize(img1,(1050, 1610))
#i2=cv2.resize(img2,(1050, 1610))
background.paste(img1, (0, 0))
background.paste(img2, (0, 0))
background.save('/Users/amanalisiddiqui/Documents/acm/final.png',"PNG")

