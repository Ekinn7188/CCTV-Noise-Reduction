import cv2

def enhance_saltpepper_image(img, median_ksize=3, clahe_clip=2.0, clahe_tile=(8, 8), l_scale=1.0, sharpen_amount=3, color_restore_strength=0.3):
    img_original = img.copy()

    denoised = cv2.medianBlur(img, median_ksize)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    l_clahe = clahe.apply(l)

    l_clahe = cv2.multiply(l_clahe, l_scale)
    l_clahe = cv2.convertScaleAbs(l_clahe)

    lab_clahe = cv2.merge((l_clahe, a, b))
    contrasted = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=0.3)
    sharpened = cv2.addWeighted(contrasted, 1 + sharpen_amount, blurred, -sharpen_amount, 0)

    final = cv2.addWeighted(sharpened, 1 - color_restore_strength, img_original, color_restore_strength, 0)

    return final

def enhance_speckle_image(img, bilateral_d=9, sigmaColor=100, sigmaSpace=50, clahe_clip=2.0, clahe_tile=(8, 8), l_scale=1.0, sharpen_amount=1.2, color_restore_strength=0.4):
    img_original = img.copy()

    denoised = cv2.bilateralFilter(img, d=bilateral_d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    l_clahe = clahe.apply(l)

    l_clahe = cv2.multiply(l_clahe, l_scale)
    l_clahe = cv2.convertScaleAbs(l_clahe)

    lab_clahe = cv2.merge((l_clahe, a, b))
    contrasted = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=0.25)
    sharpened = cv2.addWeighted(contrasted, 1 + sharpen_amount, blurred, -sharpen_amount, 0)

    final = cv2.addWeighted(sharpened, 1 - color_restore_strength, img_original, color_restore_strength, 0)

    return final


def enhance_other_image(img, denoise_h=3, denoise_hColor=0, clahe_clip=2.0, clahe_tile=(8, 8), l_scale=1.0, sharpen_amount=1.0, blur_sigma=0.25, brightness_shift=-5, color_restore_strength=0.50):
    img_original = img.copy()

    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=denoise_h, hColor=denoise_hColor, templateWindowSize=15, searchWindowSize=31)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    l_clahe = clahe.apply(l)

    l_clahe = cv2.multiply(l_clahe, l_scale)
    l_clahe = cv2.convertScaleAbs(l_clahe)

    lab_clahe = cv2.merge((l_clahe, a, b))
    contrasted = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=blur_sigma)
    sharpened = cv2.addWeighted(blurred, 1 + sharpen_amount, denoised, -sharpen_amount, 0)

    adjusted = cv2.convertScaleAbs(sharpened, alpha=1.0, beta=brightness_shift)

    final = cv2.addWeighted(adjusted, 1 - color_restore_strength, img_original, color_restore_strength, 0)
    
    return final