import cv2
import numpy as np
def overlayPNG(bgImg, imgPNG, pos, scale=1):

    # Load the foreground image with alpha channel
    imgRGB = cv2.cvtColor(imgPNG, cv2.COLOR_BGRA2RGB)

    # Resize the foreground image to match the specified scale
    imgRGB = cv2.resize(imgRGB, (0, 0), None, scale, scale)

    # Extract the alpha channel from the foreground image
    imgMask = imgPNG[:, :, 3]

    # Create a mask with three color channels
    imgMaskFull = cv2.cvtColor(imgMask, cv2.COLOR_GRAY2BGR)

    # Define the size of the second dimension of the imgMaskFull array based on the size of imgRGB array
    wf, hf = imgRGB.shape[1], imgRGB.shape[0]
    imgMaskFull = np.zeros((bgImg.shape[0], bgImg.shape[1], 3), np.uint8)

    # Overlay the foreground image onto the background image
    imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB

    # Apply the mask to the overlayed image and the background image
    imgMaskFull = cv2.bitwise_and(imgMaskFull, imgMaskFull, mask=imgMask)
    bgImg = cv2.bitwise_and(bgImg, bgImg, mask=cv2.bitwise_not(imgMask))

    # Add the overlayed image to the background image
    bgImg = cv2.add(bgImg, imgMaskFull)

    return bgImg
