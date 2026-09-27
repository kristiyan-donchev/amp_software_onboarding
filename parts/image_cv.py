import cv2
import numpy as np

class Object_Detection():

    def run(self, image):
        if image is not None:

            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # apply binary thresholding
            ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
            # visualize the binary image
            cv2.imshow('Binary image', thresh)
            cv2.waitKey(0)
            cv2.imwrite('image_thres1.jpg', thresh)
            cv2.destroyAllWindows()
            
            # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
            contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

            if len(contours) == 0:
                return image, 0, 0, 0

            # draw contours on the original image
            image_copy = image.copy()
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # see the results
            cv2.imshow('None approximation', image_copy)
            cv2.waitKey(0)
            cv2.imwrite('contours_none_image1.jpg', image_copy)
            cv2.destroyAllWindows()

            maxContour = None
            maxArea = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > maxArea:
                    maxArea = area
                    maxContour = contour

            if maxContour is None:
                return image, 0, 0, 0

            # Draw largest contour
            cv2.drawContours(image, [maxContour], 0, (255,0,0), 2)

            # Draw Contour centroid
            M = cv2.moments(maxContour)

            # Calculate the centroid of the contour with moments
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
            else:
                centroid_x = 0
                centroid_y = 0

            # Draw centroid
            cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # Return image, centroid_x, centroid_y, max_area
            return image, centroid_x, centroid_y, maxArea