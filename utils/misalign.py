import cv2
import numpy as np
import argparse

def find_misalignment(template_path, scanned_path, output_path="misalignment_output.png"):
    """
    Compares a scanned image against a template to find and highlight misalignments.

    Args:
        template_path (str): The file path for the perfect template image.
        scanned_path (str): The file path for the scanned print to check.
        output_path (str): The file path to save the result image.
    """
    print("Starting misalignment check...")

    # --- 1. Load Images ---
    # Read the template and the scanned image in grayscale.
    # Grayscale is sufficient for alignment and difference checking.
    try:
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        scanned_img = cv2.imread(scanned_path, cv2.IMREAD_GRAYSCALE)

        if template_img is None:
            print(f"Error: Could not load template image at {template_path}")
            return
        if scanned_img is None:
            print(f"Error: Could not load scanned image at {scanned_path}")
            return
            
    except Exception as e:
        print(f"An error occurred while reading the images: {e}")
        return

    print("Images loaded successfully.")

    # --- 2. Feature Detection and Matching ---
    # We use ORB (Oriented FAST and Rotated BRIEF) to find key features in both images.
    # ORB is efficient and free to use.
    MAX_FEATURES = 1000
    GOOD_MATCH_PERCENT = 0.15

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(template_img, None)
    keypoints2, descriptors2 = orb.detectAndCompute(scanned_img, None)

    # Match the features between the two images.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score. Use sorted() as it returns a new list from the tuple.
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep the best ones.
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    print(f"Found {len(matches)} good feature matches.")

    # --- 3. Image Alignment (Registration) ---
    # If we have enough good matches, we can align the scanned image to the template.
    if len(matches) > 10: # Need at least a few matches for a reliable transform
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find the homography matrix, which describes the perspective transformation.
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use the homography matrix to "warp" the scanned image to match the template's perspective.
        height, width = template_img.shape
        scanned_aligned = cv2.warpPerspective(scanned_img, h, (width, height))
        print("Scanned image aligned to template.")
    else:
        print("Not enough matches found to align images. Exiting.")
        # As a fallback, use the original scanned image, though results will be poor.
        scanned_aligned = scanned_img


    # --- 4. Difference Analysis ---
    # Now that the images are aligned, we can find the differences.
    
    # Calculate the absolute difference between the template and the aligned scan.
    # Areas that are the same will be black (0), differences will be lighter.
    diff = cv2.absdiff(template_img, scanned_aligned)

    # Apply a threshold. This makes the differences stand out as white pixels.
    # Any pixel value > 30 is considered a difference and set to 255 (white).
    thresh_val, thresh_img = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    print("Difference image calculated and thresholded.")

    # --- 5. Find and Highlight Errors ---
    # Find the contours (outlines) of the white areas (the errors).
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Load the original color scanned image to draw the results on.
    scanned_color = cv2.imread(scanned_path)
    
    misalignment_detected = False
    min_contour_area = 50 # Ignore very small, insignificant differences (noise).

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            misalignment_detected = True
            # Get a bounding box around the contour and draw it on the color image.
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(scanned_color, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red rectangle

    # --- 6. Output Result ---
    if misalignment_detected:
        print("\nResult: MISALIGNMENT DETECTED!")
        cv2.putText(scanned_color, "Misalignment Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        print("\nResult: No significant misalignment found.")
        cv2.putText(scanned_color, "No Misalignment Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Save the final image with the highlighted errors.
    cv2.imwrite(output_path, scanned_color)
    print(f"Output saved to {output_path}")


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Find misalignments between a template and a scanned image.")
    parser.add_argument("template", help="Path to the template image file.")
    parser.add_argument("scanned", help="Path to the scanned image file.")
    parser.add_argument("-o", "--output", default="misalignment_output.png", help="Path to save the output image.")
    
    args = parser.parse_args()

    # Run the main function with the provided arguments
    find_misalignment(args.template, args.scanned, args.output)

