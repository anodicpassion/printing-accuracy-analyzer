# main.py
import cv2
import argparse

def analyze_blur(image_path, threshold):
    """
    Analyzes an image for blurriness using the Laplacian variance method.

    Args:
        image_path (str): The file path to the image to be analyzed.
        threshold (float): The blur detection threshold. Lower values are more
                           likely to be flagged as blurry.

    Returns:
        tuple: A tuple containing:
            - variance (float): The calculated Laplacian variance.
            - is_blurry (bool): True if the image is blurry, False otherwise.
    """
    try:
        # Step 1: Load the image from the specified path
        # The image is loaded in full color by default.
        image = cv2.imread(image_path)

        # Check if the image was loaded successfully. If not, imread returns None.
        if image is None:
            print(f"Error: Could not read image from path: {image_path}")
            return None, None

        # Step 2: Convert the image to grayscale
        # Blur detection doesn't require color information, and working with a
        # single channel is computationally more efficient.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 3: Compute the Laplacian of the grayscale image
        # The Laplacian operator highlights regions of rapid intensity change,
        # which correspond to edges in the image.
        # ksize=3 specifies a 3x3 kernel for the operator.
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        # Step 4: Calculate the variance of the Laplacian
        # The variance of the Laplacian is a single number that represents the
        # "sharpness" of the image. A high variance indicates many sharp edges,
        # while a low variance indicates a lack of edges, i.e., blur.
        variance = laplacian.var()

        # Step 5: Compare the variance to the user-defined threshold
        # If the variance is below the threshold, we classify the image as blurry.
        is_blurry = variance < threshold
        
        return variance, is_blurry

    except Exception as e:
        print(f"An error occurred during image processing: {e}")
        return None, None

def main():
    """
    Main function to parse command-line arguments and run the blur analysis.
    """
    # Set up an argument parser to handle command-line inputs.
    # This makes the script easy to use and integrate into automated workflows.
    parser = argparse.ArgumentParser(description="Detect blur in an image using the Laplacian variance method.")
    
    # Required argument: the path to the image file.
    parser.add_argument("-i", "--image", required=True, help="Path to the input image file.")
    
    # Optional argument: the blur threshold.
    # We provide a default value, but this is the most important parameter
    # to tune for your specific use case (scanner, print quality, etc.).
    parser.add_argument("-t", "--threshold", type=float, default=100.0,
                        help="Blur detection threshold. Lower is more blurry. (Default: 100.0)")

    args = parser.parse_args()

    # --- Run the analysis ---
    variance, is_blurry = analyze_blur(args.image, args.threshold)

    # --- Display the results ---
    if variance is not None:
        # Prepare the result text.
        result_text = "Blurred" if is_blurry else "Not Blurred"
        
        # Print the final verdict and the calculated score for reference.
        print(f"Image: {args.image}")
        print(f"Laplacian Variance Score: {variance:.2f}")
        print(f"Threshold: {args.threshold}")
        print(f"Verdict: {result_text}")

        # You can also load and display the image with the verdict written on it.
        # This is useful for visual verification.
        image_to_show = cv2.imread(args.image)
        cv2.putText(image_to_show, f"{result_text} (Score: {variance:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if is_blurry else (0, 255, 0), 2)
        
        # To display the image in a window:
        # cv2.imshow("Image Analysis", image_to_show)
        # cv2.waitKey(0) # Wait for a key press to close the window
        # cv2.destroyAllWindows()

        # To save the output image with the verdict:
        output_filename = "analyzed_" + args.image.split('/')[-1]
        cv2.imwrite(output_filename, image_to_show)
        print(f"Saved analyzed image to {output_filename}")


if __name__ == "__main__":
    main()
