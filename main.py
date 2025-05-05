from src import noise_detection, HOG, enhance_saltpepper_image, enhance_speckle_image, enhance_other_image
import argparse
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='CCTV Noise Reduction',
                    description='A targeted noise reduction algorithm for CCTV images.')
    
    parser.add_argument('--train', action='store_true', help='Re-run the training on the noise detection model. Defaults to False.')
    parser.add_argument('--test', action='store_true', help='Run the testing on the noise detection model. Defaults to False.')
    parser.add_argument('--image', type=str, help='Path to the image to be processed.')
    parser.add_argument('--output', type=str, help='Path to save the processed image. Defaults to out.png.', default="out.png")
    parser.add_argument('--algorithm', type=str, choices=['saltpepper', 'speckle', 'other'], help='Type of noise to enhance.')

    args = parser.parse_args()

    #########################################
    ## Training/Data colleciton Procedures ##
    #########################################
    
    df, generated = noise_detection.gather_noise_examples(generate_train=args.train)

    if generated:
        noise_detection.visualize_train_data(df)

    ########################
    ## Testing Procedures ##
    ########################
    
    if args.test:
        matrix, TP, TN, FP, FN, accuracy, precision, recall, f1 = noise_detection.model_evaluation(df)
        title = "Model Evaluation Results:"
        print(len(title) * "-")
        print(title)
        print(f"True Positives: {TP}")
        print(f"True Negatives: {TN}")
        print(f"False Positives: {FP}")
        print(f"False Negatives: {FN}")
        print(len(title)//2 * "-")
        print("Confusion Matrix:")
        for i, label in enumerate(["saltpepper", "speckle", "other"]):
            print(f"{label}: ")
            print(matrix[i])
        print(len(title)//2 * "-")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(len(title) * "-")

    #################################
    ## Image Processing Procedures ##      
    #################################

    if args.image:
        if not args.algorithm:
            hog = HOG(args.image)

            best_vote = noise_detection.predict_noise_type(hog, df)

            print(f"Predicted Noise Type: {best_vote}")
        else:
            best_vote = args.algorithm

        img = cv2.imread(args.image)

        if best_vote == "saltpepper":
            img = enhance_saltpepper_image(img)
        elif best_vote == "speckle":
            img = enhance_speckle_image(img)
        else:
            img = enhance_other_image(img)

        if args.output:
            cv2.imwrite(args.output, img)

        print(f"Processed image saved to {args.output}.")
    elif not args.test and not args.train:
        print("No image provided. Please provide an image to process.")
