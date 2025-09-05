#!/usr/bin/env python3

import face_recognition
import pickle
from pathlib import Path
import argparse

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# create project directories
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    load the training images, encode the faces, and save the encodings to a file.
    """
    names = []
    encodings = []

    # for each person in the training directory
    for person_dir in Path("training").iterdir():
        if person_dir.is_dir():
            person_name = person_dir.name
            print(f"loading {person_name} images...")
            
            # for each image of the person
            for image_path in person_dir.glob("*"):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    print(f"   processing {image_path.name}")
                    
                    # load the image
                    image = face_recognition.load_image_file(image_path)
                    
                    # extract face encoding
                    face_encodings = face_recognition.face_encodings(image, model=model)
                    
                    if face_encodings:
                        # get the first face (assuming one face per image)
                        encoding = face_encodings[0]
                        encodings.append(encoding)
                        names.append(person_name)
                        print(f"   encoding created for {person_name}")
                    else:
                        print(f"   no face found in {image_path.name}")

    # save the encodings to a file
    name_encodings = {"names": names, "encodings": encodings}
    
    with open(encodings_location, "wb") as f:
        pickle.dump(name_encodings, f)
    
    print(f"\nencodings saved to: {encodings_location}")
    print(f"total {len(names)} face encodings created")
    
    # statistics
    unique_names = set(names)
    for name in unique_names:
        count = names.count(name)
        print(f"   - {name}: {count} encoding(s)")

def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    recognize faces in the given image
    """
    # load the encodings
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)

    # load the test image
    input_image = face_recognition.load_image_file(image_location)

    # find faces in the test image
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    print(f"found {len(input_face_encodings)} face(s) in test image")

    # try to recognize each face
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        # compare this face with known faces
        matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding
        )
        
        name = "unknown"
        if True in matches:
            # get the first match
            first_match_index = matches.index(True)
            name = loaded_encodings["names"][first_match_index]

        print(f"   recognized person: {name}")

def validate(model: str = "hog"):
    """
    test the images in validation directory
    """
    print("validation test starting...")
    
    validation_dir = Path("validation")
    if not any(validation_dir.iterdir()):
        print("validation directory is empty!")
        return
    
    for image_path in validation_dir.glob("*"):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"\ntesting: {image_path.name}")
            recognize_faces(str(image_path), model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face recognition system")
    parser.add_argument(
        "--train", action="store_true", help="process training data and create model"
    )
    parser.add_argument(
        "--validate", action="store_true", help="test validation data"
    )
    parser.add_argument(
        "--test", "-t", help="test specified image"
    )
    parser.add_argument(
        "--model", 
        default="hog", 
        choices=["hog", "cnn"], 
        help="model to use (hog: fast, cnn: more accurate)"
    )

    args = parser.parse_args()

    if args.train:
        print("training started...")
        encode_known_faces(model=args.model)
    
    if args.validate:
        validate(model=args.model)
    
    if args.test:
        print(f"testing: {args.test}")
        recognize_faces(args.test, model=args.model)
    
    if not (args.train or args.validate or args.test):
        print("usage:")
        print("  python detector.py --train        # train model")
        print("  python detector.py --validate     # test validation")
        print("  python detector.py --test image.jpg  # test single image")