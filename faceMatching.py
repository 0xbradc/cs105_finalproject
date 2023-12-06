import os
from deepface import DeepFace # https://github.com/serengil/deepface
import urllib.parse

current_dir = os.path.dirname(os.path.abspath(__file__))
student_pics_dir = os.path.join(current_dir, "picture_database")
instagram_pics_dir = os.path.join(current_dir, "picture_samples")
MODELS = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]
CURRENT_MODEL = MODELS[0] # First model by default

def getUserInput():
    global CURRENT_MODEL, MODELS
    user_input = input("Enter a number: ")
    try:
        number = int(user_input)
        if 0 <= number <= len(MODELS):
            CURRENT_MODEL = MODELS[number]
        else:
            print(f"The number is not between 0 and {len(MODELS)}. Proceeding with the default model...")
    except ValueError:
        print("Invalid input. Proceeding with the default model...")

def runMatching():
    for filename in os.listdir(instagram_pics_dir):
        if os.path.isfile(os.path.join(instagram_pics_dir, filename)):
            try:
                match = DeepFace.find(
                    img_path = os.path.join(instagram_pics_dir, filename),
                    db_path = student_pics_dir,
                    model_name=CURRENT_MODEL,
                    enforce_detection = False,
                    silent=False
                )
            except KeyboardInterrupt:
                print("Ctrl+C pressed. Exiting...")
                exit()
            except:
                continue

            # Print top 5 matches
            print("\nInstagram Picture: ", "\"file://" + urllib.parse.quote(os.path.join(instagram_pics_dir, filename)) + "\"")
            for i in range(5):
                try:
                    print("Match #" + str(i) + ": ", "\"file://" + urllib.parse.quote(match[0].iloc[i]["identity"]) + "\"")
                    print("\tWith Cosine: " + match[0].iloc[i][CURRENT_MODEL + "_cosine"].astype(str))
                except:
                    continue
            print("\n")

            # Print just the top match
            # print("\nInstagram Picture: ", "\"file://" + urllib.parse.quote(os.path.join(instagram_pics_dir, filename)) + "\"")
            # try:
            #     print("Closest Match: ", "\"file://" + urllib.parse.quote(match[0].iloc[0]["identity"]) + "\"")
            #     print("\tWith Cosine: " + match[0].iloc[0][CURRENT_MODEL + "_cosine"].astype(str))
            # except:
            #     continue
            # print("\n")


if __name__ == "__main__":
    getUserInput()
    runMatching()