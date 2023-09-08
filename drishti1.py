import os
import subprocess
import speech_recognition as sr
from gtts import gTTS
import requests
from PIL import Image
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection,AutoProcessor , BlipForConditionalGeneration , BlipProcessor, BlipForImageTextRetrieval,BlipForQuestionAnswering
import torch
import pygame
from collections import Counter
import pytesseract
import random
import sys
import time
import winsound
import socket


def transcribe_audio_from_microphone():
    sound_file_path = 'output16845854.mp3'

# Play the sound
    winsound.PlaySound(sound_file_path, winsound.SND_FILENAME)
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something!")
        say="welcome to drishti please say something"
        language = 'en'

        # Create a gTTS (Google Text-to-Speech) object
        tts = gTTS(text=say, lang=language, slow=False)
        ras=random.randint(3000, 30000000)
        cccc = f"output{ras}.mp3"
        # Save the generated audio to a file
        tts.save(cccc)

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load(cccc)

        # Play the audio using pygame
        # pygame.mixer.music.play()
        audio = recognizer.listen(source, timeout=5.0)
    try:
        transcription = recognizer.recognize_google(audio)
        print("You said: " + transcription)
        return transcription
    except sr.UnknownValueError:
        transcription = "describe"
        print(transcription)
        return transcription

# def capture_image_from_webcam():
#     # Initialize the webcam
#     cap = cv2.VideoCapture(0)
    
#     # Check if the webcam opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         return None
    
#     # Capture a frame from the webcam
#     ret, frame = cap.read()
    
#     # Release the webcam
#     cap.release()
    
#     if ret:
#         # Save the captured frame as an image
#         cv2.imwrite("captured_image.jpg", frame)
#         return "captured_image.jpg"
#     else:
#         print("Error: Could not capture image from webcam")
#         return None


# def capture_image_from_webcam(url):
#     # Open the video capture object for the IP camera feed
#     cap = cv2.VideoCapture(url)

#     # Check if the IP camera feed opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open IP camera feed")
#         return None

#     # Capture a frame from the IP camera feed
#     ret, frame = cap.read()

#     # Release the video capture object
#     cap.release()

#     if ret:
#         # Save the captured frame as an image
#         cv2.imwrite("captured_image.jpg", frame)
#         return "captured_image.jpg"
#     else:
#         print("Error: Could not capture image from IP camera feed")
#         return None

def capture_last_frame_from_ip_camera(url, timeout=5):
    # Open the video capture object for the IP camera feed
    cap = cv2.VideoCapture(url)

    # Check if the IP camera feed opened successfully
    if not cap.isOpened():
        print("Error: Could not open IP camera feed")
        return None

    # Initialize variables to store the last captured frame and time
    last_frame = None
    start_time = cv2.getTickCount()

    while True:
        # Capture a frame from the IP camera feed
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from the camera feed")
            break

        # Update the last captured frame
        last_frame = frame

        # Calculate the elapsed time
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

        # If the elapsed time exceeds the timeout, exit the loop
        if elapsed_time >= timeout:
            break

    # Release the video capture object
    cap.release()

    if last_frame is not None:
        # Save the last captured frame as an image
        cv2.imwrite("captured_image.jpg", last_frame)
        return "captured_image.jpg"
    else:
        print("Error: Could not capture image from IP camera feed")
        return None

# URL of the IP camera feed
# url = 'http://shubs:shubs@192.168.1.65:4747/video'

# # Capture an image from the IP camera and save it
# image_path = capture_image_from_webcam(url)

# if image_path:
#     print(f"Image captured from IP camera and saved as {image_path}")


def ocr(image_path):
    image = Image.open(image_path)

    # Perform OCR on the image
    text = pytesseract.image_to_string(image)

    # Print the extracted text
    print(text)

    if text:
    # Language in which you want to convert
        language = 'en'

        # Create a gTTS (Google Text-to-Speech) object
        tts = gTTS(text=text, lang=language, slow=False)

        # Save the generated audio to a file
        ras=random.randint(1000, 10000000)
        ccc = f"output{ras}.mp3"
        tts.save(ccc)

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load(ccc)

        # Play the audio using pygame
        pygame.mixer.music.play()
        time.sleep(15)
        # # Keep the script running to allow audio to play
        # input("Press Enter to exit...")
        sound_file_path = 'output16845854.mp3'

# Play the sound
        winsound.PlaySound(sound_file_path, winsound.SND_FILENAME)

def alert(image_path):
    image = Image.open(image_path)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    detected_labels = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        detected_labels.append(label_name)

        print(
            f"Detected {label_name} with confidence {round(score.item(), 3)} at location {box}"
        )

    # Count occurrences of each label
    label_counts = Counter(detected_labels)

    # Create the formatted detected_labels_text
    detected_labels_text1 = "There are "+", ".join(f"{count} {label}" for label, count in label_counts.items())
    print(f"Detected Labels Text: {detected_labels_text1}")

    language = 'en'

    # Create a gTTS (Google Text-to-Speech) object
    tts = gTTS(text=detected_labels_text1, lang=language, slow=False)
    ras=random.randint(2000, 20000000)
    cc = f"output{ras}.mp3"
    # Save the generated audio to a file
    tts.save(cc)

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(cc)

    # Play the audio using pygame
    pygame.mixer.music.play()
    time.sleep(10)

    # # Keep the script running to allow audio to play
    # input("Press Enter to exit...")


def sas(image_path):
    image = Image.open(image_path)
    # image = Image.open("captured_image.jpg").convert("RGB")

    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-flickr")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    # model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-flickr").to("cuda")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text1="a photography of"
    # inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    inputs = processor(image, text1, return_tensors="pt").to("cuda")


    # generated_ids = model.generate(**inputs, max_new_tokens=20)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    out = model.generate(**inputs)
    generated_text=processor.decode(out[0], skip_special_tokens=True)
    print(generated_text)
    language = 'en'

    # Create a gTTS (Google Text-to-Speech) object
    tts = gTTS(text=generated_text, lang=language, slow=False)
    ras=random.randint(4000, 40000000)
    bb = f"output{ras}.mp3"
    # Save the generated audio to a file
    tts.save(bb)

    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(bb)

    # Play the audio using pygame
    pygame.mixer.music.play()
    time.sleep(5)
#     sound_file_path = 'output16845854.mp3'

# # Play the sound
#     winsound.PlaySound(sound_file_path, winsound.SND_FILENAME)
    


def get_prompt():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something!")
        say="please ask the question of scenario"
        language = 'en'

        # Create a gTTS (Google Text-to-Speech) object
        tts = gTTS(text=say, lang=language, slow=False)
        ras=random.randint(9000, 90000000)
        cccc = f"output{ras}.mp3"
        # Save the generated audio to a file
        tts.save(cccc)

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load(cccc)

        # Play the audio using pygame
        pygame.mixer.music.play()
        audio = recognizer.listen(source, timeout=10.0)
    try:
        transcriptionm = recognizer.recognize_google(audio)
        print("You said: " + transcriptionm)
        return transcriptionm
    except sr.UnknownValueError:
        transcription = "Could not transcribe audio"
        print(transcriptionm)
        return transcriptionm    


def prompts():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something!")
        sound_file_path = 'output16845854.mp3'

        # Play the sound
        winsound.PlaySound(sound_file_path, winsound.SND_FILENAME)

        
        audio = recognizer.listen(source, timeout=5.0)
    try:
        transcription = recognizer.recognize_google(audio)
        print("You said: " + transcription)
        return transcription
    except sr.UnknownValueError:
        transcription = None
        print(transcription)
        language = 'en'
        says="please ask scenario questions"
        # Create a gTTS (Google Text-to-Speech) object
        tts = gTTS(text=says, lang=language, slow=False)
        ras=random.randint(7000, 70000000)
        ccccc = f"output{ras}.mp3"
        # Save the generated audio to a file
        tts.save(ccccc)

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load(ccccc)

        # Play the audio using pygame
        pygame.mixer.music.play()
        return transcription
    

def pas(image_path): 
    # while True:
    while True:
        prompt = prompts()
        if prompt=="exit":
            break
        elif prompt is not None:
       
            print("moving forward with query as: "+prompt)    

            # image = Image.open(image_path)
            image = Image.open("captured_image.jpg").convert("RGB")

            # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

            # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
            # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")


            # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            # processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-flickr")
            # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            # prompt = "Question: What is a dinosaur holding? Answer:"

            inputs = processor(image, prompt, return_tensors="pt").to("cuda")
            # inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

            # generated_ids = model.generate(**inputs, max_new_tokens=10)
            # generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            # print(generated_texts) 
            out = model.generate(**inputs)
            generated_text=processor.decode(out[0], skip_special_tokens=True)
            print(generated_text)             
            language = 'en'

            # Create a gTTS (Google Text-to-Speech) object
            tts = gTTS(text=generated_text, lang=language, slow=False)
            ras=random.randint(7000, 70000000)
            dd = f"output{ras}.mp3"
            # Save the generated audio to a file
            tts.save(dd)

            # Initialize pygame mixer
            pygame.mixer.init()

            # Load the audio file
            pygame.mixer.music.load(dd)

            # Play the audio using pygame
            pygame.mixer.music.play() 
            time.sleep(5) 
#             sound_file_path = 'output16845854.mp3'

# # Play the sound
#             winsound.PlaySound(sound_file_path, winsound.SND_FILENAME)


#for droidcam
def get_local_ip_address():
    try:
        # Create a socket object to get the local host name
        host_name = socket.gethostname()

        # Get the local IP address associated with the host name
        local_ip = socket.gethostbyname(host_name)

        return local_ip
    except Exception as e:
        print(f"Error: {e}")
        return None

# Get and print the local IP address
local_ip_address = get_local_ip_address()
if local_ip_address:
    print(f"Local IP Address: {local_ip_address}")
else:
    print("Failed to retrieve the local IP address.")

        
    


#welcome message
language = 'en'
genr="Welcome to Drishti"
# Create a gTTS (Google Text-to-Speech) object
tts = gTTS(text=genr, lang=language, slow=False)
ras=random.randint(7000, 70000000)
dd = f"output{ras}.mp3"
# Save the generated audio to a file
tts.save(dd)

# Initialize pygame mixer
pygame.mixer.init()

# Load the audio file
pygame.mixer.music.load(dd)

# Play the audio using pygame
pygame.mixer.music.play() 
time.sleep(3)

# Main loop for command recognition
url = 'http://shubs:shubs@192.168.1.65:4747/video'
# url = f'http://shubs:shubs@{local_ip_address}:4747/video'

while True:
    #beep
    #timeout
    command = transcribe_audio_from_microphone()
    
    # if command is None:
    #     image_path = capture_image_from_webcam()
    #     if image_path is not None:
    #         sas(image_path)

    if command == "exit":
        bye = "exiting drishti goodbye"
        language = 'en'

        # Create a gTTS (Google Text-to-Speech) object
        tts = gTTS(text=bye, lang=language, slow=False)
        ras = random.randint(5000, 50000000)
        cc = f"output{ras}.mp3"
        # Save the generated audio to a file
        tts.save(cc)

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load(cc)

        # Play the audio using pygame
        pygame.mixer.music.play()

        # Add a delay to allow time for the message to be spoken
        time.sleep(5)
        break
    elif command == "ocr":
        image_path = capture_last_frame_from_ip_camera(url, timeout=5)
        if image_path is not None:
            ocr(image_path)
    elif command == "alert":
        image_path = capture_last_frame_from_ip_camera(url, timeout=5)
        if image_path is not None:
            alert(image_path)
    elif command == "describe":
        image_path = capture_last_frame_from_ip_camera(url, timeout=5)
        if image_path is not None:
            sas(image_path)            
    elif command == "question":
        print("executing vqa model")
        image_path = capture_last_frame_from_ip_camera(url, timeout=5)
        if image_path is not None:
            pas(image_path)            
    # elif command=="exit":
    #     sys.exit()


# End of the program
print("Program has ended.")
