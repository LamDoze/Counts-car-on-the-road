import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk

min_contour_width = 50
min_contour_height = 50
offset = 10
line_height = 550
matches = []
cars = 0

def get_centroid(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

# Open the video source
cap = cv2.VideoCapture('../Videos/traffic.mp4')

def process_frame():
    global cars, frame1, frame2

    ret, frame2 = cap.read()
    if not ret:
        cap.release()
        window.destroy()
        return

    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    _, th = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(th, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    closing = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)

    filled_closing = closing.copy()
    h, w = filled_closing.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(filled_closing, mask, (0, 0), 255)
    filled_closing = cv2.bitwise_not(filled_closing)
    filled_closing = cv2.bitwise_or(closing, filled_closing)

    contours, _ = cv2.findContours(filled_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_contour_width and h >= min_contour_height and w * h >= 2500:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
            centroid = get_centroid(x, y, w, h)
            matches.append(centroid)
            cx, cy = centroid
            if (line_height - offset) < cy < (line_height + offset):
                cars += 1
                print(f"Số lượng xe đã được đếm: {cars}") 
                if (cx, cy) in matches:
                    matches.remove((cx, cy))


    cv2.line(frame1, (100, line_height), (900, line_height), (0, 255, 0), 2)
    cv2.putText(frame1, f"Tong so xe: {cars}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    resized_frame1 = cv2.resize(frame1, (800, 600))  # Tăng kích thước khung hình
    resized_frame2 = cv2.resize(filled_closing, (800, 600))

    img1 = Image.fromarray(cv2.cvtColor(resized_frame1, cv2.COLOR_BGR2RGB))
    imgtk1 = ImageTk.PhotoImage(image=img1)
    label1.imgtk = imgtk1
    label1.configure(image=imgtk1)

    img2 = Image.fromarray(resized_frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    label2.imgtk = imgtk2
    label2.configure(image=imgtk2)

    frame1 = frame2
    window.after(10, process_frame)

window = tk.Tk()
window.title("Dem so luong oto luu thong")
window.geometry("1700x700")  

# Create frames for side-by-side video display
frame_left = Frame(window, width=800, height=600, bg="black")  
frame_left.pack(side="left", padx=20, pady=20)

frame_right = Frame(window, width=800, height=600, bg="black")  
frame_right.pack(side="right", padx=20, pady=20)

label1 = Label(frame_left)
label1.pack()
label2 = Label(frame_right)
label2.pack()

ret, frame1 = cap.read()
if ret:
    process_frame()

window.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), window.destroy()))
window.mainloop()
