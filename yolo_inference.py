from ultralytics import YOLO

model = YOLO('models/best.pt')

res = model.predict('input_videos/08fd33_4.mp4', save=True)
print(res[0])  # first frame
print("------------------")
for box in res[0].boxes:
    print(box)