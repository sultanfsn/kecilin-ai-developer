import csv
import cv2
from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos, LoadStreams
import numpy as np

# do this once to convert YOLO to OpenVino YOLO onnx (faster)
model = YOLO('models/solution.pt')
# model.export(format='openvino') 

# model = YOLO('models/solution_openvino_model', task='detect')

is_stream = False # change to true if streaming

video_name = "03.mp4"

source = f'videos/{video_name}' # change this to desired input video

video = cv2.VideoCapture(source)
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
spf = 1/round(fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result/demo.mp4',fourcc, fps, (video_width, video_height))


frame_0 = None

region_coordinates = []

regions = []

if is_stream:
    dataset = LoadStreams(
            source,
            imgsz=640,
            vid_stride=1,
            buffer=False
        )
else:
    dataset = LoadImagesAndVideos(
                source,
                vid_stride=1,
            )
    
battle_status = 0
battle_duration = 0
battle_results = {
    "winner":None,
    "loser":None,
    "reason":None,
    "duration": 0
}

prev_frame = None

count = 0

while True:
    
    ret, frame = video.read()

    if ret:
        frame_0 = frame
        video.release()
        break

while True:

    img = frame.copy()

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Draw a circle on the image
            cv2.circle(frame, (x, y), 3, (0,255,0), -1, cv2.LINE_AA)
            region_coordinates.append([x, y])
            cv2.imshow("draw region", img)

    cv2.namedWindow("draw region")

    cv2.setMouseCallback("draw region", click_event, {"img": img})

    cv2.imshow("draw region", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("e"):
        if len(region_coordinates) != 0:
            regions.append(region_coordinates)
            region_coordinates = []
        continue

    if key == ord("q"):
        if len(region_coordinates) != 0:
            regions.append(region_coordinates)
            region_coordinates = []
        cv2.destroyAllWindows()
        break

for path, images, info, in dataset:
    is_spinning = []
    is_stopped = []

    frame = images[0]

    pts = np.array(regions[0],
               np.int32)
 
    pts = pts.reshape((-1, 1, 2))
    # print(regions[0])
    image = cv2.polylines(frame, [pts], 
                      True, (255, 0, 0), 2)

    if count == 0:
        prev_frame = frame

    results = model.predict(source=frame, conf=0.56)
    if battle_status == 0 and len(results[0].boxes) == 2:
        battle_status = 1

    if battle_status == 1 and count != 0:
        battle_duration+=spf
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        for bbox in results[0].boxes:
            x1, y1, x2, y2 = bbox.xyxy[0].tolist()
            centroid = (
                int((x1 + x2) * 0.5),
                int((y1 + y2) * 0.5),
            )
            # cv2.circle(frame, (centroid[0], centroid[1]), 3, (0,255,0), -1, cv2.LINE_AA)
            x_min = int(min(x1, max(0, x1-5)))
            y_min = int(min(y1, max(0, y1-5)))
            x_max = int(x2 + 5)
            y_max = int(y2 + 5)

            distance = cv2.pointPolygonTest(
                        np.array(regions[0]), centroid, measureDist=False
                    )
            
            if distance < 0:
                battle_status = 2
                is_stopped.append((x_min, y_min, x_max, y_max))
                cropped_image = frame[y_min:y_max, x_min:x_max]
                loser_write_path = f"result/loser.jpg"
                battle_results["loser"] = loser_write_path
                battle_results["reason"] = "out_of_bound"
                cv2.imwrite(loser_write_path, cropped_image)
                continue

            movement = np.mean(mag[y_min:y_max, x_min:x_max])
            print(movement)

            if movement < 0.60 and len(is_stopped)==0:
                battle_status = 2
                is_stopped.append((x_min, y_min, x_max, y_max))
                cropped_image = frame[y_min:y_max, x_min:x_max]
                loser_write_path = f"result/loser.jpg"
                battle_results["loser"] = loser_write_path
                battle_results["reason"] = "TKO"
                cv2.imwrite(loser_write_path, cropped_image)
                continue

            is_spinning.append((x_min, y_min, x_max, y_max))
            if len(is_stopped) != 0:
                cropped_image = frame[is_spinning[0][1]:is_spinning[0][3], is_spinning[0][0]:is_spinning[0][2]]
                winner_write_path = f"result/winner.jpg"
                battle_results["winner"] = winner_write_path
                cv2.imwrite(winner_write_path, cropped_image)
                break
        if len(is_stopped) != 0:
            cropped_image = frame[is_spinning[0][1]:is_spinning[0][3], is_spinning[0][0]:is_spinning[0][2]]
            winner_write_path = f"result/winner.jpg"
            battle_results["winner"] = winner_write_path
            cv2.imwrite(winner_write_path, cropped_image)
            break

    print(battle_duration)
    
    annotated_frame = results[0].plot()

    prev_frame = frame
    count+=1

    # Display the annotated frame
    cv2.imshow("Bakuten Shoot Corp.", annotated_frame)

    out.write(annotated_frame) 


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        break

battle_results["duration"] = battle_duration

filename = "result/battle_result.csv"

battle_data = [{
    'video_name':video_name,
    'battle_duration':battle_results["duration"],
    'winner':battle_results["winner"],
    'loser':battle_results["loser"],
    'reason':battle_results["reason"]
}]

headers = ['video_name', 'battle_duration', 'winner', 'loser', 'reason']

with open(filename, 'w') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    # writing headers (field names)
    writer.writeheader()

    # writing data rows
    writer.writerows(battle_data)









