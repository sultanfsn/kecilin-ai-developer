import cv2
import datetime
import sys
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"This video has around {frames} frames and is {round(fps)} fps!")
    print("How many frames you want to save as image?")
    frame_number = int(input())
    
    try:
        os.makedirs(output_dir, exist_ok=False)
        save_dir = output_dir
    except:
        return print("Output directory name already exist!")
    
    i = 1
    counter = 1
    pbar = tqdm(total=frames)
    intervals = int(round(frames)/frame_number)
    while(cap.isOpened()):
        pbar.update(1)

        # Capture frame-by-frame
        ret, frame = cap.read() 
        if ret == True: 
        # Display the resulting frame 
            # if (i%(fps/frame_number) != 1):
            #     i+=1
            #     continue
            # print(f"{i}-{intervals}" )
            if (i%intervals != 1):
                i+=1
                continue
            output_path = os.path.join(save_dir, "{:04d}.jpg".format(counter))
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            # cv2.imshow('Frame', frame) 
            
        # Press Q on keyboard to exit 
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    
        # if flag == False:
        #     break
        # cv2.imshow('test', frame)
        i += 1
        counter+=1
    
    pbar.close()
    print(f"{counter-1} images created!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vid2pic.py [videopath] [outputpath]")
    else:
        video_path = sys.argv[1]
        output_dir = sys.argv[2]
        extract_frames(video_path, output_dir)
