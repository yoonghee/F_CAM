# F_CAM
encore final project (object detect / tracking / web / yolo)

# YOLO_darknet
yolo object detection을 custum data학습하여 진행하기 위해 .weights파일을 만들어 주기 위해 사용 
추가로 video_detection과 다른 os에서도 동작할 수 있도록 다양한 실행파일이 구현되어 있음

# YOLO_mark
yolo_darknet에서 custum data학습을 진행하기 위해 data로 사용할 이미지에 bounding box를 만들어주는데 필요한 툴

# YOLO_v3
video to frame, frame to video, image detection, video detection, 좌표값 extraction등이 YOLO_v3로 구현되었으며
v2값을 사용할 때에는 code 값을 수정해줄 필요가 있음

# location_extraction.py
최종 인식 알고리즘으로 생성한 동영상의 좌표값을 추출하여 아이돌 멤버별로 moving average를 사용하여 tracking algorithm 적용한 좌표값 생성(4K화질)
