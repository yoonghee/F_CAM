# FOCUS_CAM
encore final project (object detect / tracking / web / yolo)

# YOLO_darknet
yolo object detection을 custum data학습하여 진행하기 위해 .weights파일을 만들어 주기 위해 사용 
추가로 video_detection과 다른 os에서도 동작할 수 있도록 다양한 실행파일이 구현되어 있음

# YOLO_mark
yolo_darknet에서 custum data학습을 진행하기 위해 data로 사용할 이미지에 bounding box를 만들어주는데 필요한 툴

# YOLO_v3
video to frame, frame to video, image detection, video detection, 좌표값 extraction등이 YOLO_v3로 구현되었으며
v2값을 사용할 때에는 code 값을 수정해줄 필요가 있음

# Final_web
최종적으로 마무리된 웹 구현 파일, mp3/mp4 파일등 큰 파일은 제외되어 있음

# Location_jupyter
최종 인식 알고리즘으로 생성한 동영상의 좌표값을 추출하여 결측치를 처리하고
아이돌 멤버별로 moving average를 사용하여 tracking algorithm 적용한 좌표값 생성(1K화질)
브라우저 로딩문제로 동영상은 최대 1k화질을 지원하며, canvas tag 호환성 문제로 chrome대신 edge브라우저 

# location_extraction.py
최종 인식 알고리즘으로 생성한 동영상의 좌표값을 추출하여 아이돌 멤버별로 moving average를 사용하여 tracking algorithm 적용한 좌표값 생성(4K화질)
