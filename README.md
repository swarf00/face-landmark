# 얼굴 검출 및 랜드마크(landmark) 검출기

## backend
* 개발환경
  * 파이썬 3.7(anaconda3-2019.10)
  * django 3.0
  * tensorflow 2.0.0
* 모델
  facial_landmark_SqueezeNet
* 참고
  * from https://github.com/songhengyang/face_landmark_factory.git
  
* 실행방법
``` bash
# cd backend
# pip install -r requirements.txt
# ./manage.py runserver 0.0.0.0:8000
```

## frontend
* 개발환경
  * nodejs v10.15.3
  * nuxt 2.0.0 (vue 2.6.11)
  * element ui

* 실행방법
``` bash
# install dependencies
$ yarn

# serve with hot reload at localhost:3000
$ yarn dev
```

## TODO
[] Solve frontend eslint warning (element ui)
[] Modify frontend ui
[] Flush temp file
