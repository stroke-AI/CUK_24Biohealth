# Self-diagnosis web
With stroke prediction model according to degree of facial asymmetry

*For The Catholic Univ. of Korea 2024 Bio Health Contest*

---
### 1. 개발 환경

- Anaconda 24.7.1
- Python 3.8.18

---
### 2. 사용 패키지

- numpy: 1.24.3
- scikit-learn: 1.3.0
- seaborn: 0.13.2
- matplotlib: 3.7.2
- tensorflow: 2.10.0
- flask: 3.0.3
- cmake: 3.26.4
- sqlalchemy: 2.0.30
- opencv-python: 4.10.0.84
- flask-cors: 5.0.0
- flask-login: 0.6.3
- flask-migrate: 4.0.7
- flask-sqlalchemy: 3.1.1
- flask-wtf: 1.2.1
- dlib: 19.24.6

---
### 3. 설치 방법

1. 아니콘다 설치

    본인의 개발 환경에 맞추어 설치하면 된다.

    https://www.anaconda.com/download

2. 비주얼 스튜디오 코드 설치
    
    본인의 개발 환경에 맞추어 설치하면 된다.
  
    https://code.visualstudio.com/download

3. 패키지 설치

    Ananoda Prompt를 열고 다음을 순서대로 입력한다.
    
    ```
    conda create -n <가상 환경 이름> python=3.8
    ```
    ```
    conda activate <가상 환경 이름>
    ```
    ```
    conda install numpy scikit-learn seaborn matplotlib tensorflow flask cmake sqlalchemy
    ```
    ```
    pip install opencv-python flask-cors flask-login flask-migrate flask-sqlalchemy flask-wtf dlib
    ```
    ###### dlib은 C++ 라이브러리로 python에 설치하기 어려울 수 있다. 만약 설치 시에 오류가 발생한다면, 다음 링크를 참고하여 설치를 진행하면 된다. [링크](https://updaun.tistory.com/entry/python-python-37-dlib-install-error)

    오류가 나지 않았다면 가상 환경 구축이 완료되었다.
    
    깃허브에 있는 모든 파일을 다운로드한 후,  압축을 풀어 파일을 준비한다.

    비주얼 스튜디오 코드에 들어가 해당 파일을 연 뒤, F1을 눌러 파이썬 인터프리터 
    설정에서 현재까지 만든 가상 환경을 실행시킨다.

4. 실행

     파일 경로가 다음과 같다고 가정하고 진행하겠다. 모든 파일 경로를 상대 경로로써 설정해 두었으니 만일 경로 오류가 난다면 코드들을 열어 파일 경로를 본인 경로에 맞추어 바꾸어 주면 된다. 설명은 생략하겠다.

    1. 모델 저장
    
        비주얼 스튜디오 코드에서 <u>Ctrl + `</u> 를 눌러 터미널을 열고 다음을 입력하여 모델 훈련 코드가 있는 파일로 넘어간다.
        ```
        cd BaseFile
        ```
        이후, BaseFile을 열어 CNN_train.py를 실행시킨다. 이때 cnn_model.h5 파일이 BaseFile 내부에 생길 것인데, 이게 예측 모델이다.

    2. 로컬 서버 접속
   
        다시 터미널을 열어 다음을 입력해 상위 폴더로 이동한다.
        
        ```
         cd ..
        ```
          이후, app.py를 실행시키면 로컬 서버 주소가 비주얼 스튜디오 코드의 터미널에 뜰 것이고 이를 접속하여 로컬 서버를 이용하면 된다.
