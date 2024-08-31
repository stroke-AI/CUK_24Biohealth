from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from werkzeug.utils import secure_filename
from result import predict_stroke  # result.py에서 predict_stroke 함수 가져오기
import numpy as np
import cv2
from threading import Timer
from flask_cors import CORS

# Flask 설정
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'supersecretkey'

# 데이터베이스 설정
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 데이터베이스 모델 정의
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    blood_pressure = db.Column(db.String(10), nullable=True)
    cholesterol = db.Column(db.String(10), nullable=True)
    systolic_bp = db.Column(db.Integer, nullable=True)
    diastolic_bp = db.Column(db.Integer, nullable=True)
    total_cholesterol = db.Column(db.Integer, nullable=True)
    smoking = db.Column(db.String(10), nullable=True)

# Flask-Login 설정
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 허용된 파일 확장자 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 업로드 폴더 생성 (존재하지 않을 경우)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# uploads 폴더 초기화
def delete_file_later(file_path, delay):
    def delete_file():
        try:
            os.remove(file_path)
            print(f'File {file_path} deleted successfully (Backup delete)')
        except FileNotFoundError:
            print(f'File {file_path} already deleted.')
        except Exception as e:
            print(f'Error deleting file {file_path}: {e}')
    
    timer = Timer(delay, delete_file)
    timer.start()

# 로비
@app.route('/')
def index():
    return render_template('index.html', logged_in=current_user.is_authenticated)

# 파일 업로드
@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/upload_file', methods=['POST'])
@login_required
def upload_file():
    if 'userPhoto' not in request.files:
        flash('파일이 선택되지 않았습니다.', 'error')
        return redirect(url_for('upload'))
    
    file = request.files['userPhoto']
    
    if file.filename == '':
        flash('선택된 파일이 없습니다.', 'error')
        return redirect(url_for('upload'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 저장된 파일 경로를 predict_stroke 함수에 전달
            result, confidence = predict_stroke(filepath)

            # 이미지 30초 후 삭제 예약
            delete_file_later(filepath, 30)  # 30초 후 이미지 파일 삭제
            print("30초 후 uploads 파일이 초기화됩니다.")

            # 결과를 결과 페이지로 전달
            return render_template('result.html', result=result, confidence=confidence, filename=filename)
        except Exception as e:
            flash(f'처리 중 오류 발생: {str(e)}', 'error')
            return redirect(url_for('upload'))
    else:
        flash('허용되지 않는 파일 형식입니다.', 'error')
        return redirect(url_for('upload'))

    
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 결과
@app.route('/result')
def result():
    # result.html에 예측 결과랑 확신하는 확률 반환
    return render_template('result.html', result='', confidence='')

# 회원가입
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # 사용자 존재 여부 확인
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('이미 존재하는 사용자입니다.', 'error')
            return redirect(url_for('signup'))
        
        # 사용자 추가
        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('회원가입이 완료되었습니다!', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

# 로그인
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            login_user(user)
            flash('로그인 성공!', 'success')
            return redirect(url_for('index'))
        else:
            flash('이메일이나 비밀번호가 잘못되었습니다.', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

# 정보 수정
@app.route('/setting', methods=['GET', 'POST'])
@login_required
def setting():
    user = User.query.filter_by(email=current_user.email).first()
    
    if request.method == 'POST':
        # 나이
        user.age = request.form['age']
        # 성별
        user.gender = request.form['gender']
        # 혈압
        user.blood_pressure = request.form['blood-pressure-option']
        user.systolic_bp = request.form['systolic-bp']
        user.diastolic_bp = request.form['diastolic-bp']
        # 콜레스테롤 수치
        user.cholesterol = request.form['cholesterol-option']
        user.total_cholesterol = request.form['cholesterol']
        # 흡연 여부
        user.smoking = request.form['smoking']

        db.session.commit()
        flash('정보가 수정되었습니다.', 'success')
        return redirect(url_for('index'))
        
    return render_template('setting.html', user=user)

# 로그아웃
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('로그아웃 되었습니다.', 'info')
    return redirect(url_for('index'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # 데이터베이스와 테이블을 생성
    app.run(debug=True)
