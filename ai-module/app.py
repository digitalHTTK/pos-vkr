from flask import Flask, request
from app_service import AppService
import json

app = Flask(__name__)
appService = AppService()


@app.route('/')
def home():
    return "Endpoints base: /api/v0/"

@app.route('/api/v0/debug')
def tasks():
    return appService.debug()

@app.route('/api/v0/train/svm')
def debug():
    return appService.debug()

@app.route('/api/v0/task', methods=['POST'])
def create_task():
    request_data = request.get_json()
    task = request_data['task']
    return appService.create_task(task)
