import json
import os
import random
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# 전역 변수로 환자 데이터를 메모리에 유지합니다 (시뮬레이션용)
GLOBAL_PATIENTS = []

def load_initial_data():
    global GLOBAL_PATIENTS
    try:
        with open('patients.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            GLOBAL_PATIENTS = data.get('patients', [])
    except Exception as e:
        print(f"데이터 로드 실패: {e}")

# 서버 시작 시 데이터 로드
load_initial_data()

def simulate_10_seconds():
    """10초마다 환자의 상태를 무작위로 변화시키고 추천을 다시 계산합니다."""
    global GLOBAL_PATIENTS
    postures = ["Left", "Right", "Supine"]
    action_labels = {"Left": "좌측위", "Right": "우측위", "Supine": "앙와위"}
    
    for p in GLOBAL_PATIENTS:
        # 1. 경과 시간 증가 (시각적 효과를 위해 10분씩 팍팍 올림)
        p['elapsedMinutes'] += 10
        
        # 2. 위험 점수를 무작위로 요동치게 만듦 (-15점에서 +15점 사이)
        change = random.randint(-15, 15)
        p['riskScore'] = max(0, min(100, p.get('riskScore', 0) + change))
        
        # 3. 점수에 따라 알림 상태 변경
        if p['riskScore'] >= 70:
            p['status'] = 'immediate'
        elif p['riskScore'] >= 40:
            p['status'] = 'caution'
        else:
            p['status'] = 'stable'
            
        # 4. 체위 이력(히트맵) 무작위 추가
        if 'heatmap' not in p:
            p['heatmap'] = []
        p['heatmap'].append({"p": p['currentPosition'], "m": 10})
        if len(p['heatmap']) > 10: # 최근 10개만 유지
            p['heatmap'].pop(0)
            
        # 5. 새로운 권장 체위 계산 (가장 안 쓴 체위 추천)
        usage_counts = {"Left": 0, "Right": 0, "Supine": 0}
        for seg in p['heatmap']:
            usage_counts[seg.get('p', 'Supine')] += seg.get('m', 1)
            
        candidates = [pos for pos in postures if pos != p['currentPosition']]
        p['recommendedPosition'] = min(candidates, key=lambda pos: usage_counts.get(pos, 0))
        
        # 6. 행동 지침 텍스트 업데이트
        label = action_labels.get(p['recommendedPosition'], p['recommendedPosition'])
        if p['status'] == 'immediate':
            p['recommendedAction'] = f"{label}로 즉시 변경"
        elif p['status'] == 'caution':
            p['recommendedAction'] = f"{label}로 변경 준비"
        else:
            p['recommendedAction'] = "상태 재평가"
            
    # 7. 전체 환자 우선순위 다시 정렬 (점수가 높은 순)
    GLOBAL_PATIENTS.sort(key=lambda x: x['riskScore'], reverse=True)
    for i, p in enumerate(GLOBAL_PATIENTS):
        p['priority'] = i + 1

# --- API 엔드포인트 ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/patients', methods=['GET'])
def get_patients():
    # 프론트엔드가 이 주소를 부를 때마다 시뮬레이션 가동!
    simulate_10_seconds()
    return jsonify(GLOBAL_PATIENTS)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)