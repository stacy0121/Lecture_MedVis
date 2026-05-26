import json
import os
import random
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# 전역 변수로 환자 데이터를 메모리에 유지
GLOBAL_PATIENTS = []

def load_initial_data():
    global GLOBAL_PATIENTS
    try:
        with open('patients.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            GLOBAL_PATIENTS = data.get('patients', [])
    except Exception as e:
        print(f"데이터 로드 실패: {e}")

# 서버 시작 시 데이터 최초 1회 로드
load_initial_data()

def simulate_logical_changes():
    """10초마다 환자의 세부 요인에 기반해 논리적으로 점수와 우선순위를 갱신합니다."""
    global GLOBAL_PATIENTS
    postures = ["Left", "Right", "Supine"]
    action_labels = {"Left": "좌측위", "Right": "우측위", "Supine": "앙와위"}
    
    for p in GLOBAL_PATIENTS:
        max_time = p.get('maxAllowedMinutes', 120)
        
        # 1. 간호사 개입 시뮬레이션 (상태가 위험할수록 개입 확률 급증)
        change_prob = 0.05
        if p.get('status') == 'immediate':
            change_prob = 0.70
        elif p.get('status') == 'caution':
            change_prob = 0.25
            
        if random.random() < change_prob:
            # 🟢 [체위 변경됨] -> 시스템이 직전에 '권장했던 체위'로 변경함
            recommended = p.get('recommendedPosition')
            
            # 만약 권장 체위 데이터가 없거나 현재 체위와 같다면(예외 상황), 남은 체위 중 선택
            if not recommended or recommended == p.get('currentPosition'):
                candidates = [pos for pos in postures if pos != p.get('currentPosition')]
                recommended = random.choice(candidates)
                
            p['currentPosition'] = recommended
            p['elapsedMinutes'] = 0
        else:
            # 🔴 [체위 유지됨] -> 경과 시간 15분씩 증가
            p['elapsedMinutes'] += 15 

        # 2. 히트맵(체위 유지 시간) 논리적 업데이트
        if 'heatmap' not in p or not p['heatmap']:
            p['heatmap'] = [{"p": p['currentPosition'], "m": 15}]
        else:
            last_seg = p['heatmap'][-1]
            if last_seg['p'] == p['currentPosition']:
                last_seg['m'] += 15 # 자세를 유지 중이면 블록의 시간이 진해짐
            else:
                p['heatmap'].append({"p": p['currentPosition'], "m": 15})
                if len(p['heatmap']) > 6:
                    p['heatmap'].pop(0)

        # 3. 위험 요인(factors) 합산을 통한 총점(riskScore) 산출
        if 'riskFactors' in p:
            # 누워있는 시간에 비례하여 '압력 지속(pressureDuration)' 위험도 상승
            time_ratio = p['elapsedMinutes'] / max_time
            p['riskFactors']['pressureDuration'] = min(100, int(time_ratio * 100))
            
            # 각 요인의 가중치를 곱해 종합적인 논리적 위험도 계산
            factors = p['riskFactors']
            calculated_score = (
                factors.get('pressureDuration', 0) * 0.40 +  # 압력 40%
                factors.get('frictionShear', 0) * 0.20 +    # 마찰 20%
                factors.get('moisture', 0) * 0.15 +         # 습기 15%
                factors.get('mobility', 0) * 0.15 +         # 이동성 15%
                factors.get('nutrition', 0) * 0.10          # 영양 10%
            )
            # 최종 점수에 소수점이 없도록 정수 처리
            p['riskScore'] = round(min(100, max(0, calculated_score)))
            
        # 4. 점수에 따른 알림 상태 지정
        score = p.get('riskScore', 0)
        if score >= 70:
            p['status'] = 'immediate'
        elif score >= 40:
            p['status'] = 'caution'
        else:
            p['status'] = 'stable'
            
        # 5. 권장 체위 도출 (히트맵 기준 가장 안 쓴 자세 추천)
        usage = {"Left": 0, "Right": 0, "Supine": 0}
        for seg in p.get('heatmap', []):
            usage[seg.get('p', 'Supine')] += seg.get('m', 1)
            
        cands = [pos for pos in postures if pos != p['currentPosition']]
        p['recommendedPosition'] = min(cands, key=lambda pos: usage.get(pos, 0))
        
        # 6. 행동 지침 텍스트
        label = action_labels.get(p['recommendedPosition'], p['recommendedPosition'])
        if p['status'] == 'immediate':
            p['recommendedAction'] = f"{label}로 즉시 변경"
        elif p['status'] == 'caution':
            p['recommendedAction'] = f"{label}로 변경 준비"
        else:
            p['recommendedAction'] = "상태 재평가"
            
    # 7. 위험 점수(riskScore)를 기준으로 내림차순 정렬하여 순위(priority) 논리적 재할당
    GLOBAL_PATIENTS.sort(key=lambda x: x.get('riskScore', 0), reverse=True)
    for i, p in enumerate(GLOBAL_PATIENTS):
        p['priority'] = i + 1

# --- API 엔드포인트 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/patients', methods=['GET'])
def get_patients():
    simulate_logical_changes()
    return jsonify(GLOBAL_PATIENTS)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)