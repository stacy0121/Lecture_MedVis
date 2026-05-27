import json
import os
import random
from flask import Flask, jsonify, render_template

app = Flask(__name__)

GLOBAL_PATIENTS = []

def load_initial_data():
    global GLOBAL_PATIENTS
    try:
        with open('patients.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            GLOBAL_PATIENTS = data.get('patients', [])
    except Exception as e:
        print(f"데이터 로드 실패: {e}")

load_initial_data()

POSTURE_SPOT_WEIGHTS = {
    "Supine": {"occiput": 0.8, "scapula_l": 0.5, "scapula_r": 0.5, "sacrum": 1.0, "heel_l": 0.6, "heel_r": 0.6},
    "Left":   {"occiput": 0.2, "scapula_l": 0.9, "scapula_r": 0.0, "sacrum": 0.4, "heel_l": 0.9, "heel_r": 0.1},
    "Right":  {"occiput": 0.2, "scapula_l": 0.0, "scapula_r": 0.9, "sacrum": 0.4, "heel_l": 0.1, "heel_r": 0.9}
}

def simulate_logical_changes():
    global GLOBAL_PATIENTS
    postures = ["Left", "Right", "Supine"]
    action_labels = {"Left": "좌측위", "Right": "우측위", "Supine": "앙와위"}
    
    for p in GLOBAL_PATIENTS:
        max_time = p.get('maxAllowedMinutes', 120)
        
        # 1. 🔥 실제 임상 프로토콜 기반 체위 변경 로직 🔥
        needs_change = False
        
        # 조건 A: 체위 유지 시간이 권장 시간(120분)에 도달했거나 초과한 경우 (정규 체위 변경)
        if p.get('elapsedMinutes', 0) >= max_time:
            needs_change = True
        # 조건 B: 120분이 안 되었더라도, 상태가 '즉시 조치(Immediate)' 위험 수준인 경우 (응급 개입)
        elif p.get('status') == 'immediate':
            needs_change = True
            
        if needs_change:
            # 🟢 [체위 변경] 간호사가 권장 체위로 환자의 자세를 변경함
            recommended = p.get('recommendedPosition')
            if not recommended or recommended == p.get('currentPosition'):
                candidates = [pos for pos in postures if pos != p.get('currentPosition')]
                recommended = random.choice(candidates)
                
            p['currentPosition'] = recommended
            p['elapsedMinutes'] = 0 # 체위 변경 후 경과 시간 0으로 초기화
        else:
            # 🔴 [체위 유지] 아직 120분이 안 되었고 위험하지 않으므로 15분 경과
            p['elapsedMinutes'] += 15 

        # 2. 히트맵 업데이트
        if 'heatmap' not in p or not p['heatmap']:
            p['heatmap'] = [{"p": p['currentPosition'], "m": 15}]
        else:
            last_seg = p['heatmap'][-1]
            if last_seg['p'] == p['currentPosition']:
                last_seg['m'] += 15 
            else:
                p['heatmap'].append({"p": p['currentPosition'], "m": 15})
                if len(p['heatmap']) > 6:
                    p['heatmap'].pop(0)

        # 3. 위험 요인(factors) 합산을 통한 총점 산출
        if 'riskFactors' in p:
            time_ratio = p['elapsedMinutes'] / max_time
            p['riskFactors']['pressureDuration'] = min(100, int(time_ratio * 100))
            
            factors = p['riskFactors']
            calculated_score = (
                factors.get('pressureDuration', 0) * 0.40 +
                factors.get('frictionShear', 0) * 0.20 +
                factors.get('moisture', 0) * 0.15 +
                factors.get('mobility', 0) * 0.15 +
                factors.get('nutrition', 0) * 0.10
            )
            p['riskScore'] = round(min(100, max(0, calculated_score)))
            
        # 4. 신체 부위별 압력 위험도(spotRisks) 논리적 갱신
        if 'spotRisks' not in p:
            p['spotRisks'] = {"occiput": 0, "scapula_l": 0, "scapula_r": 0, "sacrum": 0, "heel_l": 0, "heel_r": 0}
            
        current_pos = p['currentPosition']
        weights = POSTURE_SPOT_WEIGHTS.get(current_pos, POSTURE_SPOT_WEIGHTS["Supine"])
        pressure_duration = p.get('riskFactors', {}).get('pressureDuration', 0)
        
        braden = p.get('bradenScore', 15)
        vulnerability_base = max(0, (23 - braden) * 1.5)
        
        for spot, weight in weights.items():
            target_risk = min(100, (pressure_duration * weight) + vulnerability_base)
            current_val = p['spotRisks'].get(spot, 0)
            p['spotRisks'][spot] = round(current_val + (target_risk - current_val) * 0.3)

        # 5. 점수에 따른 알림 상태
        score = p.get('riskScore', 0)
        if score >= 70:
            p['status'] = 'immediate'
        elif score >= 40:
            p['status'] = 'caution'
        else:
            p['status'] = 'stable'
            
        # 6. 권장 체위 도출
        usage = {"Left": 0, "Right": 0, "Supine": 0}
        for seg in p.get('heatmap', []):
            usage[seg.get('p', 'Supine')] += seg.get('m', 1)
            
        cands = [pos for pos in postures if pos != p['currentPosition']]
        p['recommendedPosition'] = min(cands, key=lambda pos: usage.get(pos, 0))
        
        # 7. 행동 지침 텍스트
        label = action_labels.get(p['recommendedPosition'], p['recommendedPosition'])
        if p['status'] == 'immediate':
            p['recommendedAction'] = f"{label}로 즉시 변경"
        elif p['status'] == 'caution':
            p['recommendedAction'] = f"{label}로 변경 준비"
        else:
            p['recommendedAction'] = "상태 재평가"
            
    # 8. 정렬
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