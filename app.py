import json
import os
import time
import cv2
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

GLOBAL_PATIENTS = []
# 1초마다 들어오는 YOLO 체위 분류 결과를 환자별로 10초 동안 모아두는 버퍼
POSTURE_VOTES = {}

MODEL_PATH = "models/best.pt"
VIDEO_DIR = "static/videos"

model = YOLO(MODEL_PATH)
SERVER_START_TIME = time.time()

POSTURE_SPOT_WEIGHTS = {
    "Supine": {"occiput": 0.8, "scapula_l": 0.5, "scapula_r": 0.5, "sacrum": 1.0, "heel_l": 0.6, "heel_r": 0.6},
    "Left":   {"occiput": 0.2, "scapula_l": 0.9, "scapula_r": 0.0, "sacrum": 0.4, "heel_l": 0.9, "heel_r": 0.1},
    "Right":  {"occiput": 0.2, "scapula_l": 0.0, "scapula_r": 0.9, "sacrum": 0.4, "heel_l": 0.1, "heel_r": 0.9},
}

ACTION_LABELS = {"Left": "좌측위", "Right": "우측위", "Supine": "앙와위"}

CLASS_NAME_MAP = {
    "left": "Left", "Left": "Left", "좌측위": "Left",
    "right": "Right", "Right": "Right", "우측위": "Right",
    "supine": "Supine", "Supine": "Supine", "앙와위": "Supine",
}

POSTURE_KEYS = ["Left", "Right", "Supine"]


def empty_posture_probabilities():
    return {key: 0.0 for key in POSTURE_KEYS}


def extract_posture_probabilities(result):
    """YOLO-cls result.probs를 프론트에서 쓰기 쉬운 체위별 확률 dict로 변환한다."""
    probabilities = empty_posture_probabilities()
    probs_obj = getattr(result, "probs", None)
    probs_data = getattr(probs_obj, "data", None)

    if probs_data is None:
        return probabilities

    if hasattr(probs_data, "detach"):
        raw_scores = probs_data.detach().cpu().tolist()
    elif hasattr(probs_data, "cpu"):
        raw_scores = probs_data.cpu().tolist()
    else:
        raw_scores = list(probs_data)

    for class_index, score in enumerate(raw_scores):
        if isinstance(result.names, dict):
            raw_class_name = result.names.get(class_index, result.names.get(str(class_index), str(class_index)))
        else:
            raw_class_name = result.names[class_index]

        posture = CLASS_NAME_MAP.get(raw_class_name, raw_class_name)
        if posture in probabilities:
            probabilities[posture] = max(probabilities[posture], float(score))

    total = sum(probabilities.values())
    if total > 0:
        probabilities = {key: round(value / total, 3) for key, value in probabilities.items()}

    return probabilities


def load_initial_data():
    global GLOBAL_PATIENTS
    try:
        with open("patients.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            GLOBAL_PATIENTS = data.get("patients", [])
        print(f"✅ {len(GLOBAL_PATIENTS)}명 환자 데이터 로드 완료")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")


def get_video_duration_sec(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0
    return frame_count / fps


def extract_frame_by_time(video_path, target_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    return frame


def classify_posture_from_video(video_path, target_sec=None):
    duration = get_video_duration_sec(video_path)
    if duration <= 0:
        return None, 0.0, empty_posture_probabilities()

    if target_sec is None:
        elapsed_server_time = time.time() - SERVER_START_TIME
        current_video_sec = elapsed_server_time % duration
    else:
        current_video_sec = float(target_sec) % duration

    frame = extract_frame_by_time(video_path, current_video_sec)
    if frame is None:
        return None, 0.0, empty_posture_probabilities()

    results = model.predict(frame, verbose=False)
    if not results:
        return None, 0.0, empty_posture_probabilities()

    result = results[0]

    # YOLO classification 모델이면 result.probs가 존재한다.
    # detection 모델이나 잘못된 가중치에서는 probs가 None일 수 있으므로 서버가 죽지 않게 방어한다.
    if getattr(result, "probs", None) is None:
        print("⚠️ 현재 모델 출력에 probs가 없습니다. 체위 분류용 YOLO-cls 모델인지 확인하세요.")
        return None, 0.0, empty_posture_probabilities()

    class_index = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    if isinstance(result.names, dict):
        raw_class_name = result.names.get(class_index, result.names.get(str(class_index), str(class_index)))
    else:
        raw_class_name = result.names[class_index]
    posture = CLASS_NAME_MAP.get(raw_class_name, raw_class_name)
    probabilities = extract_posture_probabilities(result)
    return posture, confidence, probabilities


def classify_single_patient_video(p, target_sec=None):
    video_file = p.get("videoFile")
    if not video_file:
        return None, 0.0, None, empty_posture_probabilities()

    video_path = os.path.join(VIDEO_DIR, video_file)
    if not os.path.exists(video_path):
        print(f"⚠️ 영상 없음: {video_path}")
        return None, 0.0, None, empty_posture_probabilities()

    posture, confidence, probabilities = classify_posture_from_video(video_path, target_sec=target_sec)
    if posture not in POSTURE_KEYS:
        print(f"⚠️ 알 수 없는 체위 클래스: {posture}")
        return None, 0.0, None, empty_posture_probabilities()

    video_url = f"/static/videos/{video_file}"
    return posture, confidence, video_url, probabilities


def record_posture_vote(patient_id, posture, confidence, probabilities=None):
    """1초마다 들어온 실시간 체위 분류 결과를 환자별 버퍼에 저장한다."""
    now = time.time()

    if patient_id not in POSTURE_VOTES:
        POSTURE_VOTES[patient_id] = []

    POSTURE_VOTES[patient_id].append({
        "t": now,
        "posture": posture,
        "confidence": float(confidence),
        "probabilities": probabilities or empty_posture_probabilities()
    })

    # 메모리가 계속 늘지 않도록 최근 30초만 유지한다.
    POSTURE_VOTES[patient_id] = [
        v for v in POSTURE_VOTES[patient_id]
        if now - v["t"] <= 30
    ]


def get_majority_posture(patient_id, window_sec=10):
    """최근 window_sec 동안 가장 많이 나온 체위를 대표 체위로 반환한다."""
    now = time.time()
    votes = [
        v for v in POSTURE_VOTES.get(patient_id, [])
        if now - v["t"] <= window_sec
    ]

    if not votes:
        return None, 0.0, 0

    counts = {}
    conf_sum = {}

    for v in votes:
        posture = v["posture"]
        counts[posture] = counts.get(posture, 0) + 1
        conf_sum[posture] = conf_sum.get(posture, 0.0) + float(v.get("confidence", 0.0))

    # 가장 많이 나온 체위 선택. 동률이면 평균 신뢰도가 높은 체위 선택.
    majority_posture = max(
        counts.keys(),
        key=lambda pos: (counts[pos], conf_sum[pos] / max(counts[pos], 1))
    )

    avg_conf = conf_sum[majority_posture] / max(counts[majority_posture], 1)
    return majority_posture, avg_conf, len(votes)


def get_recent_votes(patient_id, window_sec=10):
    now = time.time()
    return [
        v for v in POSTURE_VOTES.get(patient_id, [])
        if now - v["t"] <= window_sec
    ]


def get_vote_distribution(patient_id, window_sec=10):
    """최근 window_sec 동안의 체위 vote 분포를 0~1 확률 형태로 반환한다."""
    votes = get_recent_votes(patient_id, window_sec=window_sec)

    if not votes:
        return empty_posture_probabilities()

    counts = {key: 0 for key in POSTURE_KEYS}
    for v in votes:
        posture = v.get("posture")
        if posture in counts:
            counts[posture] += 1

    total = max(len(votes), 1)
    return {key: round(counts[key] / total, 3) for key in POSTURE_KEYS}


def get_average_posture_probabilities(patient_id, window_sec=10):
    """최근 window_sec 동안 누적된 YOLO softmax 확률의 평균을 반환한다."""
    votes = get_recent_votes(patient_id, window_sec=window_sec)
    if not votes:
        return empty_posture_probabilities()

    sums = {key: 0.0 for key in POSTURE_KEYS}
    usable = 0
    for v in votes:
        probs = v.get("probabilities") or {}
        if not any(float(probs.get(key, 0.0)) > 0 for key in POSTURE_KEYS):
            continue
        for key in POSTURE_KEYS:
            sums[key] += float(probs.get(key, 0.0))
        usable += 1

    if usable == 0:
        return empty_posture_probabilities()

    return {key: round(sums[key] / usable, 3) for key in POSTURE_KEYS}


def update_posture_summary_timeline(p, posture):
    """10초마다 대표 체위를 최근 체위 변화 타임라인에 추가한다."""
    if "postureSummaryTimeline" not in p or not isinstance(p["postureSummaryTimeline"], list):
        p["postureSummaryTimeline"] = []

    p["postureSummaryTimeline"].append({
        "p": posture,
        "t": time.strftime("%H:%M:%S")
    })

    # 최근 6개 구간만 표시한다. 10초 단위이면 최대 최근 60초 요약이다.
    if len(p["postureSummaryTimeline"]) > 6:
        p["postureSummaryTimeline"] = p["postureSummaryTimeline"][-6:]


def update_heatmap_by_posture(p, posture):
    if not p.get("heatmap"):
        p["heatmap"] = [{"p": posture, "m": 15}]
        return

    last = p["heatmap"][-1]
    if last["p"] == posture:
        last["m"] += 15
    else:
        p["heatmap"].append({"p": posture, "m": 15})
        if len(p["heatmap"]) > 6:
            p["heatmap"].pop(0)


def update_elapsed_minutes(p, posture):
    previous_posture = p.get("currentPosition")
    if previous_posture == posture:
        p["elapsedMinutes"] = p.get("elapsedMinutes", 0) + 15
    else:
        p["elapsedMinutes"] = 0


def update_risk_score(p):
    max_time = p.get("maxAllowedMinutes", 120)
    elapsed = p.get("elapsedMinutes", 0)

    if "riskFactors" not in p:
        return

    time_ratio = elapsed / max_time
    p["riskFactors"]["pressureDuration"] = min(100, int(time_ratio * 100))

    f = p["riskFactors"]
    calculated = (
        f.get("pressureDuration", 0) * 0.40 +
        f.get("frictionShear", 0) * 0.20 +
        f.get("moisture", 0) * 0.15 +
        f.get("mobility", 0) * 0.15 +
        f.get("nutrition", 0) * 0.10
    )
    p["riskScore"] = round(min(100, max(0, calculated)))


def update_spot_risks(p):
    if "spotRisks" not in p:
        p["spotRisks"] = {k: 0 for k in POSTURE_SPOT_WEIGHTS["Supine"]}

    cur_pos = p.get("currentPosition", "Supine")
    weights = POSTURE_SPOT_WEIGHTS.get(cur_pos, POSTURE_SPOT_WEIGHTS["Supine"])
    pressure = p.get("riskFactors", {}).get("pressureDuration", 0)
    braden = p.get("bradenScore", 15)
    vuln = max(0, (23 - braden) * 1.5)

    for spot, weight in weights.items():
        target = min(100, pressure * weight + vuln)
        current = p["spotRisks"].get(spot, 0)
        p["spotRisks"][spot] = round(current + (target - current) * 0.3)


def update_status_and_recommendation(p):
    postures = ["Left", "Right", "Supine"]
    score = p.get("riskScore", 0)
    p["status"] = "immediate" if score >= 70 else "caution" if score >= 40 else "stable"

    usage = {"Left": 0, "Right": 0, "Supine": 0}
    for seg in p.get("heatmap", []):
        usage[seg.get("p", "Supine")] += seg.get("m", 1)

    current = p.get("currentPosition", "Supine")
    candidates = [pos for pos in postures if pos != current]
    p["recommendedPosition"] = min(candidates, key=lambda pos: usage.get(pos, 0))

    label = ACTION_LABELS.get(p["recommendedPosition"], p["recommendedPosition"])
    if p["status"] == "immediate":
        p["recommendedAction"] = f"{label}로 즉시 변경"
    elif p["status"] == "caution":
        p["recommendedAction"] = f"{label}로 변경 준비"
    else:
        p["recommendedAction"] = "상태 재평가"


def update_patients_by_video():
    for p in GLOBAL_PATIENTS:
        patient_id = p.get("id")

        # 10초마다 실행되는 전체 대시보드 갱신용.
        # 최근 10초 동안 1초 단위로 수집된 YOLO 결과 중 가장 많이 나온 체위를 대표 체위로 사용한다.
        posture, confidence, vote_count = get_majority_posture(patient_id, window_sec=10)
        probabilities = get_average_posture_probabilities(patient_id, window_sec=10) if vote_count else empty_posture_probabilities()

        video_url = None
        if p.get("videoFile"):
            video_path = os.path.join(VIDEO_DIR, p.get("videoFile"))
            if os.path.exists(video_path):
                video_url = f"/static/videos/{p.get('videoFile')}"

        # 아직 상세 페이지를 열지 않아 실시간 vote가 없는 경우에는 현재 영상 프레임을 한 번 분류해 fallback으로 사용한다.
        if posture is None:
            posture, confidence, video_url_from_cls, probabilities = classify_single_patient_video(p)
            if video_url_from_cls:
                video_url = video_url_from_cls

        if posture is None:
            continue

        update_elapsed_minutes(p, posture)
        p["currentPosition"] = posture
        p["postureConfidence"] = round(confidence, 3)
        p["postureVoteCount10s"] = vote_count
        p["postureProbabilities"] = probabilities
        p["postureVoteDistribution10s"] = get_vote_distribution(patient_id, window_sec=10)

        if video_url:
            p["videoUrl"] = video_url

        # 기존 히트맵과 위험도는 10초마다 대표 체위 기준으로 갱신한다.
        update_heatmap_by_posture(p, posture)

        # 최근 체위 변화 타임라인은 10초마다 대표 체위를 1개씩 추가한다.
        update_posture_summary_timeline(p, posture)

        update_risk_score(p)
        update_spot_risks(p)
        update_status_and_recommendation(p)

    GLOBAL_PATIENTS.sort(key=lambda x: x.get("riskScore", 0), reverse=True)
    for i, p in enumerate(GLOBAL_PATIENTS):
        p["priority"] = i + 1


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/posture/<patient_id>", methods=["GET"])
def get_realtime_posture(patient_id):
    p = next((item for item in GLOBAL_PATIENTS if item.get("id") == patient_id), None)
    if p is None:
        return jsonify({"error": "patient not found"}), 404

    # 프론트에서 현재 재생 중인 video.currentTime을 넘기면
    # 실제 화면에 보이는 프레임 기준으로 YOLO 분류한다.
    target_sec = request.args.get("t", default=None, type=float)
    posture, confidence, video_url, probabilities = classify_single_patient_video(p, target_sec=target_sec)

    if posture is None:
        # 서버 오류로 처리하지 않고 기존 체위를 반환한다.
        # 이렇게 해야 프론트의 1초 polling이 중단되지 않는다.
        posture = p.get("currentPosition", "Supine")
        confidence = 0.0
        probabilities = p.get("postureProbabilities", empty_posture_probabilities())
        if p.get("videoFile"):
            video_url = f"/static/videos/{p.get('videoFile')}"

    # 실시간 현재 체위 표시용 값만 즉시 갱신한다.
    # 위험 점수, 히트맵, 우선순위는 /api/patients에서 10초마다 대표 체위로 갱신한다.
    p["currentPosition"] = posture
    p["postureConfidence"] = round(confidence, 3)
    p["postureProbabilities"] = probabilities
    if video_url:
        p["videoUrl"] = video_url

    # 10초 요약 타임라인 계산을 위해 실시간 분류 결과를 버퍼에 누적한다.
    record_posture_vote(p.get("id"), posture, confidence, probabilities)
    majority_posture, majority_confidence, vote_count = get_majority_posture(p.get("id"), window_sec=10)

    return jsonify({
        "id": p.get("id"),
        "currentPosition": p.get("currentPosition"),
        "postureConfidence": p.get("postureConfidence"),
        "postureProbabilities": p.get("postureProbabilities", empty_posture_probabilities()),
        "postureVoteCount10s": vote_count,
        "postureMajority10s": majority_posture,
        "postureMajorityConfidence10s": round(majority_confidence, 3),
        "postureVoteDistribution10s": get_vote_distribution(p.get("id"), window_sec=10),
        "videoUrl": p.get("videoUrl"),
        "voteBuffered": True
    })


@app.route("/api/patients", methods=["GET"])
def get_patients():
    update_patients_by_video()
    return jsonify(GLOBAL_PATIENTS)


if __name__ == "__main__":
    load_initial_data()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
