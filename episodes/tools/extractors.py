"""키포인트 추출기. 장비가 정해지면 이 파일만 바꾼다 (역할 가이드 4단계).

공통 출력 규약:
  EXO_NAMES 순서의 키포인트 2D 픽셀 (T,K,2), 못 찾으면 NaN.
  ego 는 프레임별 엄지-검지 거리 (T,) 만 반환.
기본 구현은 MediaPipe. exo 는 카메라 3대(정면·왼쪽·오른쪽) 각각에서 2D 를 뽑고 multiview.py 가 삼각측량으로 3D 를 만든다.
"""
import numpy as np

# 어깨·팔꿈치는 Pose, 손 관련은 Hands 에서 뽑는다. thumb_tip / index_tip 이름은 5-2번 코드가 찾는 이름.
EXO_NAMES = ["shoulder", "elbow", "wrist", "index_mcp", "pinky_mcp", "thumb_tip", "index_tip"]

# MediaPipe 인덱스
_HAND = {"wrist": 0, "thumb_tip": 4, "index_mcp": 5, "index_tip": 8, "pinky_mcp": 17}
_POSE = {"right": {"shoulder": 12, "elbow": 14}, "left": {"shoulder": 11, "elbow": 13}}


def _frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()


def _mp_label(hand, mirrored):
    """MediaPipe Hands 는 입력이 셀카처럼 좌우 반전됐다고 가정하고 손 라벨을 붙인다.
    반전하지 않은 영상(exo, ego)에서는 실제 오른손이 "Left" 로 나온다. 그래서 기본은 라벨을 맞바꿔 찾는다.
    파일럿 영상 한 프레임을 overlay.py 로 찍어 반드시 확인할 것."""
    if mirrored:
        return hand
    return "left" if hand == "right" else "right"


def _pick_hand(res, hand, mirrored=False):
    """검출된 손 중 요청한 손(left/right)만."""
    if not res.multi_hand_landmarks:
        return None
    want = _mp_label(hand, mirrored)
    for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
        if hd.classification[0].label.lower() == want:
            return lm
    return None


def extract_exo_2d(video_path, hand="right", mirrored=False):
    import mediapipe as mp
    H, P = mp.solutions.hands, mp.solutions.pose
    out = []
    with H.Hands(max_num_hands=2, model_complexity=1) as hands, P.Pose(model_complexity=1) as pose:
        for rgb in _frames(video_path):
            h, w = rgb.shape[:2]
            row = np.full((len(EXO_NAMES), 2), np.nan)
            hl = _pick_hand(hands.process(rgb), hand, mirrored)
            if hl is not None:
                for n, i in _HAND.items():
                    row[EXO_NAMES.index(n)] = (hl.landmark[i].x * w, hl.landmark[i].y * h)
            pr = pose.process(rgb)
            if pr.pose_landmarks:
                for n, i in _POSE[hand].items():
                    p = pr.pose_landmarks.landmark[i]
                    row[EXO_NAMES.index(n)] = (p.x * w, p.y * h)
            out.append(row)
    return np.stack(out) if out else np.zeros((0, len(EXO_NAMES), 2))


def extract_ego_pinch(video_path, hand="right", mirrored=False):
    """ego 영상에서 엄지끝-검지끝 거리 (T,). world landmark(손 크기 기준 미터)를 써서 카메라 위치와 무관.
    절대 크기보다 에피소드 사이 일관성이 중요하므로 같은 도구·설정을 전량에 쓴다."""
    import mediapipe as mp
    H = mp.solutions.hands
    want = _mp_label(hand, mirrored)
    out = []
    with H.Hands(max_num_hands=2, model_complexity=1) as hands:
        for rgb in _frames(video_path):
            res = hands.process(rgb)
            d = np.nan
            if res.multi_hand_world_landmarks:
                for wl, hd in zip(res.multi_hand_world_landmarks, res.multi_handedness):
                    if hd.classification[0].label.lower() == want:
                        a, b = wl.landmark[4], wl.landmark[8]
                        d = float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))
                        break
            out.append(d)
    return np.array(out)
