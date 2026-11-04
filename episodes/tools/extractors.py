"""키포인트 추출기. 장비가 정해지면 이 파일만 바꾼다 (역할 가이드 4단계).

공통 출력 규약:
  EXO_NAMES 순서의 키포인트 2D 픽셀 (T,K,2), 못 찾으면 NaN.
  ego 는 프레임별 엄지-검지 거리 (T,) 만 반환.
기본 구현은 MediaPipe. exo 는 카메라 3대(정면·왼쪽·오른쪽) 각각에서 2D 를 뽑고 multiview.py 가 삼각측량으로 3D 를 만든다.
"""
import numpy as np

# 어깨·팔꿈치는 Pose, 손 관련은 Hands 에서 뽑는다. thumb_tip / index_tip 이름은 5-2번 코드가 찾는 이름.
EXO_NAMES = ["shoulder", "elbow", "wrist", "index_mcp", "pinky_mcp", "thumb_tip", "index_tip"]

# MediaPipe 인덱스 (Pose 의 좌우는 사람 몸 기준)
_HAND = {"wrist": 0, "thumb_tip": 4, "index_mcp": 5, "index_tip": 8, "pinky_mcp": 17}
_POSE = {"right": {"shoulder": 12, "elbow": 14, "wrist": 16}, "left": {"shoulder": 11, "elbow": 13, "wrist": 15}}

# 손 주변 잘라내기: 손 중심 ~ 손목 + CROP_SHIFT x (손목 - 팔꿈치), 한 변 = CROP_SIZE x 아래팔 길이
CROP_SHIFT, CROP_SIZE, CROP_MIN_PX = 0.4, 1.6, 160
MATCH_MAX = 0.5      # Hands 손목이 Pose 손목에서 아래팔 길이의 이 배수 안이어야 같은 손으로 봄


def _frames(path, max_frames=None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = 0
    while max_frames is None or n < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        n += 1
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()


def _ego_label(hand, mirrored):
    """ego 에서 찾을 MediaPipe 손 라벨. 1인칭(손등이 보임) 영상은 반전하지 않아도 실제 오른손이 "Right" 로 나온다
    (T1_E052 60프레임: Right 21 / Left 3). 셀카처럼 반전된 영상이면 반대."""
    if not mirrored:
        return hand
    return "left" if hand == "right" else "right"


def _hand_box(wrist, elbow, w, h):
    """Pose 손목·팔꿈치(px) -> 손을 담는 잘라내기 영역 (x0, y0, x1, y1), 아래팔 길이 L."""
    L = max(float(np.hypot(*(wrist - elbow))), 1.0)
    c = wrist + CROP_SHIFT * (wrist - elbow)
    half = max(CROP_SIZE * L, CROP_MIN_PX) / 2
    x0, y0 = int(max(c[0] - half, 0)), int(max(c[1] - half, 0))
    x1, y1 = int(min(c[0] + half, w)), int(min(c[1] + half, h))
    return (x0, y0, x1, y1), L


def extract_exo_2d(video_path, hand="right", mirrored=False, max_frames=None):
    """exo 는 손 라벨(Left/Right)을 쓰지 않는다. 카메라마다 라벨이 섞이고(T1 exo_left) 다른 사람 손도 잡히기 때문.
    Pose 로 그 사람의 손목·팔꿈치를 찾고, 손 주변을 잘라 확대해 Hands 를 돌린 뒤 Pose 손목에 가장 가까운 손을 쓴다.
    (전체 화면에서는 컵을 쥔 손이 작아 자주 놓침.) mirrored 면 Pose 좌우를 바꾼다."""
    import mediapipe as mp
    H, P = mp.solutions.hands, mp.solutions.pose
    side = hand if not mirrored else ("left" if hand == "right" else "right")
    out = []
    with H.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1) as hands, \
         P.Pose(model_complexity=1) as pose:
        for rgb in _frames(video_path, max_frames):
            h, w = rgb.shape[:2]
            row = np.full((len(EXO_NAMES), 2), np.nan)
            pr = pose.process(rgb)
            if pr.pose_landmarks:
                pp = {n: np.array([pr.pose_landmarks.landmark[i].x * w, pr.pose_landmarks.landmark[i].y * h])
                      for n, i in _POSE[side].items()}
                row[EXO_NAMES.index("shoulder")] = pp["shoulder"]
                row[EXO_NAMES.index("elbow")] = pp["elbow"]
                (x0, y0, x1, y1), L = _hand_box(pp["wrist"], pp["elbow"], w, h)
                if x1 - x0 > 8 and y1 - y0 > 8:
                    res = hands.process(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
                    cw, ch = x1 - x0, y1 - y0
                    best, best_d = None, MATCH_MAX * L
                    for lm in res.multi_hand_landmarks or []:
                        d = np.hypot(x0 + lm.landmark[0].x * cw - pp["wrist"][0], y0 + lm.landmark[0].y * ch - pp["wrist"][1])
                        if d < best_d:
                            best, best_d = lm, d
                    if best is not None:
                        for n, i in _HAND.items():
                            row[EXO_NAMES.index(n)] = (x0 + best.landmark[i].x * cw, y0 + best.landmark[i].y * ch)
            out.append(row)
    return np.stack(out) if out else np.zeros((0, len(EXO_NAMES), 2))


def extract_ego_pinch(video_path, hand="right", mirrored=False):
    """ego 영상에서 엄지끝-검지끝 거리 (T,). world landmark(손 크기 기준 미터)를 써서 카메라 위치와 무관.
    절대 크기보다 에피소드 사이 일관성이 중요하므로 같은 도구·설정을 전량에 쓴다."""
    import mediapipe as mp
    H = mp.solutions.hands
    want = _ego_label(hand, mirrored)
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
