"""품질 검사와 보정 (역할 가이드 8단계). 보정은 한 번만, 모든 에피소드에 같은 설정."""
import numpy as np
from scipy.signal import savgol_filter

# 결과를 보기 전에 고정. max_speed 는 파일럿 데이터를 보고 멘토와 정한 뒤 바꾼다(바꾸면 전량 재처리).
DEFAULT_SETTINGS = {
    "max_gap": 5,          # 이 프레임 수 이하의 누락만 선형 보간
    "max_speed": 5.0,      # m/s. 이보다 빠르면 튐으로 표시 (임시값)
    "savgol_window": 7,    # 홀수
    "savgol_poly": 2,
}


def missing_ratio(kp):
    """키포인트별 누락 프레임 비율 (K,)."""
    return np.isnan(kp).any(axis=2).mean(axis=0)


def spike_mask(kp, fps, max_speed):
    """프레임 사이 속도가 max_speed 초과인 (T,K) bool. 튄 프레임(뒤쪽)을 표시."""
    v = np.linalg.norm(np.diff(kp, axis=0), axis=2) * fps
    m = np.zeros(kp.shape[:2], dtype=bool)
    with np.errstate(invalid="ignore"):
        m[1:] = v > max_speed
    return m


def _runs(valid):
    """True 연속 구간 (start, end_exclusive) 목록."""
    d = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fill_short_gaps(x, max_gap):
    """1D 시계열의 NaN 중 길이 <= max_gap 인 구간만 선형 보간. 앞뒤가 비면 채우지 않는다."""
    x = x.copy()
    bad = np.isnan(x)
    if not bad.any() or bad.all():
        return x
    for s, e in _runs(bad):
        if s == 0 or e == len(x) or (e - s) > max_gap:
            continue
        x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return x


def correct(kp_raw, fps, settings=None):
    """튐 제거(NaN 처리) -> 짧은 누락 보간 -> 연속 구간별 Savitzky-Golay. (보정본, 품질 dict)."""
    st = {**DEFAULT_SETTINGS, **(settings or {})}
    kp = kp_raw.copy()
    spikes = spike_mask(kp, fps, st["max_speed"])
    kp[spikes] = np.nan
    T, K, _ = kp.shape
    for k in range(K):
        for c in range(3):
            kp[:, k, c] = fill_short_gaps(kp[:, k, c], st["max_gap"])
    w = st["savgol_window"]
    for k in range(K):
        valid = ~np.isnan(kp[:, k]).any(axis=1)
        for s, e in _runs(valid):
            if e - s >= w:
                kp[s:e, k] = savgol_filter(kp[s:e, k], w, st["savgol_poly"], axis=0)
    report = {
        "frames": int(T),
        "missing_raw": missing_ratio(kp_raw).round(4).tolist(),
        "spike_ratio": spikes.mean(axis=0).round(4).tolist(),
        "missing_after": missing_ratio(kp).round(4).tolist(),
        "longest_gap_after": [_longest_nan_run(kp[:, k]) for k in range(K)],
    }
    return kp, report, st


def _longest_nan_run(x):
    bad = np.isnan(x).any(axis=1)
    runs = _runs(bad)
    return int(max((e - s for s, e in runs), default=0))


def settings_string(st):
    return "fill<=%df, spike>%.1fm/s->NaN, savgol(w=%d,p=%d)" % (
        st["max_gap"], st["max_speed"], st["savgol_window"], st["savgol_poly"])
