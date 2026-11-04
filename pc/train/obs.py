# -*- coding: utf-8 -*-
"""obs.py 鈥?106 缁磋娴嬬粍瑁呭櫒锛岄€愪笅鏍囧榻愬浐浠?``cfc_task.c`` 鐨?``build_obs()``銆?
鏉冨▉鏉ユ簮鏄?*鍥轰欢浠ｇ爜**锛屼笉鏄?TECH-PLAN 鐨勮〃锛歚`cfc_task.c:323`` 鐨?``build_obs()``
鍐冲畾姣忎竴涓笅鏍囧啓浠€涔堛€傝繖閲岄€愯鐓ф妱锛屼笅鏍囪〃瑙?:data:`OBS_NAMES`銆?
    >>> from obs import DistEstimator, assemble, DomainGuard
    >>> dist = DistEstimator()                       # 姣?session 涓€涓紝瑕?reset
    >>> o = assemble(perc, prop, prev_act, t_s, dist.step(seen, person_h))
    >>> o.shape, o.dtype
    ((106,), dtype('float32'))
    >>> x, ratio = DomainGuard(mean, std, 4.0).apply(o)
    >>> ratio                       # 琚?clip 鐨勬瘮渚嬶紝> 0.05 瑕佸憡璀?    0.0

涓庡浐浠剁殑涓夊**瀹规槗琚畝鍖栨帀**鐨勭粏鑺傦紝杩欓噷閮戒繚鐣欙細

1. ``obs[67]`` 鐨勮窛绂绘槸**浣庨€氬悗**鐨?``dist_k / person_h``锛屼笉鏄灛鏃跺€?   锛坄`cfc_task.c:346``锛涚灛鏃跺€间細璁╀汉涓€韫蹭笅灏辫鍒ゆ垚"璧拌繙浜?鐒跺悗鍓嶅啿锛夈€?2. ``obs[74]`` 鐨?``wave_amp`` 鏄?*閫忎紶** ``PerceptionOut.wave_amp``
   锛坄`cfc_task.c:372`` 鐩存帴鍐?``f->wave_amp``锛夈€傛棭鍏堣繖閲屽啓鎴愮敤
   ``sqrt(sin^2+cos^2)`` 閲嶇畻锛屽苟娉ㄩ噴璇?鍥轰欢閲嶇畻"鈥斺€?*娉ㄩ噴鍜屽浐浠朵唬鐮佺浉鍙?*锛?   瀵艰嚧璁粌涓庨儴缃插湪杩欎竴缁翠笂鐢ㄤ簡涓嶅悓鐨勯噺銆傚凡鏀规涓洪€忎紶銆?   ``obs[75]`` 鐨?``gesture_conf`` 鍥轰欢涔熺洿鎺ュ彇 ``f->gesture_conf``銆?3. ``obs[76]``/``obs[77]`` 鍥轰欢鍐欐 0锛岃缁冨寘閲屼篃蹇呴』鏄?0 鈥斺€?杩欎袱缁寸殑 std 鍥犳
   鎭掍负 0锛屽鍑烘椂蹇呴』缁?std 璁句笅闄愶紙瑙?:func:`export_blob.std_with_floor`锛夛紝
   鍚﹀垯鍥轰欢鐨?``blob_parse`` 浼氫互 "bad input statistics" 鎷掔粷鏁翠釜妯″瀷銆?
鍗曚綅锛歚`theta_img`` 鏄?*搴?*锛堜笉鏄綊涓€鍖栵級锛宍`yaw``/``q_head`` 鏄害锛宍`wheel_x`` 绫筹紝
``vx`` 绫?绉掞紝``vyaw`` 搴?绉掞紝``feature_age_ms`` 姣銆?"""

from __future__ import annotations

import math

import numpy as np

F32 = np.float32

#: 固件 OBS_DIM（cfc_task.c:43）
OBS_DIM = 106
#: 固件 FEAT_MAX_DIM（cfc_task.c:42）：观测里 0..63 是视觉特征
FEAT_DIM = 64

#: 固件 arb_cfg_t 默认值里与本文件相关的几项（control_arbiter.c:91 cfg_defaults）
ARB_DEFAULTS = {
    "fov_h": 60.0,
    "dist_k": 0.90,
    "dist_lpf": 0.50,
    "dist_target": 1.00,
    "cfc_hz": 50,
    "head_max": 45.0,
    "vx_max": 0.30,
    "vyaw_max": 90.0,
    "clip_sigma": 4.0,
}


# --------------------------------------------------------------------------- #
# 涓嬫爣琛紙鍞竴涓€浠斤紝瀵煎嚭鐨?manifest 涓庨獙鏀堕兘璇诲畠锛?# --------------------------------------------------------------------------- #
def _build_names():
    names = ["feat_%02d" % i for i in range(FEAT_DIM)]
    names += [
        "theta_img",       # 64
        "person_w",        # 65
        "person_h",        # 66
        "dist_est",        # 67  浣庨€氬悗鐨勮窛绂
        "person_conf",     # 68
        "seen",            # 69
        "hand_up_l",       # 70
        "hand_up_r",       # 71
        "wave_sin",        # 72
        "wave_cos",        # 73
        "wave_amp",        # 74  sqrt(sin^2+cos^2)锛屽浐浠堕噸绠
        "gesture_conf",    # 75  鍥轰欢閲?= person_conf
        "reserved_76",     # 76  鍥轰欢鍐欐 0
        "reserved_77",     # 77  鍥轰欢鍐欐 0
        "roll",            # 78
        "pitch",           # 79
        "pit_gyro",        # 80
        "dq",              # 81
        "yaw",             # 82  澶撮儴 IMU 鐨勭粷瀵?yaw锛堝害锛
        "q_head",          # 83
        "target_head_pos", # 84
        "head_err",        # 85  q_head - target_head_pos
        "wheel1_vel",      # 86
        "wheel2_vel",      # 87
        "wheel_dvel",      # 88  wheel1_vel - wheel2_vel
        "wheel_x",         # 89
        "stable_pos",      # 90
        "pos_err",         # 91  wheel_x - stable_pos
        "body_yaw",        # 92  yaw - q_head
        "yaw_err",         # 93  (yaw - q_head) - stable_yaw
        "robot_state",     # 94
        "vx",              # 95
        "vyaw",            # 96
        "control_mode",    # 97
        "feat_age_s",      # 98  绉掞紱-1 琛ㄧず浠庢湭鏀跺埌
        "gaze_danger",     # 99
        "prev_head_n",     # 100 s_prev_head / 45
        "prev_vyaw_n",     # 101 s_prev_vyaw / 90
        "prev_mode_n",     # 102 s_prev_mode / 3
        "prev_vx_n",       # 103 s_prev_vx / 0.30
        "clk_sin",         # 104 sin(2*pi*t/10)
        "clk_cos",         # 105 cos(2*pi*t/10)
    ]
    assert len(names) == OBS_DIM, len(names)
    return tuple(names)


#: 106 维观测的逐维名称，供打印与断言使用
OBS_NAMES = _build_names()

#: (组名, 起始下标, 长度)，与固件 build_obs 的分段一一对应
OBS_GROUPS = (
    ("vision_feat", 0, 64),
    ("target", 64, 6),
    ("gesture", 70, 8),
    ("proprio", 78, 22),
    ("feedback", 100, 4),
    ("clock", 104, 2),
)

#: 固件写死为 0 的两维；导出时 std 必须设下限，否则 blob_parse 会拒绝
CONSTANT_OBS_DIMS = (76, 77)

#: 上一步动作反馈各通道的满量程，与固件 s_prev_* 一致
FEEDBACK_SCALE = {"head": 45.0, "vyaw": 90.0, "mode": 3.0, "vx": 0.30}


# --------------------------------------------------------------------------- #
# 鍙栧€煎皬宸ュ叿锛歱erc/prop 鏃㈠彲浠ユ槸 dataclass锛圥erceptionOut锛変篃鍙互鏄?dict
# --------------------------------------------------------------------------- #
_MISSING = object()


def _get(obj, name, default=_MISSING):
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)
    if default is not _MISSING:
        return default
    raise KeyError("瑙傛祴杈撳叆缂哄皯瀛楁 %r锛圥erceptionOut / proprio dict 蹇呴』缁欏叏锛" % (name,))


def _f(v):
    """float32 鏍囬噺锛汵aN/Inf 涓€寰嬪彉 0锛堝浐浠剁殑 clampf 瀵?NaN 涔熻繑鍥?0锛夈€?"""
    x = float(v)
    if not math.isfinite(x):
        return F32(0.0)
    return F32(x)


# --------------------------------------------------------------------------- #
# 璺濈浼拌锛堝鍒?cfc_task.c:346 鐨勯潤鎬佷綆閫氾級
# --------------------------------------------------------------------------- #
class DistEstimator:
    """鍗曠洰璺濈鐨?*浣庨€?*浼拌锛屼笌鍥轰欢 ``build_obs()`` 閲岀殑 ``s_dist`` 閫愪綅鍚屾瀯銆?
    鍥轰欢鎶婂畠鍋氭垚浜嗗嚱鏁板唴 ``static``锛堣法甯т繚鐣欙級锛涜法 session 閲嶆斁鏃跺繀椤绘樉寮?    :meth:`reset`锛屽惁鍒欑涓€鏉?session 鐨勫熬宸翠細姹℃煋涓嬩竴鏉°€?
    Args:
        dist_k: ``dist = dist_k / person_h`` 鐨勫垎瀛愶紝榛樿鍙栧浐浠堕粯璁?0.90銆?        dist_lpf: 浣庨€氭椂闂村父鏁帮紙绉掞級锛岄粯璁?0.50銆?        dist_target: 璺濈鍒濆€硷紙绫筹級锛岄粯璁?1.00銆?        cfc_hz: 鎺ㄧ悊棰戠巼锛岀敤鏉ョ畻 dt锛岄粯璁?50銆?    """

    def __init__(self, dist_k=ARB_DEFAULTS["dist_k"], dist_lpf=ARB_DEFAULTS["dist_lpf"],
                 dist_target=ARB_DEFAULTS["dist_target"], cfc_hz=ARB_DEFAULTS["cfc_hz"]):
        self.dist_k = float(dist_k)
        self.dist_lpf = float(dist_lpf)
        self.dist_target = float(dist_target)
        self.cfc_hz = int(cfc_hz)
        self.reset()

    def reset(self):
        """娓呯┖浣庨€氱姸鎬侊紙姣忎釜 session / 姣忔鏂祦閲嶈繛鍚庤皟鐢級銆?"""
        self.dist = 0.0

    def step(self, seen, person_h):
        """鎺ㄨ繘涓€甯э紝杩斿洖鏈抚閫佽繘 ``obs[67]`` 鐨勮窛绂伙紙绫筹級銆?
        閫愯瀵瑰簲 ``cfc_task.c:346-359``锛歳aw 鍦?娌＄湅瑙佷汉"鏃跺彇涓婁竴甯х殑婊ゆ尝鍊硷紝
        鎵€浠ヤ涪澶辩洰鏍囨椂璺濈淇濇寔涓嶅姩锛岃€屼笉鏄帀鍒?0銆?        """
        tau = min(max(float(self.dist_lpf), 0.02), 10.0)
        dt = 1.0 / float(self.cfc_hz if self.cfc_hz > 0 else 50)
        alpha = min(max(dt / (tau + dt), 0.0), 1.0)
        h = float(person_h)
        if bool(int(seen)) and h > 0.02:
            raw = self.dist_k / h
        else:
            raw = self.dist
        if self.dist <= 0.0:
            self.dist = self.dist_target
        self.dist += alpha * (raw - self.dist)
        self.dist = min(max(self.dist, 0.05), 20.0)
        return F32(self.dist)


# --------------------------------------------------------------------------- #
# 缁勮
# --------------------------------------------------------------------------- #
def _prev_fields(prev):
    """鎶婁笂涓€姝ュ姩浣滅粺涓€鎴?(head_deg, vyaw_dps, vx_mps, mode)銆?"""
    if prev is None:
        return 0.0, 0.0, 0.0, 0
    if isinstance(prev, dict):
        head = prev.get("head_deg", prev.get("head_des", 0.0))
        vyaw = prev.get("vyaw_dps", prev.get("vyaw_des", 0.0))
        vx = prev.get("vx_mps", prev.get("vx_des", 0.0))
        mode = prev.get("mode", 0)
    else:
        head = getattr(prev, "head_des", getattr(prev, "head_deg", 0.0))
        vyaw = getattr(prev, "vyaw_des", getattr(prev, "vyaw_dps", 0.0))
        vx = getattr(prev, "vx_des", getattr(prev, "vx_mps", 0.0))
        mode = getattr(prev, "mode", 0)
    return float(head), float(vyaw), float(vx), int(mode)


def assemble(perc, prop, prev=None, t_s=0.0, dist_est=0.0, input_dim=OBS_DIM):
    """缁勮涓€甯ц娴嬨€?
    Args:
        perc: ``PerceptionOut`` 鎴栧悓鍚嶅瓧娈电殑 dict銆傜敤鍒?``feat_64``锛堜笉鏄?            ``feat_raw``锛夈€乣`theta_img``銆乣`person_w``銆乣`person_h``銆?            ``person_conf``銆乣`seen``銆乣`hand_up_L``銆乣`hand_up_R``銆?            ``wave_sin``銆乣`wave_cos``銆乣`gesture_conf``銆?        prop: 鏈綋鎰熺煡 dict锛岄敭瑙?:data:`OBS_NAMES` 鐨?78..99 娈碉紙鍥轰欢鍏ㄥ眬鍙橀噺鍚屽悕锛夈€?            ``yaw`` 鏄?*澶撮儴 IMU** 鐨勭粷瀵?yaw锛堝害锛夛紝鍥犳 ``yaw - q_head`` 鎵嶆槸鏈鸿韩 yaw銆?        prev: 涓婁竴姝ュ姩浣滐紙``TeacherAction`` 鎴?``{"head_des","vyaw_des","vx_des","mode"}``锛夈€?            鍥轰欢鐨?``s_prev_*`` 鏄?*涓婁竴娆?publish 鐨勫€?*锛屾墍浠ョ t 甯х敤绗?t-1 甯х殑鍔ㄤ綔銆?        t_s: 鍗曡皟鏃堕棿鎴筹紙绉掞級锛屽彧鐢ㄤ簬 104/105 鐨勪綆棰戞椂閽熴€?        dist_est: 鏈抚鐨勪綆閫氳窛绂伙紙鏉ヨ嚜 :meth:`DistEstimator.step`锛夈€?        input_dim: 妯″瀷澹版槑鐨勮緭鍏ョ淮搴︼紱< 106 鏃跺熬閮ㄦ竻闆讹紙澶嶅埢鍥轰欢 cfc_task.c:415锛夈€?
    Returns:
        ``(input_dim,) float32``銆? 106 鐨勯儴鍒嗘亽涓?0锛堝浐浠?memset 鍚庡彧鍐?106 涓級銆?    """
    n = max(int(input_dim), OBS_DIM)
    obs = np.zeros(n, dtype=F32)

    feat = np.asarray(_get(perc, "feat_64"), dtype=F32).reshape(-1)
    if feat.shape[0] < FEAT_DIM:
        raise ValueError("feat_64 鑷冲皯瑕?%d 缁达紝鏀跺埌 %d" % (FEAT_DIM, feat.shape[0]))
    obs[0:FEAT_DIM] = feat[:FEAT_DIM]

    seen = int(_get(perc, "seen", 0))
    person_h = _get(perc, "person_h", 0.0)
    wave_sin = _get(perc, "wave_sin", 0.0)
    wave_cos = _get(perc, "wave_cos", 0.0)
    person_conf = _get(perc, "person_conf", 0.0)

    # 64..69 鐩爣
    obs[64] = _f(_get(perc, "theta_img", 0.0))
    obs[65] = _f(_get(perc, "person_w", 0.0))
    obs[66] = _f(person_h)
    obs[67] = _f(dist_est)
    obs[68] = _f(person_conf)
    obs[69] = F32(seen)

    # 70..77 鎵嬪娍鍩哄厓
    obs[70] = F32(int(_get(perc, "hand_up_L", 0)))
    obs[71] = F32(int(_get(perc, "hand_up_R", 0)))
    obs[72] = _f(wave_sin)
    obs[73] = _f(wave_cos)
    # 鍥轰欢閫忎紶 A 渚х殑 wave_amp锛坈fc_task.c:372 鍐?f->wave_amp锛夛紝涓嶈兘鍦ㄨ繖閲岄噸绠
    obs[74] = _f(_get(perc, "wave_amp", 0.0))
    obs[75] = _f(_get(perc, "gesture_conf", person_conf))
    # 76/77 鍥轰欢鍐欐 0锛岃妯″潡鏂囨。绗?3 鏉?
    # 78..99 鏈綋鎰熺煡
    def g(name):
        return _f(prop[name])

    q_head = g("q_head")
    target_head = g("target_head_pos")
    yaw = g("yaw")
    w1 = g("wheel1_vel")
    w2 = g("wheel2_vel")
    wheel_x = g("wheel_x")
    stable_pos = g("stable_pos")
    stable_yaw = g("stable_yaw")
    obs[78] = g("roll")
    obs[79] = g("pitch")
    obs[80] = g("pit_gyro")
    obs[81] = g("dq")
    obs[82] = yaw
    obs[83] = q_head
    obs[84] = target_head
    obs[85] = F32(q_head - target_head)
    obs[86] = w1
    obs[87] = w2
    obs[88] = F32(w1 - w2)
    obs[89] = wheel_x
    obs[90] = stable_pos
    obs[91] = F32(wheel_x - stable_pos)
    obs[92] = F32(yaw - q_head)
    obs[93] = F32((yaw - q_head) - stable_yaw)
    obs[94] = g("robot_state")
    obs[95] = g("vx")
    obs[96] = g("vyaw")
    obs[97] = g("control_mode")
    age = _get(prop, "feature_age_ms", -1)
    obs[98] = F32(-1.0 if age is None or float(age) < 0 else float(age) / 1000.0)
    obs[99] = F32(1.0 if int(_get(prop, "gaze_danger", 0)) else 0.0)

    # 100..103 涓婁竴姝ュ姩浣滃弽棣堬紙鍥轰欢 s_prev_* 宸叉槸宸ョ▼鍗曚綅锛岄櫎浠ュ悇鑷殑婊￠噺绋嬶級
    head_deg, vyaw_dps, vx_mps, mode = _prev_fields(prev)
    obs[100] = F32(head_deg / FEEDBACK_SCALE["head"])
    obs[101] = F32(vyaw_dps / FEEDBACK_SCALE["vyaw"])
    obs[102] = F32(mode / FEEDBACK_SCALE["mode"])
    obs[103] = F32(vx_mps / FEEDBACK_SCALE["vx"])

    # 104..105 浣庨鏃堕挓
    ph = 2.0 * math.pi * float(t_s) / 10.0
    obs[104] = F32(math.sin(ph))
    obs[105] = F32(math.cos(ph))

    if input_dim < OBS_DIM:
        obs[int(input_dim):] = 0.0
    return np.ascontiguousarray(obs[:int(input_dim)], dtype=F32)


# --------------------------------------------------------------------------- #
# 鍩熷瀹堟姢 + 褰掍竴鍖栵紙澶嶅埢 cfc_model.c:935 鐨?forward 绗?1 姝ワ級
# --------------------------------------------------------------------------- #
def normalize_obs(obs, mean, std, clip_sigma=ARB_DEFAULTS["clip_sigma"]):
    """鍥轰欢褰掍竴鍖栫殑鍚戦噺鍖栫増鏈細``(鈥? input_dim)`` 鈫?``(x, clip_ratio)``銆?
    閫愬厓绱犱笌 ``cfc_model_forward()`` 绗竴姝ュ悓鏋勶紙闈炴湁闄愬€?-> 鍧囧€?-> 澶瑰彇 -> 褰掍竴鍖栵級锛?    鍙槸鎶?Python 寰幆鎹㈡垚 numpy銆傝缁冩椂鐢ㄥ畠锛岄獙鏀舵椂鐢ㄥ畠锛屼袱杈瑰彛寰勪竴鑷淬€?
    Args:
        obs: ``(input_dim,)`` 鎴?``(N, input_dim)``銆?        mean: ``(input_dim,)``銆?        std: ``(input_dim,)``锛屽繀椤诲叏 > 0銆?        clip_sigma: 澶瑰彇鍊嶆暟锛堝浐浠跺彧鍦?(0.5, 16) 鍐呮帴鍙楋級銆?
    Returns:
        ``(x_normalized, clip_ratio)``锛沗`x`` 涓?``obs`` 鍚屽舰鐘讹紝float32銆?    """
    o = np.asarray(obs, dtype=F32)
    mu = np.asarray(mean, dtype=F32).reshape(-1)
    sd = np.asarray(std, dtype=F32).reshape(-1)
    if o.shape[-1] != mu.shape[0]:
        raise ValueError("瑙傛祴鏈€鍚庝竴缁?%d != 瀹堟姢缁村害 %d" % (o.shape[-1], mu.shape[0]))
    if not (sd > 0).all():
        raise ValueError("std 蹇呴』鍏?> 0锛堝浐浠惰姹?sd >= 1e-6锛屽惁鍒欐嫆缁濇ā鍨嬶級")
    lo = mu - F32(clip_sigma) * sd
    hi = mu + F32(clip_sigma) * sd
    bad = ~np.isfinite(o)
    v = np.where(bad, mu, o)
    clipped = bad | (v < lo) | (v > hi)
    v = np.clip(v, lo, hi)
    x = ((v - mu) / sd).astype(F32, copy=False)
    ratio = float(np.count_nonzero(clipped)) / float(o.size)
    return x, ratio


class DomainGuard:
    """鍥轰欢 ``cfc_model_forward()`` 鐨勫煙澶栧畧鎶わ紝閫愪綅鍚屾瀯銆?
    鍥轰欢鍙ｅ緞锛坄`cfc_model.c:936-951``锛?:

        v = obs[j]
        if (!isfinite(v)) v = mu
        v = clip(v, mu - sigma*sd, mu + sigma*sd)
        x[j] = (v - mu) / sd

    Args:
        mean: ``(input_dim,)`` 璁粌闆嗗潎鍊硷紙float32锛夈€?        std: ``(input_dim,)`` 璁粌闆嗘爣鍑嗗樊锛坒loat32锛屽繀椤诲凡鍔犱笅闄愶級銆?        clip_sigma: 鍩熷瀹堟姢鍊嶆暟锛屽浐浠跺彧鍦?(0.5, 16) 鍐呮帴鍙楋紝榛樿 4.0銆?
    Note:
        ``clip_ratio`` 鐨勫彛寰勪笌鍥轰欢 ``clip_ratio()`` 涓€鑷达細**闈炴湁闄愬€间篃绠楄 clip**銆?    """

    def __init__(self, mean, std, clip_sigma=ARB_DEFAULTS["clip_sigma"]):
        self.mean = np.ascontiguousarray(mean, dtype=F32).reshape(-1)
        self.std = np.ascontiguousarray(std, dtype=F32).reshape(-1)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean/std 褰㈢姸涓嶄竴鑷? %s vs %s"
                             % (self.mean.shape, self.std.shape))
        self.clip_sigma = float(clip_sigma)
        if not (self.std > 0).all():
            raise ValueError("std 蹇呴』鍏?> 0锛堝浐浠惰姹?sd >= 1e-6锛屽惁鍒欐嫆缁濇ā鍨嬶級")

    @property
    def input_dim(self):
        return int(self.mean.shape[0])

    def apply(self, obs):
        """杩斿洖 ``(x_normalized, clip_ratio)``銆?"""
        return normalize_obs(obs, self.mean, self.std, self.clip_sigma)


def fit_stats(obs_matrix, std_floor=1e-2):
    """鍦ㄨ缁冮泦涓婄粺璁?``input_mean`` / ``input_std``銆?
    Args:
        obs_matrix: ``(N, input_dim)`` 鍘熷瑙傛祴锛?*鍙敤璁粌闆?*锛屼笉鑳界敤楠岃瘉闆嗭級銆?        std_floor: ``std`` 鐨勪笅闄愩€傚浐浠?``blob_parse`` 閬囧埌 ``sd < 1e-6`` 鐩存帴
            "bad input statistics" 鎷掔粷锛涜缁冮噷鎭掑畾鐨勭淮锛堝浐瀹?0 鐨?76/77銆?            甯搁噺鐨?robot_state/control_mode锛夊繀椤婚潬杩欎釜涓嬮檺鍏滀綇銆?
    Returns:
        ``(mean, std)``锛屽潎涓?``(input_dim,) float32``銆?    """
    a = np.asarray(obs_matrix, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("obs_matrix 蹇呴』鏄?(N, input_dim)锛屾敹鍒?%s" % (a.shape,))
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    # 闈炴湁闄愬€间笉璇ュ嚭鐜帮紱鐪熷嚭鐜颁簡灏辨寜 0 鍧囧€?涓嬮檺 std 澶勭悊锛屽埆鎶?NaN 甯﹁繘 blob
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 0.0)
    std = np.maximum(std, float(std_floor))
    return (np.ascontiguousarray(mean, dtype=F32),
            np.ascontiguousarray(std, dtype=F32))


def clip_report(obs_batch, guard):
    """瀵逛竴鎵硅娴嬬粺璁℃瘡涓垎缁勭殑 clip 姣斾緥锛屾柟渚垮畾浣?鍝竴璺暟鎹湪瓒婄晫"銆?"""
    a = np.asarray(obs_batch, dtype=F32)
    _x, ratio = guard.apply(a[0])
    out = {"total": ratio}
    for name, start, count in OBS_GROUPS:
        sub = a[:, start:start + count]
        mu = guard.mean[start:start + count]
        sd = guard.std[start:start + count]
        lo = mu - F32(guard.clip_sigma) * sd
        hi = mu + F32(guard.clip_sigma) * sd
        bad = ~np.isfinite(sub)
        v = np.where(bad, mu, sub)
        clipped = bad | (v < lo) | (v > hi)
        out[name] = float(np.count_nonzero(clipped)) / float(sub.size)
    return out


# --------------------------------------------------------------------------- #
# CLI锛氭墦鍗?106 缁翠笅鏍囪〃锛堜笌鍥轰欢閫愯瀵圭収鐢級+ 鑷
# --------------------------------------------------------------------------- #
def _self_test():
    """鍑犳潯涓嶄緷璧?torch 鐨勪竴鑷存€ц嚜妫€銆?"""
    import numpy as np

    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               ("  " + detail) if detail else ""))

    ck("涓嬫爣琛ㄩ暱搴?106", len(OBS_NAMES) == OBS_DIM)
    ck("鍒嗙粍瑕嗙洊 0..105 涓斾笉閲嶅彔",
       sorted([i for _n, s, c in OBS_GROUPS for i in range(s, s + c)]) == list(range(OBS_DIM)))
    perc = {"feat_64": np.zeros(FEAT_DIM, F32), "theta_img": 12.0, "person_w": 0.3,
            "person_h": 0.5, "person_conf": 0.9, "seen": 1, "hand_up_L": 0,
            "hand_up_R": 1, "wave_sin": 0.6, "wave_cos": 0.8, "wave_amp": 0.35,
            "gesture_conf": 0.4}
    prop = {"roll": 0, "pitch": 0, "pit_gyro": 0, "dq": 0, "yaw": 5.0, "q_head": 5.0,
            "target_head_pos": 5.0, "wheel1_vel": 0.1, "wheel2_vel": 0.1, "wheel_x": 0.2,
            "stable_pos": 0.2, "stable_yaw": 0.0, "robot_state": 1, "vx": 0.1,
            "vyaw": 0.0, "control_mode": 2, "feature_age_ms": 0, "gaze_danger": 0}
    prev = {"head_des": 45.0, "vyaw_des": -90.0, "vx_des": 0.30, "mode": 3}
    o = assemble(perc, prop, prev, t_s=2.5, dist_est=1.0)
    ck("assemble 杈撳嚭 (106,) float32", o.shape == (OBS_DIM,) and o.dtype == F32)
    ck("obs[64]=theta_img", o[64] == F32(12.0))
    ck("obs[67]=dist_est", o[67] == F32(1.0))
    ck("obs[74] 透传 wave_amp（固件 cfc_task.c:372 不重算）",
       abs(float(o[74]) - 0.35) < 1e-6, "%.6f" % o[74])
    ck("obs[76..77] 鍥轰欢鍐欐 0", o[76] == 0.0 and o[77] == 0.0)
    ck("obs[92]=yaw-q_head", abs(float(o[92])) < 1e-6)
    ck("obs[100..103] 鍙嶉褰掍竴鍖",
       o[100] == F32(1.0) and o[101] == F32(-1.0) and o[103] == F32(1.0))
    ck("obs[104..105] 鏃堕挓", abs(float(o[104]) - np.sin(2 * np.pi * 0.25)) < 1e-6)
    d = DistEstimator()
    v0 = float(d.step(1, 0.9))
    ck("璺濈浣庨€氾細棣栧抚涓嶈 dist_target 鎶㈣蛋", abs(v0 - 1.0) < 0.2, "%.3f" % v0)
    d.reset()
    v_lost = float(d.step(0, 0.0))
    ck("璺濈浣庨€氾細娌＄湅瑙佷汉鏃朵繚鎸", abs(v_lost - 0.9) < 0.2, "%.3f" % v_lost)
    mean, std = fit_stats(np.zeros((10, OBS_DIM), F32))
    ck("fit_stats 缁?std 涓嬮檺", float(std.min()) >= 0.0099, "min=%.3f" % std.min())
    xn, r = DomainGuard(mean, std, 4.0).apply(o)
    ck("DomainGuard 杈撳嚭鏈夐檺涓斾笉瓒呰繃 clip_sigma",
       np.isfinite(xn).all() and float(np.abs(xn).max()) <= 4.0 + 1e-6,
       "max=%.3f clip=%.3f" % (np.abs(xn).max(), r))
    big = o.copy()
    big[64] = 1e6
    _xn, r2 = DomainGuard(mean, std, 4.0).apply(big)
    ck("DomainGuard 缁熻 clip 姣斾緥", r2 > 0.0, "%.4f" % r2)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="106 缁磋娴嬬粍瑁呭櫒锛氭墦鍗颁笅鏍囪〃锛堜笌鍥轰欢 cfc_task.c 瀵圭収锛夋垨鍋氳嚜妫€銆",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--layout", action="store_true", help="鎵撳嵃 106 涓笅鏍囩殑鍚嶅瓧涓庡垎缁")
    ap.add_argument("--self-test", action="store_true", help="璺戜笉渚濊禆 torch 鐨勪竴鑷存€ц嚜妫€")
    args = ap.parse_args(argv)
    if args.layout or not args.self_test:
        print("涓嬫爣  鍚嶅瓧                 鍒嗙粍")
        group_of = {}
        for name, start, count in OBS_GROUPS:
            for i in range(start, start + count):
                group_of[i] = name
        for i, name in enumerate(OBS_NAMES):
            print("%4d  %-20s %s" % (i, name, group_of.get(i, "")))
        print("\n鍒嗙粍: " + "  ".join("%s[%d:%d]" % (n, s, s + c)
                                    for n, s, c in OBS_GROUPS))
        print("鍙嶉褰掍竴鍖栧洜瀛? %s" % FEEDBACK_SCALE)
        print("鍥轰欢鍐欐涓?0 鐨勭淮: %s锛堝鍑烘椂 input_std 蹇呴』鍔犱笅闄愶級"
              % list(CONSTANT_OBS_DIMS))
    if args.self_test or not args.layout:
        print("\n鑷:")
        return _self_test()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

