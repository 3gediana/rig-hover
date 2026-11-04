"""ls_admin.py — RIG-Hover 的 Label Studio 账号/项目管理工具。

复用 D:\\smartcar\\labelstudio 的安装（同一套 label-studio 包），但数据目录是
D:\\robot_work\\labelstudio\\data，端口 3334 —— 和智能车那个实例（3333）完全隔离，
共用一个安装只是省一次 pip install。

用法：
    venv\\Scripts\\python.exe ls_admin.py user 邮箱 用户名 密码 [--superuser]
    venv\\Scripts\\python.exe ls_admin.py project [--title ...]
    venv\\Scripts\\python.exe ls_admin.py token 邮箱

账号相关操作直接连数据库（服务停不停都能跑），项目操作走 HTTP API。
"""
import os
import sys

os.environ.setdefault("LABEL_STUDIO_BASE_DATA_DIR", "D:/robot_work/labelstudio/data")
os.environ.setdefault("DJANGO_TELEMETRY", "0")

import label_studio  # noqa: E402

sys.path.insert(0, os.path.dirname(label_studio.__file__))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")

import django  # noqa: E402

django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from organizations.models import Organization  # noqa: E402
from rest_framework.authtoken.models import Token  # noqa: E402

BASE = os.environ.get("LS_URL", "http://127.0.0.1:3334")

#: 场景 v2 只需要三类框。``hand`` 是张开/放松的手，``fist`` 是握紧的拳：
#: 少了 ``hand`` 这一类，模型就无从知道"这是手但不是拳"，会把所有手都判成拳。
#: ACK 场景（wave / pointing / thumbs_up）不在这里，v2 用不到。
LABELS = ["face", "fist", "hand"]

LABEL_CONFIG = f"""<View>
  <Image name="image" value="$image" zoom="true" zoomBy="2" maxWidth="100%"/>
  <RectangleLabels name="label" toName="image">
    <Label value="face" background="#FF0000" hotkey="1"/>
    <Label value="fist" background="#00FF00" hotkey="2"/>
    <Label value="hand" background="#0000FF" hotkey="3"/>
  </RectangleLabels>
  <View>
    <Header value="场景 v2：只标三类 —— face(1) / fist(2) / hand(3)"/>
    <Text name="hint" value="拳头要框到手指握紧的那一团；张开的手框 hand。
      脸只框脸，不要带脖子。240×240 很小，可以按住 Ctrl 滚轮放大。"/>
  </View>
</View>"""


def ensure_org_membership(user):
    """LS 1.22 登录要求用户有组织成员身份，纯 ORM 建用户会漏。

    同时打开 ``jwt.legacy_api_tokens_enabled``：1.23 默认关掉传统 Token 认证，
    不打开的话所有 ``Authorization: Token xxx`` 请求都返回 401
    （"legacy token authentication has been disabled for this organization"），
    而预标注/导出脚本全靠它。
    """
    if Organization.objects.exists():
        org = Organization.objects.first()
    else:
        org = Organization.create_organization(created_by=user, title="RIG-Hover")
    org.add_user(user)
    user.active_organization = org
    user.save(update_fields=["active_organization"])
    try:
        # 用 apps.get_model 而不是 import：jwt_auth 的 app_label 由 Django 注册表
        # 决定，直接 import 模型类会报 "doesn't declare an explicit app_label"。
        jwt_cls = django.apps.apps.get_model("jwt_auth", "JWTSettings")
        jwt, _ = jwt_cls.objects.get_or_create(organization_id=org.id)
        if not jwt.legacy_api_tokens_enabled:
            jwt.legacy_api_tokens_enabled = True
            jwt.save(update_fields=["legacy_api_tokens_enabled"])
    except Exception as exc:  # noqa: BLE001 - 老版本没有这个模型时不该挡住建号
        print(f"[warn] 无法开启 legacy token 认证: {exc}")
    return org


def create_user(email, username, password, is_superuser=False):
    email = email.lower()
    U = get_user_model()
    u, created = U.objects.get_or_create(email=email, defaults={"username": username})
    u.username = username
    u.set_password(password)
    u.is_active = True
    u.is_superuser = u.is_staff = bool(is_superuser)
    u.save()
    org = ensure_org_membership(u)
    token, _ = Token.objects.get_or_create(user=u)
    return u, created, token, org


def api(method, path, token=None, **kw):
    import requests
    s = requests.Session()
    s.trust_env = False          # 本机有 HTTP_PROXY，绕开
    if token:
        s.headers["Authorization"] = f"Token {token}"
    return s.request(method, f"{BASE}{path}", timeout=30, **kw)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]

    if cmd == "user":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]
        superuser = "--superuser" in sys.argv
        if len(args) != 3:
            print(__doc__)
            return 1
        email, username, password = args
        u, created, token, org = create_user(email, username, password, superuser)
        print(f"OK created={created} user={u.username} email={u.email} "
              f"org={org.title} superuser={u.is_superuser}\ntoken={token.key}")
        return 0

    if cmd == "token":
        if len(sys.argv) != 3:
            print(__doc__)
            return 1
        u = get_user_model().objects.get(email=sys.argv[2].lower())
        token, _ = Token.objects.get_or_create(user=u)
        print(token.key)
        return 0

    if cmd == "project":
        import json
        from pathlib import Path
        tok = Path("D:/robot_work/labelstudio/.ls_token")
        if not tok.is_file():
            print("缺 .ls_token，先跑 user 子命令建号")
            return 1
        token = tok.read_text(encoding="utf-8").strip()
        r = api("GET", "/api/projects", token)
        if r.status_code != 200:
            print(f"读项目列表失败 HTTP {r.status_code}: {r.text[:200]}")
            return 1
        for p in r.json().get("results", []):
            if p.get("label_config", "").find("RectangleLabels") >= 0 and \
                    all(f'value="{l}"' in p.get("label_config", "") for l in LABELS):
                print(f"已存在项目 id={p['id']} title={p['title']}")
                return 0
        body = {
            "title": "RIG-Hover 场景v2 face/fist/hand",
            "description": "桌面场景 240×240 灰度。只标 face / fist / hand 三类。",
            "label_config": LABEL_CONFIG,
            "expert_instruction": (
                "拳头框到握紧的那一团；张开或放松的手框 hand；脸只框脸。"
                "Ctrl+滚轮放大后再框，240×240 很小。"),
            "show_instruction": True,
            "show_skip_button": True,
            "enable_empty_annotation": True,
        }
        r = api("POST", "/api/projects", token, json=body)
        if r.status_code not in (200, 201):
            print(f"建项目失败 HTTP {r.status_code}: {r.text[:300]}")
            return 1
        p = r.json()
        print(f"created project id={p['id']} title={p['title']}")
        print(json.dumps({"id": p["id"], "labels": LABELS}, ensure_ascii=False))
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
