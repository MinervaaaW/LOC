import pkg_resources
import sys

def check_requirements(requirements_file="requirements.txt"):
    print("🔍 正在检查当前环境与 requirements.txt 匹配情况...\n")

    # 1. 读取 requirements.txt
    required = {}
    try:
        with open(requirements_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"❌ 未找到 {requirements_file}")
        return

    # 2. 解析包名 + 版本要求
    for line in lines:
        try:
            req = pkg_resources.Requirement.parse(line)
            required[req.name] = req
        except Exception:
            pass

    # 3. 检查当前环境
    errors = []
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    for pkg_name, req in required.items():
        pkg_key = pkg_name.lower()
        if pkg_key not in installed_packages:
            errors.append(f"❌ 缺失包：{pkg_name}（需要：{req}）")
            continue

        current_ver = installed_packages[pkg_key]
        if current_ver not in req:
            errors.append(f"⚠️  版本不匹配：{pkg_name} 当前={current_ver}，需要={req}")

    # 4. 输出结果
    if not errors:
        print("✅ 所有依赖完全匹配！")
    else:
        print("==================== 不匹配项 ====================")
        for err in errors:
            print(err)

if __name__ == "__main__":
    check_requirements()