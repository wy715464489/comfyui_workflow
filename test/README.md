# Test 脚本说明

所有验证脚本放在此目录，运行前确保 ComfyUI 已启动（`http://127.0.0.1:8188`）。

## 快速运行

```bash
# 在项目根目录执行
cd /Users/zero/data/comfyui_workflow

# 1. API 连通性 + 模型/节点检查（最快，无需等待生成）
python test/test_api.py

# 2. 角色立绘工作流验证（头部 XL + 全身 XL，约 3-4 分钟）
python test/test_character.py

# 3. 场景工作流验证（室外 + 室内 + 视差三层，约 8-10 分钟）
python test/test_scene.py
```

## 脚本说明

| 文件 | 验证内容 | 预计耗时 |
|---|---|---|
| `test_api.py` | ComfyUI 连通、模型加载、节点注册 | <5s |
| `test_character.py` | 头部立绘 XL（1024×1024）、全身立绘 XL（832×1216） | ~3-4 分钟 |
| `test_scene.py` | 室外场景、室内/地牢场景、视差三层（共 5 张图） | ~8-10 分钟 |

## 新增工作流时

新建 `test/test_{模块}.py`，按照以下模板：

1. 导入 `json, urllib.request, time, os, PIL.Image`（标准库 + PIL，不引入 requests）
2. 定义 `build_api_prompt(wf_ui)` → 转换 UI 格式到 API 格式
3. 定义 `submit_workflow(wf_path)` → 提交并返回 prompt_id
4. 定义 `wait_for_result(prompt_id, label, timeout)` → 等待完成并打印 `✅`/`❌`/`⏰`
5. `if __name__ == "__main__"` 中先检查 ComfyUI 连通，再依次提交验证

详见 `test_scene.py` 作为参考实现。
