"""
ComfyUI HTTP / WebSocket API 封装。
使用前确保 ComfyUI 已在本地启动（默认 http://127.0.0.1:8188）。
"""

import json
import time
import uuid
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

COMFYUI_URL = "http://127.0.0.1:8188"


def queue_prompt(workflow: dict, client_id: str = "") -> str:
    """提交工作流到队列，返回 prompt_id。"""
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=payload,
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["prompt_id"]


def get_history(prompt_id: str) -> dict:
    """获取指定 prompt_id 的历史记录（含输出图像信息）。"""
    url = f"{COMFYUI_URL}/history/{urllib.parse.quote(prompt_id)}"
    return json.loads(urllib.request.urlopen(url).read())


def poll_until_done(prompt_id: str, timeout: int = 300, interval: float = 2.0) -> dict:
    """
    轮询直到指定工作流执行完成，返回 history 记录。
    超时则抛出 TimeoutError。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = get_history(prompt_id)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(interval)
    raise TimeoutError(f"工作流 {prompt_id} 在 {timeout}s 内未完成")


def download_outputs(history_record: dict, save_dir: str | Path,
                     prefix: str = "") -> list[Path]:
    """
    从 history 记录中下载所有输出图像到 save_dir。
    返回已保存的文件路径列表。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    outputs = history_record.get("outputs", {})
    for node_id, node_output in outputs.items():
        for img_info in node_output.get("images", []):
            filename = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            img_type = img_info.get("type", "output")

            params = urllib.parse.urlencode({
                "filename": filename,
                "subfolder": subfolder,
                "type": img_type,
            })
            url = f"{COMFYUI_URL}/view?{params}"
            data = urllib.request.urlopen(url).read()

            save_name = f"{prefix}_{filename}" if prefix else filename
            dest = save_dir / save_name
            dest.write_bytes(data)
            saved.append(dest)

    return saved


def run_workflow(workflow: dict, save_dir: str | Path,
                 output_prefix: str = "", timeout: int = 300) -> list[Path]:
    """
    一站式接口：提交工作流 → 等待完成 → 下载输出图像。
    返回已保存的文件路径列表。
    """
    client_id = str(uuid.uuid4())
    prompt_id = queue_prompt(workflow, client_id)
    print(f"[ComfyAPI] 已提交，prompt_id={prompt_id}")
    record = poll_until_done(prompt_id, timeout=timeout)
    paths = download_outputs(record, save_dir, prefix=output_prefix)
    print(f"[ComfyAPI] 完成，已保存 {len(paths)} 张图像到 {save_dir}")
    return paths
