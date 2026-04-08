import gymnasium as gym
import mediapy as media
import numpy as np

# 1. 取得環境 (優先從 model 中取得)
if hasattr(model, 'env') and model.env is not None:
    # Correctly assign render_env to the VecFrameStack object (model.env)
    render_env = model.env
else:
    render_env = env

print("正在重置環境並檢查渲染器...")
# 修正：不要將整個 obs 轉為 np.array，因為它是個字典
reset_output = render_env.reset()
if isinstance(reset_output, tuple):
    obs, info = reset_output
else:
    obs = reset_output
    info = {}

# 2. 測試渲染功能
first_frame = render_env.render()
if first_frame is None or np.max(first_frame) == 0:
    print("警告：偵測到渲染輸出為空或黑畫面。請確保 %env MUJOCO_GL=osmesa 已設定。")
else:
    print("渲染器檢查正常。")

frames = []
num_steps = 500

for i in range(num_steps):
    # 修正：直接傳入 obs 字典，SB3 會自行處理其中的 Tensor 轉換
    action, _ = model.predict(obs, deterministic=True)

    # 執行物理步進
    step_output = render_env.step(action)
    if isinstance(step_output, tuple) and len(step_output) == 5:
        obs, reward, done, truncated, info = step_output
    else:
        # Fallback for unexpected number of return values from step
        # Assuming (obs, reward, done, info) or similar, trying to be robust
        obs, reward, done, info = step_output[:4] if len(step_output) >= 4 else (step_output[0], 0, False, {})
        truncated = False # Default to False if not explicitly returned

    # 捕獲當前畫面
    frame = render_env.render()
    if frame is not None:
        frames.append(frame)

    if done or truncated:
        reset_output = render_env.reset()
        if isinstance(reset_output, tuple):
            obs, info = reset_output
        else:
            obs = reset_output
            info = {}

# 3. 顯示結果
if len(frames) > 0:
    print(f"成功生成 {len(frames)} 幀影片。")
    media.show_video(frames, fps=30)
else:
    print("錯誤：未能捕捉到任何有效幀。")
