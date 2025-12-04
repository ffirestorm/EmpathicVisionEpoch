import os
import cv2
import numpy as np
import time
from collections import deque
from paddlex import create_pipeline

############################
# 1. 参数与配置
############################

# Cityscapes 中可行走地面的类别 id
GROUND_IDS = [0, 1]  # 0: road, 1: sidewalk

# 决策规则参数
TH_GROUND = 0.15        # 地面比例阈值：大于这个值认为“这一块有明显地面”
DELTA_SIDE = 0.10       # 左右地面比例差值，大于此认定为“多出一侧通路”

# 判定稳定事件所需的帧数（这里只处理单张图片，可设为 1）
STABLE_FRAMES = 1

# 输入输出文件夹
DATA_DIR = "data"   # 输入图片目录
RES_DIR = "res"     # 输出结果目录


############################
# 2. PaddleX 模型加载
############################

def load_seg_model(model_dir):
    """
    加载 PaddleX 导出的语义分割推理模型.
    model_dir: 推理模型目录，例如 'inference_model'
    """
    model = create_pipeline(pipeline="semantic_segmentation")
    return model

def get_ground_mask(pred):
    """
    pred: H x W 的类别 id 图 (int)
    返回: H x W 的 bool 数组，True 表示地面 (road/sidewalk)
    """
    ground_mask = np.isin(pred, GROUND_IDS)
    return ground_mask

############################
# 4. 根据 ground_mask 计算决策点事件（单张图）
############################

def detect_outdoor_events(ground_mask,
                          th_ground=TH_GROUND,
                          delta_side=DELTA_SIDE):
    """
    输入: ground_mask (H x W, bool)
    输出: events: list of (type, direction)
       type 可为: "TURN", "CROSS", "T_JUNCTION", "OBSTACLE_CENTER_BYPASSABLE"
       direction: "LEFT", "RIGHT", "AHEAD" 或 None
    """
    events = []

    h, w = ground_mask.shape

    # 下半部分代表视障者脚前区域
    roi = ground_mask[int(h * 0.5):, :]  # 可根据实际情况调整 0.5 的比例

    # 左中右三块
    left = roi[:, :w // 3]
    center = roi[:, w // 3: 2 * w // 3]
    right = roi[:, 2 * w // 3:]

    def ratio(region):
        # region 是 bool 数组, True 表示地面
        return float(np.mean(region))

    gl = ratio(left)
    gc = ratio(center)
    gr = ratio(right)

    # 可选：调参阶段可打印
    # print(f"gl={gl:.2f}, gc={gc:.2f}, gr={gr:.2f}")

    straight = gc > th_ground

    # 1. 右侧出现明显新通路(右侧地面多于左侧且明显大)
    if straight and (gr - gl) > delta_side and gr > th_ground:
        events.append(("TURN", "RIGHT"))

    # 2. 左侧出现明显新通路
    if straight and (gl - gr) > delta_side and gl > th_ground:
        events.append(("TURN", "LEFT"))

    # 3. 十字路口/大路口：三块都有不少地面
    if gc > th_ground and gl > th_ground and gr > th_ground:
        events.append(("CROSS", "AHEAD"))

    # 4. 丁字路口：前方地面少，但两侧多
    if gc < th_ground and gl > th_ground and gr > th_ground:
        events.append(("T_JUNCTION", "STOP_AHEAD"))

    # 5. 中央有障碍，但左右至少一侧可绕行
    if gc < th_ground and (gl > th_ground or gr > th_ground):
        events.append(("OBSTACLE_CENTER_BYPASSABLE", None))

    return events

############################
# 5. 事件转成字符串（用于文件名）
############################

def event_to_tag(event):
    """
    event: (type, direction)
    输出: 例如 'TURN_LEFT', 'CROSS', 'T_JUNCTION', 'OBSTACLE_CENTER_BYPASSABLE'
    """
    etype, edir = event
    if etype == "TURN":
        if edir == "RIGHT":
            return "TURN_RIGHT"
        elif edir == "LEFT":
            return "TURN_LEFT"
        else:
            return "TURN"
    elif etype == "CROSS":
        return "CROSS"
    elif etype == "T_JUNCTION":
        return "T_JUNCTION"
    elif etype == "OBSTACLE_CENTER_BYPASSABLE":
        return "OBSTACLE_CENTER_BYPASSABLE"
    return "UNKNOWN"

def overlay_ground_mask(image, ground_mask, alpha=0.4):
    """
    将地面区域以半透明绿色覆盖到原图上，帮助可视化。
    """
    vis = image.copy()
    green = np.zeros_like(vis, dtype=np.uint8)
    green[:] = (0, 255, 0)
    mask_3c = np.stack([ground_mask] * 3, axis=-1)

    vis[mask_3c] = cv2.addWeighted(
        vis, 1 - alpha, green, alpha, 0
    )[mask_3c]
    return vis

############################
# 7. 针对单张图片处理：推理 + 决策点 + 保存
############################

def process_single_image(model, img_path, save_dir):
    """
    对单张图片进行：
    1) 读取 & 分割
    2) 决策点识别
    3) 若有决策点，则保存到 save_dir，文件名加上决策点类型
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 读取失败: {img_path}")
        return

    output=model.predict(input=img_path, target_size = -1)
    for res in output:
        pred =res.json
    pred =np.array(pred['res']['pred'][0])
    if not isinstance(pred, np.ndarray) or pred.ndim != 2:
        print(f"[ERROR] 预测结果 pred 不是 HxW 的 np.ndarray，请检查 model.predict 的返回格式.")
        return

    # ---------- 2. 获取地面 mask ----------
    ground_mask = get_ground_mask(pred)

    # ---------- 3. 检测决策点 ----------
    events = detect_outdoor_events(ground_mask)

    if not events:
        # 没有任何决策点，就不保存（根据你需求可改为仍然保存）
        print(f"[INFO] {os.path.basename(img_path)} 未识别到决策点，跳过保存")
        return

    # ---------- 4. 叠加可视化（可选） ----------
    vis = overlay_ground_mask(img, ground_mask)

    # 也可以在图像上写上事件文字方便调试
    y0 = 30
    for event in events:
        tag = event_to_tag(event)
        cv2.putText(vis, tag, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)
        y0 += 40

    # ---------- 5. 拼接输出文件名 ----------
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)

    # 可能出现多个事件类型，将它们的 tag 用 '_' 连接起来
    tags = [event_to_tag(e) for e in events]
    tags_str = "_".join(sorted(set(tags)))  # 去重并排序一下，避免重复

    out_name = f"{name}_{tags_str}{ext}"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)

    # ---------- 6. 保存结果图 ----------
    cv2.imwrite(out_path, vis)
    print(f"[SAVE] {out_path}")



def main():
    model_dir = "inference_model"  # TODO: 改成你的推理模型路径
    model = load_seg_model(model_dir)

    os.makedirs(RES_DIR, exist_ok=True)

    # 遍历 data 目录下的所有图片文件
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        # 简单筛选一下常见图片后缀
        ext = os.path.splitext(fname)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        print(f"[PROCESS] {fpath}")
        process_single_image(model, fpath, RES_DIR)


if __name__ == "__main__":
    main()

