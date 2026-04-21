from ultralytics import YOLO


def predict_video():
    # 1. 加载训练好的模型（最佳权重文件）
    model = YOLO(r'best.pt')  # 替换为你的最佳权重路径（如 runs/detect/car_person_model/weights/best.pt）

    # 2. 预测视频并保存结果
    results = model.predict(
        source='mp4/3e1f993c2a404fc08ca4e395a196abaa.mp4',  # 视频文件路径（或摄像头ID，如0）
        save=True,  # 保存带有检测框的视频
        imgsz=640,  # 图像输入尺寸（与训练时一致）
        conf=0.25,  # 置信度阈值（过滤低置信度预测）
        device=0,  # 使用GPU（0表示第一个GPU，或'cpu'）
        project='runs',  # 结果保存主目录
        name='video_predictions'  # 结果子目录名称（避免覆盖）
    )

    # 3. 打印预测结果（可选）
    for result in results:
        print(f"视频帧数: {len(result.boxes)}")  # 输出每帧检测到的对象数量

    return results


if __name__ == "__main__":
    predict_video()
