from ultralytics import YOLO

def train_yolo_model():
    # 1. 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用Nano模型作为起点
    # 2. 开始训练
    results = model.train(
        data='my_dataset/data.yaml',  # 替换为您的yaml文件路径
        epochs=50,  # 训练轮数
        imgsz=640,  # 输入图像大小
        batch=8,  # 批次大小
        name='car_person_model',  # 训练结果保存名称
        project='runs',  # 结果保存目录
        exist_ok=True,  # 如果目录存在则覆盖
        patience=50,  # 早停 patience
        save_period=1,  # 每1个epoch保存一次模型
        device=0  # 使用GPU（0表示第一个GPU）
    )
    return model

if __name__ == "__main__":
    trained_model = train_yolo_model()
    print("训练完成！")
