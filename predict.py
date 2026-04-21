from ultralytics import YOLO

def predict_and_save():
    # 1. 加载训练好的模型（最佳权重文件）
    model = YOLO(r'best.pt')  # 替换为你的最佳权重路径（如 runs/detect/car_person_model/weights/best.pt）

    # 2. 预测并保存结果（指定保存目录）
    results = model.predict(
        source='my_dataset/images/val',  # 验证集图片路径（或单个图片路径）
        save=True,  # 保存带有检测框的图片
        imgsz=640,  # 图像输入尺寸（与训练时一致）
        conf=0.25,  # 置信度阈值（过滤低置信度预测）
        device=0,  # 使用GPU（0表示第一个GPU，或'cpu'）
        project='runs',  # 主保存目录（与训练时的project一致）
        name='detect/car_person_model_val'  # 子目录名称（匹配你的目标路径）
    )

    # 3. 打印预测结果（可选）
    for result in results:
        print(f"检测到的对象数量: {len(result.boxes)}")

    return results


if __name__ == "__main__":
    predict_and_save()
