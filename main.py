from ultralytics import YOLO
import os

# loading model

model = YOLO("yolov8n.pt")


def main():
    print("Initiating training on e-waste vs waste dataset")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data.yaml", 
        epochs=100, 
        imgsz=640,
        device='cpu',
        batch=16
        )

    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")




if __name__ == "__main__":
    main()
