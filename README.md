深度学习yolov8模型 处理训练工业类电焊缺陷数据集 识别检测焊接焊缝缺陷检测数据集 建立基于深度学习的yolov8焊接焊点焊缝缺陷检测
@[toc]
以下文字及代码仅供参考。

焊接焊缝缺陷检测数据集

1400 5类 缺陷
相邻缺陷：凝固的小液滴，局部熔化
完整性缺陷：异物，小孔，焊穿，凸起，焊接产生废料
几何缺陷：凹陷，熔合不够
处理后缺陷：加工后损伤，加工后留下异物
未焊接好缺陷：未焊透，未熔合大面积

### 🔧 焊接焊缝缺陷检测数据集分类（ 5 类）

| 主类别编号 | 主类别名称         | 子类缺陷类型                                                                 | 示例说明 |
|------------|--------------------|-----------------------------------------------------------------------------|----------|
| 1          | 相邻缺陷           | - 凝固的小液滴<br>- 局部熔化                                               | 焊接过程中飞溅、局部过热造成的不规则附着或熔融状态 |
| 2          | 完整性缺陷         | - 异物<br>- 小孔<br>- 焊穿<br>- 凸起<br>- 焊接产生废料                     | 表示焊缝结构不完整，存在空洞、杂质或多余材料 |
| 3          | 几何缺陷           | - 凹陷<br>- 熔合不够                                                       | 形状尺寸异常，如焊缝凹陷、两侧未充分融合 |
| 4          | 处理后缺陷         | - 加工后损伤<br>- 加工后留下异物                                           | 焊后处理不当导致的新缺陷或残留物 |
| 5          | 未焊接好缺陷       | - 未焊透<br>- 未熔合大面积                                                 | 焊接未完成，母材未完全融合，影响强度和密封性 |

---




![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fe1a76e9817b465b9902f9b0abf72e21.png)

yolo格式txt标签，yolo
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c74e7eabf5464546b2106c0a900fcfaa.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6f9df30d5008427b9cdee22ba892f446.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d3cee5b35bd414d92cdbf3bad4043d1.png)
YOLOv8训练焊接焊缝缺陷检测数据集，基于训练好的权重进行推理。从环境准备、数据组织、模型训练到推理的完整流程。

### 一、环境准备

首先确保安装了必要的Python包：

```bash
pip install ultralytics opencv-python matplotlib numpy
```

### 二、数据组织

确保你的数据集按照以下结构组织：

```
welding_defect_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

- `images/train/` 和 `labels/train/` 分别存放训练图像及其对应的标签文件。
- `images/val/` 和 `labels/val/` 分别存放验证图像及其对应的标签文件。

每个图像都需要有一个对应的`.txt`文件，格式为：`class_id center_x center_y width height`，所有坐标归一化到[0,1]。

### 三、创建 `data.yaml` 文件

在项目根目录下创建 `data.yaml` 文件，内容如下：

```yaml
train: ./welding_defect_dataset/images/train
val: ./welding_defect_dataset/images/val

nc: 5
names: ['Adjacent_Defect', 'Integrity_Defect', 'Geometric_Defect', 'Post_Process_Defect', 'Unwelded_Defect']
```

### 四、模型训练

使用YOLOv8进行模型训练的代码示例如下：

```python
from ultralytics import YOLO

# 加载YOLOv8模型架构配置或预训练模型
model = YOLO('yolov8s.yaml')  # 根据需求选择不同大小的模型如'n', 'm', 'l', 'x'

# 开始训练
results = model.train(
    data='./welding_defect_dataset/data.yaml',
    epochs=100,  # 可以根据需要调整epoch数
    imgsz=640,
    batch=16,
    name='welding_defect_train',
    project='runs/welding_defect',
    exist_ok=True,
    device=0 if torch.cuda.is_available() else 'cpu'
)

# 保存最佳权重
best_weights_path = model.save('./welding_defect_best.pt')
print(f"Best weights saved to {best_weights_path}")
```

### 五、模型验证与性能评估

训练完成后，可以使用以下代码来评估模型在验证集上的表现：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('./welding_defect_best.pt')

# 在验证集上评估
metrics = model.val()

# 打印关键指标
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"Recall: {metrics.box.recall.mean():.4f}")
print(f"Precision: {metrics.box.precision.mean():.4f}")
```

### 六、单图推理 + 结果可视化

下面是如何对单张图片进行推理并显示结果：

```python
from ultralytics import YOLO
import cv2

def detect_and_show(image_path):
    model = YOLO('./welding_defect_best.pt')

    results = model.predict(source=image_path, conf=0.25)
    
    for result in results:
        im_array = result.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detection Result", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 测试一张图片
detect_and_show('path/to/test_image.jpg')
```

### 七、批量推理并保存结果

如果需要对一批图片进行推理并将结果保存：

```python
from ultralytics import YOLO

model = YOLO('./welding_defect_best.pt')

# 批量处理指定文件夹下的图片，并保存结果
model.predict(
    source='./welding_defect_dataset/images/val',
    save=True,
    project='runs/welding_defect/predictions',
    name='val_results',
    conf=0.25
)
```


### 一、加载训练好的模型

首先，已经完成了模型的训练，并保存了最佳权重（例如 `welding_defect_best.pt`）。接下来同学，使用这些权重来初始化模型并进行推理。

```python
from ultralytics import YOLO

# 加载训练好的YOLOv8模型
model = YOLO('path/to/welding_defect_best.pt')
```

### 二、推理与结果可视化

#### 单张图片推理

下面是如何对单张图片进行推理并显示结果：

```python
import cv2

def detect_and_show(image_path):
    # 使用模型进行预测
    results = model.predict(source=image_path, conf=0.25)
    
    for result in results:
        # 将结果绘制成图像
        im_array = result.plot()  # 绘制边界框和标签
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)  # 转换颜色空间以便于显示
        cv2.imshow("Detection Result", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 测试一张图片
detect_and_show('path/to/test_image.jpg')
```

#### 批量图片推理并保存结果

想要批量处理图片并将结果保存下来：

```python
# 批量处理指定文件夹下的图片，并保存结果
model.predict(
    source='path/to/images/folder',
    save=True,
    project='runs/detect/predictions',  # 结果保存路径
    name='batch_results',
    conf=0.25
)
```

### 三、构建用户界面

为了使系统更加用户友好，可以考虑构建一个简单的图形用户界面（GUI），让用户能够轻松上传图片或视频进行检测。这里以PyQt5为例：

```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
import cv2
from PyQt5.QtCore import Qt

class WeldingDefectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Welding Defect Detection System")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                    "Images (*.png *.xpm *.jpg)", options=options)
        if file_name:
            self.detect_and_display(file_name)

    def detect_and_display(self, image_path):
        results = model.predict(source=image_path, conf=0.25)
        for result in results:
            im_array = result.plot()
            im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
            height, width, channel = im.shape
            bytes_per_line = 3 * width
            q_img = QPixmap.fromImage(QImage(im.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped())
            self.image_label.setPixmap(q_img.scaled(self.image_label.size(), aspectRatioMode=Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = WeldingDefectDetectionApp()
    mainWin.show()
    sys.exit(app.exec_())
```

### 四、Web应用部署

这个系统作为Web服务提供给用户，可以使用Flask框架来快速搭建一个API接口。

```python
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = YOLO('path/to/welding_defect_best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    results = model.predict(source=img, conf=0.25)
    for result in results:
        im_array = result.plot()
        im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        
        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return jsonify({"status": "success", "result": base64.b64encode(img_byte_arr).decode('utf-8')}), 200

if __name__ == '__main__':
    app.run(debug=True)
```


