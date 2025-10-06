# 电商虚拟试穿平台 Coze 工作流搭建文档

## 项目概述

本文档详细说明如何在 Coze 平台上搭建一个电商虚拟试穿平台工作流，实现用户上传身高体重和全身照片，智能生成360度虚拟试穿视频的完整功能。

## 技术架构

### 核心功能模块
- **输入处理**：用户数据验证和预处理
- **AI识别**：人体姿态识别和体型分析
- **3D建模**：基于体型参数的3D人体建模
- **虚拟试穿**：服装与人体模型的智能适配
- **视频渲染**：360度试穿效果视频生成

### 技术栈要求
- **AI模型**：MediaPipe/OpenPose (人体识别)
- **3D引擎**：Ready Player Me / MetaHuman Creator
- **渲染服务**：Blender Render Farm / Unity Cloud Build
- **视频处理**：FFmpeg / OpenCV

## 工作流搭建指南

### 第一阶段：数据接收与预处理

#### 节点1：开始节点配置
```
节点名称：用户输入接收
节点类型：Start
```

**参数配置：**
1. 点击"添加输入参数"
2. 配置输入字段：
   - `user_height` (数字类型) - 用户身高，单位cm，范围100-250
   - `user_weight` (数字类型) - 用户体重，单位kg，范围30-200
   - `user_photo` (文件类型) - 用户全身照片，支持jpg/png格式
   - `clothing_id` (文本类型) - 选择的服装商品ID
   - `clothing_image` (文件类型) - 服装商品图片
   - `clothing_size` (文本类型) - 服装尺码信息

**验证规则：**
```json
{
  "user_height": {
    "type": "number",
    "min": 100,
    "max": 250,
    "required": true
  },
  "user_weight": {
    "type": "number",
    "min": 30,
    "max": 200,
    "required": true
  },
  "user_photo": {
    "type": "file",
    "formats": ["jpg", "jpeg", "png"],
    "maxSize": "10MB",
    "required": true
  }
}
```

#### 节点2：输入验证节点
```
节点名称：参数完整性检查
节点类型：Condition
```

**条件设置：**
1. 创建条件分支：
   - **条件1**：`user_height >= 100 AND user_height <= 250`
   - **条件2**：`user_weight >= 30 AND user_weight <= 200`
   - **条件3**：`user_photo.size <= 10MB`
   - **条件4**：`user_photo.format IN ['jpg','jpeg','png']`

**分支逻辑：**
- 所有条件满足 → 继续下一节点
- 任一条件不满足 → 转至错误处理节点

```javascript
// 验证逻辑代码
if (user_height < 100 || user_height > 250) {
  return {error: "身高范围应在100-250cm之间"};
}
if (user_weight < 30 || user_weight > 200) {
  return {error: "体重范围应在30-200kg之间"};
}
if (!['jpg','jpeg','png'].includes(user_photo.format)) {
  return {error: "照片格式仅支持JPG/PNG"};
}
// 验证通过，继续流程
return {status: "validated", data: input_data};
```

### 第二阶段：AI识别与分析

#### 节点3：人体姿态识别
```
节点名称：人体关键点检测
节点类型：AI Plugin
```

**插件配置：**
1. 选择插件：`人体姿态识别` (MediaPipe Plugin)
2. 模型设置：
   - 模型版本：MediaPipe Pose v1.3
   - 置信度阈值：0.8
   - 检测精度：高精度模式

**输入配置：**
```json
{
  "image": "{{user_photo}}",
  "detection_confidence": 0.8,
  "tracking_confidence": 0.8,
  "model_complexity": 2,
  "enable_segmentation": true
}
```

**输出处理：**
```javascript
// 提取关键点坐标
const keypoints = response.pose_landmarks;
const key_measurements = {
  shoulder_width: calculate_distance(keypoints[11], keypoints[12]),
  torso_length: calculate_distance(keypoints[11], keypoints[23]),
  arm_length: calculate_distance(keypoints[11], keypoints[15]),
  leg_length: calculate_distance(keypoints[23], keypoints[27]),
  hip_width: calculate_distance(keypoints[23], keypoints[24])
};
```

#### 节点4：体型参数计算
```
节点名称：人体尺寸分析
节点类型：Code Execution
```

**代码实现：**
```python
import numpy as np
import cv2

def calculate_body_measurements(keypoints, height, weight):
    """
    基于关键点和身高体重计算详细体型参数
    """
    # BMI计算
    bmi = weight / ((height/100) ** 2)

    # 像素到实际尺寸的转换比例
    pixel_to_cm_ratio = height / calculate_body_height_in_pixels(keypoints)

    # 关键尺寸计算 (单位：cm)
    measurements = {
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "shoulder_width": get_shoulder_width(keypoints) * pixel_to_cm_ratio,
        "chest_circumference": estimate_chest_circumference(keypoints, pixel_to_cm_ratio),
        "waist_circumference": estimate_waist_circumference(keypoints, pixel_to_cm_ratio, bmi),
        "hip_circumference": estimate_hip_circumference(keypoints, pixel_to_cm_ratio),
        "arm_length": get_arm_length(keypoints) * pixel_to_cm_ratio,
        "leg_length": get_leg_length(keypoints) * pixel_to_cm_ratio,
        "neck_circumference": estimate_neck_circumference(bmi, height),
        "body_proportions": calculate_body_proportions(keypoints)
    }

    return measurements

def estimate_chest_circumference(keypoints, ratio):
    """基于肩宽和体型估算胸围"""
    shoulder_width = get_shoulder_width(keypoints) * ratio
    # 使用人体学公式估算
    chest_circumference = shoulder_width * 2.1  # 经验公式
    return chest_circumference

def estimate_waist_circumference(keypoints, ratio, bmi):
    """基于BMI和体型比例估算腰围"""
    hip_width = get_hip_width(keypoints) * ratio
    # BMI影响因子
    bmi_factor = 0.8 + (bmi - 22) * 0.02
    waist_circumference = hip_width * 0.8 * bmi_factor
    return waist_circumference
```

#### 节点5：服装特征提取
```
节点名称：服装图像分析
节点类型：AI Plugin
```

**插件配置：**
1. 选择插件：`服装识别与分割`
2. 模型设置：
   - 分割模型：DeepFashion2-Segmentation
   - 分类模型：FashionNet v2.0

**配置参数：**
```json
{
  "input_image": "{{clothing_image}}",
  "tasks": ["segmentation", "classification", "attribute_detection"],
  "output_format": "detailed",
  "categories": ["tops", "bottoms", "dresses", "outerwear", "accessories"]
}
```

**特征提取逻辑：**
```javascript
// 服装特征分析
const clothing_analysis = {
  category: response.category,           // 服装类别
  color_palette: response.colors,       // 主要颜色
  material_type: response.material,     // 材质类型
  style_attributes: response.attributes, // 风格属性
  size_dimensions: response.dimensions,  // 尺寸信息
  fit_type: response.fit_style,         // 版型（修身/宽松/标准）
  segmentation_mask: response.mask      // 分割掩码
};
```

### 第三阶段：3D建模与适配

#### 节点6：3D人体建模
```
节点名称：3D人体模型生成
节点类型：HTTP Request
```

**API配置：**
1. 请求方法：POST
2. API地址：`https://api.readyplayer.me/v1/avatars`
3. 认证方式：Bearer Token

**请求体配置：**
```json
{
  "bodyType": "fullbody",
  "textureAtlas": "1024",
  "morphTargets": "ARKit",
  "customizations": {
    "height": "{{measurements.height}}",
    "bodyShape": {
      "weight": "{{measurements.weight}}",
      "muscleMass": "{{calculated_muscle_mass}}",
      "bodyFat": "{{calculated_body_fat}}"
    },
    "bodyMeasurements": {
      "shoulderWidth": "{{measurements.shoulder_width}}",
      "chestCircumference": "{{measurements.chest_circumference}}",
      "waistCircumference": "{{measurements.waist_circumference}}",
      "hipCircumference": "{{measurements.hip_circumference}}"
    }
  }
}
```

**响应处理：**
```javascript
// 处理3D模型响应
const avatar_data = {
  model_url: response.modelUrl,
  texture_url: response.textureUrl,
  skeleton_data: response.skeleton,
  morph_targets: response.morphTargets,
  model_id: response.id,
  generation_status: response.status
};

// 验证模型生成状态
if (avatar_data.generation_status !== "completed") {
  throw new Error("3D模型生成失败");
}
```

#### 节点7：服装3D适配与建模
```
节点名称：服装3D模型适配
节点类型：AI Plugin
```

**插件配置：**
1. 选择插件：`3D服装建模与适配`
2. 物理引擎：Cloth Simulation Engine

**适配参数：**
```json
{
  "base_avatar": "{{avatar_data.model_url}}",
  "clothing_image": "{{clothing_image}}",
  "clothing_category": "{{clothing_analysis.category}}",
  "fit_preferences": {
    "fit_type": "{{clothing_analysis.fit_type}}",
    "size_adjustment": "auto",
    "stretch_factor": 0.95,
    "wrinkle_simulation": true
  },
  "material_properties": {
    "fabric_type": "{{clothing_analysis.material_type}}",
    "elasticity": "auto",
    "thickness": "medium",
    "drape_coefficient": "auto"
  },
  "physics_settings": {
    "gravity": 9.81,
    "wind_resistance": false,
    "collision_detection": true,
    "self_collision": true
  }
}
```

**服装适配逻辑：**
```python
def fit_clothing_to_avatar(avatar_measurements, clothing_specs):
    """
    根据人体尺寸调整服装模型
    """
    # 尺寸映射计算
    size_mapping = {
        "chest": avatar_measurements["chest_circumference"],
        "waist": avatar_measurements["waist_circumference"],
        "hip": avatar_measurements["hip_circumference"],
        "shoulder": avatar_measurements["shoulder_width"]
    }

    # 服装缩放因子计算
    scale_factors = calculate_scale_factors(size_mapping, clothing_specs)

    # 应用物理约束
    constraints = {
        "stretch_limit": 1.2,  # 最大拉伸20%
        "compression_limit": 0.8,  # 最大压缩20%
        "shape_preservation": 0.9  # 保持90%原始形状
    }

    # 生成适配后的3D服装模型
    fitted_clothing = apply_fitting_algorithm(
        clothing_specs,
        scale_factors,
        constraints
    )

    return fitted_clothing
```

### 第四阶段：渲染与生成

#### 节点8：360度视频渲染
```
节点名称：虚拟试穿视频渲染
节点类型：HTTP Request
```

**渲染服务配置：**
1. 渲染引擎：Blender Render Farm API
2. API地址：`https://api.renderservice.com/v2/360render`

**渲染参数：**
```json
{
  "scene_setup": {
    "avatar_model": "{{avatar_data.model_url}}",
    "clothing_models": ["{{fitted_clothing.model_url}}"],
    "environment": "studio_lighting",
    "background": "neutral_gradient"
  },
  "camera_settings": {
    "rotation_type": "360_horizontal",
    "rotation_speed": 30,  // 每秒30度
    "camera_distance": 2.5,  // 2.5米距离
    "camera_height": "chest_level",
    "tracking_target": "center_of_mass"
  },
  "render_quality": {
    "resolution": "1920x1080",
    "frame_rate": 30,
    "duration": 12,  // 12秒视频
    "anti_aliasing": "4x_msaa",
    "lighting_quality": "high",
    "shadow_quality": "high"
  },
  "animation_settings": {
    "pose_variation": true,
    "clothing_physics": true,
    "facial_expression": "neutral_friendly",
    "breathing_animation": true
  },
  "output_format": {
    "codec": "h264",
    "bitrate": "8000kbps",
    "color_space": "sRGB"
  }
}
```

**渲染状态监控：**
```javascript
// 渲染进度监控
async function monitor_render_progress(render_job_id) {
  let status = "processing";
  let progress = 0;

  while (status !== "completed" && status !== "failed") {
    await sleep(5000); // 每5秒检查一次

    const response = await fetch(`/api/render/${render_job_id}/status`);
    const data = await response.json();

    status = data.status;
    progress = data.progress_percentage;

    console.log(`渲染进度: ${progress}%`);

    if (status === "failed") {
      throw new Error(`渲染失败: ${data.error_message}`);
    }
  }

  return data.output_urls;
}
```

#### 节点9：渲染质量检查
```
节点名称：视频质量验证
节点类型：Condition
```

**质量检查条件：**
```javascript
// 视频质量检查逻辑
function validate_video_quality(video_url, expected_specs) {
  const quality_metrics = analyze_video(video_url);

  const checks = {
    resolution_check: quality_metrics.resolution === expected_specs.resolution,
    duration_check: Math.abs(quality_metrics.duration - expected_specs.duration) < 1,
    framerate_check: quality_metrics.fps >= expected_specs.min_fps,
    filesize_check: quality_metrics.filesize < expected_specs.max_filesize,
    corruption_check: !quality_metrics.has_corruption,
    clothing_visibility: quality_metrics.clothing_detection_confidence > 0.9
  };

  const passed_checks = Object.values(checks).filter(Boolean).length;
  const total_checks = Object.keys(checks).length;
  const quality_score = passed_checks / total_checks;

  return {
    passed: quality_score >= 0.8,
    score: quality_score,
    failed_checks: Object.keys(checks).filter(key => !checks[key])
  };
}
```

**分支条件设置：**
- 质量分数 ≥ 0.8 → 继续到输出节点
- 质量分数 < 0.8 → 返回重新渲染或错误处理

### 第五阶段：输出与后处理

#### 节点10：视频后处理优化
```
节点名称：视频质量优化
节点类型：Code Execution
```

**优化处理代码：**
```python
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def enhance_video_quality(input_video_path):
    """
    视频质量优化处理
    """
    # 读取视频
    cap = cv2.VideoCapture(input_video_path)

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频设置
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'enhanced_' + input_video_path
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 图像增强处理
        enhanced_frame = enhance_frame(frame)
        out.write(enhanced_frame)

    cap.release()
    out.release()

    # 音频处理（如果需要）
    finalize_video_with_audio(output_path)

    return output_path

def enhance_frame(frame):
    """
    单帧图像增强
    """
    # 1. 锐化处理
    kernel_sharpen = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
    sharpened = cv2.filter2D(frame, -1, kernel_sharpen)

    # 2. 颜色增强
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] * 1.1  # 增加饱和度
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. 亮度对比度调整
    alpha = 1.1  # 对比度
    beta = 10    # 亮度
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    return enhanced

def compress_video_for_web(input_path, target_size_mb=50):
    """
    Web优化压缩
    """
    clip = VideoFileClip(input_path)

    # 计算目标比特率
    duration = clip.duration
    target_bitrate = f"{int((target_size_mb * 8 * 1024) / duration)}k"

    # 压缩输出
    compressed_path = input_path.replace('.mp4', '_compressed.mp4')
    clip.write_videofile(
        compressed_path,
        bitrate=target_bitrate,
        codec='libx264',
        audio_codec='aac'
    )

    return compressed_path
```

#### 节点11：生成预览图
```
节点名称：多角度预览图生成
节点类型：Code Execution
```

**预览图生成逻辑：**
```python
def generate_preview_images(video_path, angles=[0, 45, 90, 135, 180, 225, 270, 315]):
    """
    从360度视频中提取不同角度的预览图
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    preview_images = []

    for angle in angles:
        # 计算对应帧位置
        frame_position = int((angle / 360) * total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

        ret, frame = cap.read()
        if ret:
            # 图像质量优化
            enhanced_frame = enhance_frame(frame)

            # 保存预览图
            preview_path = f'preview_{angle}deg.jpg'
            cv2.imwrite(preview_path, enhanced_frame,
                       [cv2.IMWRITE_JPEG_QUALITY, 95])

            preview_images.append({
                'angle': angle,
                'image_path': preview_path,
                'thumbnail_path': create_thumbnail(preview_path)
            })

    cap.release()
    return preview_images

def create_thumbnail(image_path, size=(300, 400)):
    """
    创建缩略图
    """
    image = cv2.imread(image_path)
    thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    thumbnail_path = image_path.replace('.jpg', '_thumb.jpg')
    cv2.imwrite(thumbnail_path, thumbnail,
               [cv2.IMWRITE_JPEG_QUALITY, 80])

    return thumbnail_path
```

#### 节点12：结果输出
```
节点名称：最终结果输出
节点类型：End
```

**输出数据结构：**
```json
{
  "success": true,
  "processing_time": "45.2s",
  "results": {
    "main_video": {
      "url": "https://cdn.example.com/videos/tryon_360_12345.mp4",
      "duration": 12,
      "resolution": "1920x1080",
      "filesize": "48.5MB",
      "quality_score": 0.92
    },
    "preview_images": [
      {
        "angle": 0,
        "image_url": "https://cdn.example.com/previews/front_12345.jpg",
        "thumbnail_url": "https://cdn.example.com/thumbs/front_12345.jpg"
      },
      {
        "angle": 90,
        "image_url": "https://cdn.example.com/previews/side_12345.jpg",
        "thumbnail_url": "https://cdn.example.com/thumbs/side_12345.jpg"
      },
      {
        "angle": 180,
        "image_url": "https://cdn.example.com/previews/back_12345.jpg",
        "thumbnail_url": "https://cdn.example.com/thumbs/back_12345.jpg"
      }
    ],
    "metadata": {
      "user_measurements": {
        "height": 170,
        "weight": 65,
        "chest_circumference": 88,
        "waist_circumference": 74,
        "hip_circumference": 92
      },
      "clothing_info": {
        "id": "clothing_12345",
        "category": "dress",
        "size": "M",
        "color": "navy_blue",
        "material": "cotton_blend"
      },
      "processing_details": {
        "pose_detection_confidence": 0.94,
        "fitting_accuracy": 0.89,
        "render_quality": 0.92
      }
    }
  }
}
```

## 错误处理与异常情况

### 常见错误处理节点

#### 错误处理节点：输入验证失败
```
节点名称：输入错误处理
节点类型：End (Error)
```

**错误响应格式：**
```json
{
  "success": false,
  "error_code": "INVALID_INPUT",
  "error_message": "用户输入参数不符合要求",
  "details": {
    "height_error": "身高必须在100-250cm范围内",
    "photo_error": "照片格式仅支持JPG/PNG，大小不超过10MB"
  },
  "retry_suggestions": [
    "请检查身高体重数值是否正确",
    "请上传清晰的正面全身照",
    "确保照片格式为JPG或PNG"
  ]
}
```

#### 错误处理节点：AI识别失败
```
节点名称：AI识别错误处理
节点类型：Condition + End
```

**处理逻辑：**
```javascript
// AI识别失败处理
if (pose_detection_confidence < 0.7) {
  return {
    success: false,
    error_code: "POSE_DETECTION_FAILED",
    error_message: "无法准确识别人体姿态",
    suggestions: [
      "请确保照片中人物姿态清晰可见",
      "建议采用正面站立姿势",
      "确保光线充足，背景简洁"
    ]
  };
}
```

#### 错误处理节点：渲染服务异常
```
节点名称：渲染服务错误处理
节点类型：HTTP Request + Condition
```

**重试机制：**
```javascript
async function render_with_retry(render_params, max_retries = 3) {
  for (let attempt = 1; attempt <= max_retries; attempt++) {
    try {
      const result = await call_render_service(render_params);
      if (result.status === "completed") {
        return result;
      }
    } catch (error) {
      console.log(`渲染尝试 ${attempt} 失败: ${error.message}`);

      if (attempt === max_retries) {
        throw new Error("渲染服务多次尝试后仍然失败");
      }

      // 等待后重试
      await sleep(5000 * attempt);
    }
  }
}
```

## 性能优化建议

### 1. 并行处理优化
- 人体识别和服装分析可以并行执行
- 3D建模和纹理生成可以异步处理
- 多个角度的预览图可以批量生成

### 2. 缓存策略
- 相同体型参数的3D模型可以缓存复用
- 常用服装的3D模型可以预先生成
- 渲染结果可以根据参数哈希进行缓存

### 3. 资源管理
- 设置合理的并发处理限制
- 实施任务队列管理
- 定期清理临时文件和缓存

## 部署与测试

### 工作流测试用例

#### 测试用例1：标准用例
```json
{
  "test_name": "标准身材女性试穿连衣裙",
  "input": {
    "user_height": 165,
    "user_weight": 55,
    "user_photo": "test_photos/standard_female.jpg",
    "clothing_id": "dress_001",
    "clothing_image": "clothing/dress_navy_m.jpg"
  },
  "expected_output": {
    "success": true,
    "video_duration": 12,
    "quality_score": "> 0.8"
  }
}
```

#### 测试用例2：边界值测试
```json
{
  "test_name": "最小身高体重边界测试",
  "input": {
    "user_height": 100,
    "user_weight": 30,
    "user_photo": "test_photos/min_size.jpg",
    "clothing_id": "top_001"
  },
  "expected_output": {
    "success": true,
    "notes": "测试系统对极端体型的处理能力"
  }
}
```

### 部署清单

1. **前置依赖检查**
   - [ ] Coze平台账号和权限
   - [ ] AI插件授权（人体识别、服装分析）
   - [ ] 3D建模API密钥
   - [ ] 渲染服务账号
   - [ ] CDN存储配置

2. **工作流部署步骤**
   - [ ] 创建新的Coze工作流项目
   - [ ] 按顺序添加所有节点
   - [ ] 配置节点间的连接关系
   - [ ] 设置错误处理分支
   - [ ] 配置环境变量和密钥

3. **测试验证**
   - [ ] 单元节点功能测试
   - [ ] 端到端流程测试
   - [ ] 性能压力测试
   - [ ] 错误场景测试

4. **监控告警配置**
   - [ ] 处理时间监控
   - [ ] 成功率监控
   - [ ] 资源使用监控
   - [ ] 错误率告警

## 总结

本文档详细描述了在Coze平台搭建电商虚拟试穿平台的完整工作流程。通过合理的节点设计和插件配置，可以实现从用户输入到最终360度试穿视频生成的全自动化处理。

关键成功要素：
- 准确的人体姿态识别和尺寸计算
- 高质量的3D建模和服装适配算法
- 稳定的渲染服务和质量保证机制
- 完善的错误处理和重试机制

建议在实际部署时，先进行小规模测试，逐步优化各个环节的参数和算法，确保系统稳定性和用户体验。