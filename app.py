import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, Form, Request
from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr
from pydantic import BaseModel, Field, model_validator
from starlette.responses import JSONResponse


# ----------------------------
# 自定义异常和响应
# ----------------------------
class CustomException(Exception):
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class CommonResponse:
    def __init__(self, code: int, message: str, data=None):
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "data": self.data
        }


def success_response(data=None):
    return JSONResponse(content=CommonResponse(code=200, message="success", data=data).to_dict())


def error_response(code=400, message="error", data=None):
    return JSONResponse(content=CommonResponse(code=code, message=message, data=data).to_dict())


# ----------------------------
# 配置和初始化
# ----------------------------
RESULT_FOLDER = 'results'
Path(RESULT_FOLDER).mkdir(parents=True, exist_ok=True)

# Set up logging to log exceptions
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

app = FastAPI(title="OCR API", description="基于 PaddleOCR 的图像识别接口", version="1.0.0", debug=False)
print("FastAPI 初始化完成！")

ocr = PaddleOCR(lang="ch",
                use_gpu=True,
                det_model_dir="D:\model\ch_PP-OCRv4_det_server_infer",
                rec_model_dir="D:\model\ch_PP-OCRv4_rec_server_infer",
                cls_model_dir="D:\model\ch_ppocr_mobile_v2.0_cls_slim_infer",
                det=True,
                cls_batch_num=8,
                use_angle_cls=True,
                use_dilation=True,
                drop_score=0.8,
                use_mp=True,
                total_process_num=16)
print("PaddleOCR 初始化完成！")


# ----------------------------
# 数据模型
# ----------------------------
class ColorGroup(BaseModel):
    r: int = Field(..., ge=0, le=255, description="红色通道值 (0-255)")
    g: int = Field(..., ge=0, le=255, description="绿色通道值 (0-255)")
    b: int = Field(..., ge=0, le=255, description="蓝色通道值 (0-255)")
    threshold: int = Field(..., gt=0, le=255, description="颜色差异阈值")


class OCRRequestModel(BaseModel):
    image: Optional[UploadFile] = Field(None, description="上传的图像文件")
    colorEnhance: bool = Field(False, description="是否启用颜色增强")
    colorGroups: Optional[List[ColorGroup]] = Field(None, description="颜色增强的参数组")
    imageRotate: bool = Field(False, description="是否启用图片旋转")

    @model_validator(mode='before')
    def validate_color_enhance(cls, values):
        if values.get('colorEnhance') and not values.get('colorGroups'):
            raise CustomException("当启用颜色增强时，必须提供 colorGroups 参数", error_code=400)
        if not values.get('image'):
            raise CustomException("图像文件是必须的", error_code=400)
        return values


# ----------------------------
# 工具函数
# ----------------------------
# 基于霍夫变换检测旋转角度
def detect_rotation_angle(image):
    # 将图片转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (11, 11), 16)
    # cv2.imwrite('blurred.png', blurred)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # 使用霍夫变换检测直线，增加阈值，避免过多无关的直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)  # 调低阈值以获取更多线

    # 如果没有检测到直线，返回0度
    if lines is None:
        return 0

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)  # 将角度从弧度转换为度数

        # 只考虑接近水平或垂直的直线
        if 75 < angle < 105:  # 垂直直线 (0° 或 90°)
            angles.append(angle)
        elif 165 < angle < 195:  # 水平直线 (180° 或 0°)
            angles.append(angle - 180)

    # 如果没有符合条件的角度，返回0度
    if len(angles) == 0:
        return 0

    # 使用加权平均方法来计算旋转角度
    weighted_angle = np.median(angles)  # 中位数方法可以减少异常值的影响
    return weighted_angle


async def rotation_image(angle, img_array):
    (h, w) = img_array.shape[:2]
    center = (w // 2, h // 2)
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 计算旋转后图像的边界框大小
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    # 调整旋转矩阵以使图像适应新的边界框
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    # 旋转图像并避免裁剪
    img_array = cv2.warpAffine(img_array, rotation_matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img_array


def save_file(img: np.ndarray, filename: str, folder: str = RESULT_FOLDER) -> Path:
    save_path = Path(folder) / filename
    cv2.imwrite(str(save_path), img)
    return save_path


def save_annotated_image(image: np.ndarray, result, filename: str, folder: str = RESULT_FOLDER) -> Path:
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = [line[0] for group in result for line in group]
        texts = [line[1][0] for group in result for line in group]
        scores = [line[1][1] for group in result for line in group]

        annotated_image = draw_ocr(image_pil, boxes, texts, scores)
        annotated_image = Image.fromarray(annotated_image)

        save_path = Path(folder) / filename
        annotated_image.save(save_path)
        return save_path
    except Exception as e:
        raise CustomException(f"标注图像保存失败: {str(e)}", error_code=500)


def image_color_replace(img: np.ndarray, color_groups: List[ColorGroup]) -> np.ndarray:
    try:
        img_b, img_g, img_r = cv2.split(img)
        final_mask = np.zeros_like(img_b, dtype=bool)

        for group in color_groups:
            diff = np.abs(np.dstack((img_b - group.b, img_g - group.g, img_r - group.r)))
            mask = np.any(diff > group.threshold, axis=-1)
            final_mask |= mask

        img_masked = img.copy()
        img_masked[final_mask] = [255, 255, 255]
        return img_masked
    except Exception as e:
        raise CustomException(f"图像颜色替换失败: {str(e)}", error_code=500)


def run_ocr(image: np.ndarray) -> list:
    if image is None:
        raise ValueError("输入图像为空，无法执行OCR")

    try:
        raw_result = ocr.ocr(image)
        if raw_result is None or (raw_result.__len__() == 1 and raw_result[0] is None):
            raise ValueError("OCR 未返回结果 :" + json.dumps(raw_result))
        return raw_result
    except Exception as e:
        raise CustomException(message=str(e), error_code=500)


def process_ocr_result(raw_result: list) -> dict:
    if not raw_result or not isinstance(raw_result, list):
        return {"text": "", "wordList": []}

    try:
        word_list = [
            {
                "word": line[1][0],
                "confidence": round(line[1][1], 2),
                "positionList": [
                    {"x": line[0][0][0], "y": line[0][0][1]},
                    {"x": line[0][1][0], "y": line[0][1][1]},
                    {"x": line[0][2][0], "y": line[0][2][1]},
                    {"x": line[0][3][0], "y": line[0][3][1]},
                ],
            }
            for group in raw_result for line in group
        ]
        text = " ".join([word["word"] for word in word_list])
        return {"text": text, "wordList": word_list}
    except Exception as e:
        raise CustomException(message=str(e), error_code=500)


# ----------------------------
# API 路由
# ----------------------------
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, CustomException):
        # 如果是自定义异常，按自定义异常处理，避免重复日志
        logging.error(f"自定义异常处理: {exc.message}\n{traceback.format_exc()}")
        return error_response(code=exc.error_code or 500, message=exc.message)
    else:
        # 处理其他标准异常，只有当是非自定义异常时才打印日志
        logging.error(f"未知异常处理: {str(exc)}\n{traceback.format_exc()}")
        return error_response(code=500, message=str(exc))


@app.post('/ocr', summary="OCR 识别接口")
async def ocr_endpoint(
        image: UploadFile,
        colorEnhance: bool = Form(False),
        colorGroups: Optional[str] = Form(None),
        imageRotate: bool = Form(False)
):
    # 解析和验证请求数据
    color_groups = [ColorGroup(**group) for group in eval(colorGroups)] if colorGroups else []
    request_data = OCRRequestModel(image=image, colorEnhance=colorEnhance, colorGroups=color_groups, imageRotate=imageRotate)

    # 读取图像数据
    image_data = await request_data.image.read()
    img_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        raise ValueError("图像加载失败，可能文件格式不支持")

    angle = None
    if request_data.imageRotate:
        # 检测图像的旋转角度
        angle = detect_rotation_angle(img_array)
        print(f"检测到的旋转角度: {angle}°")

        # 旋转图像回正，避免图像被截取
        img_array = await rotation_image(angle, img_array)

    # 执行颜色增强（如果启用）
    if request_data.colorEnhance and request_data.colorGroups:
        img_array = image_color_replace(img_array, request_data.colorGroups)

    # 执行 OCR
    raw_result = run_ocr(img_array)

    # 格式化结果
    ocr_result = process_ocr_result(raw_result)

    # 保存文件
    generated_uuid = str(uuid.uuid4()).replace("-", "")
    generated_uuid = 'test'
    enhanced_img_path = save_file(img_array, f"{generated_uuid}_enhanced.png")
    annotated_img_path = save_annotated_image(img_array, raw_result, f"{generated_uuid}_annotated.png")

    ocr_result['enhancedImgPath'] = str(enhanced_img_path)
    ocr_result['annotatedImgPath'] = str(annotated_img_path)
    ocr_result['imageRotated'] = str(angle)

    return success_response(ocr_result)


# ----------------------------
# 启动服务
# ----------------------------
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

    print("服务启动完成！")
