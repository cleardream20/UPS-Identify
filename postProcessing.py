import os
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
from osgeo import gdal, ogr, osr
import glob
from PIL import Image
from tqdm import tqdm

from mmseg.apis import init_model, inference_model, show_result_pyplot

Image.MAX_IMAGE_PIXELS = None  # 禁用解压缩炸弹检查


def squares_all_in_one(input_tif_path, config_file, checkpoint_file, output_dir, patch_size=1024, overlap=128):
    """
    仿照uvl论文，一整套完整后处理流程
    参数:
        input_tif_path: 输入tif路径
        config_file: 模型config文件路径
        checkpoint_file: 模型checkpoint/权重/.pth文件路径
        output_dir: 所有文件输出目录
    """
    print("\n=== 开始处理 ===")
    print(f"输入文件: {input_tif_path}")
    print(f"输出目录: {output_dir}")

    # init_model()初始化/设置模型
    print("\n=== 初始化模型 ===")
    # config_file = './mmsegmentation/MyConfigs/UpsDataset_KNet.py'
    # checkpoint_file = './mmsegmentation/work_dirs/UpsDataset-KNet/iter_12500.pth'
    device = 'cuda:0'

    # 检查文件是否存在
    print(f"检查配置文件: {config_file}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    print(f"检查模型文件: {checkpoint_file}")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_file}")

    # 增加详细日志
    print(f"\n正在加载模型...")
    print(f"设备: {device}")
    print(f"配置文件: {os.path.abspath(config_file)}")
    print(f"模型权重: {os.path.abspath(checkpoint_file)}")
    print("这可能需要一些时间，请等待...")

    try:
        model = init_model(config_file, checkpoint_file, device=device)
        print("✓ 模型加载成功!")
    except Exception as e:
        print("✗ 模型加载失败!")
        print(f"错误详情: {str(e)}")
        raise

    # 检查CUDA是否可用
    if 'cuda' in device:
        import torch
        print(f"\nCUDA可用性检查:")
        print(f"Torch CUDA可用: {torch.cuda.is_available()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
        print(f"模型已加载到: {next(model.parameters()).device}")

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # 1. tif to jpg
    jpg_path = os.path.join(temp_dir, "original.jpg")
    convert_tif_to_jpg(input_tif_path, jpg_path)

    # 2. 图像分割
    # 清理之前的patches和pred_patches文件夹
    patch_dir = os.path.join(temp_dir, "patches")
    pred_patch_dir = os.path.join(temp_dir, "pred_patches")

    # 删除patches文件夹（如果存在）
    if os.path.exists(patch_dir):
        print(f"\n删除旧的patches文件夹: {patch_dir}")
        import shutil
        shutil.rmtree(patch_dir)

    # 删除pred_patches文件夹（如果存在）
    if os.path.exists(pred_patch_dir):
        print(f"删除旧的pred_patches文件夹: {pred_patch_dir}")
        shutil.rmtree(pred_patch_dir)

    # 创建新的patches和pred_patches文件夹
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(pred_patch_dir, exist_ok=True)
    print(f"创建新的patches和pred_patches文件夹")

    split_image_to_patches(jpg_path, patch_dir, patch_size, overlap)

    # 3. 切片图像预测结果文件夹创建
    os.makedirs(pred_patch_dir, exist_ok=True)

    # mmseg_predict()一次预测一张
    # 循环，一张一张来
    # 输入和输出保持同名
    for patch_path in tqdm(glob.glob(os.path.join(patch_dir, "*.jpg"))):
        patch_name = os.path.basename(patch_path)
        predicted_patch_path = os.path.join(pred_patch_dir, patch_name)
        mmseg_predict(patch_path, predicted_patch_path, model)

    # 4.merge方法把预测的小结果(.jpg)合成大结果(.tif)
    # 地理坐标信息获取加到这一步来
    merged_tif_path = os.path.join(temp_dir, "merged.tif")
    merge_patches_to_tif(jpg_path, pred_patch_dir, merged_tif_path, patch_size, overlap, input_tif_path, min_area=8420)

    # 5.coordinate_mapping_to_shp方法进行坐标映射和shp转换
    final_shp_path = os.path.join(output_dir, "square.shp")
    coordinate_mapping_to_shp(merged_tif_path, final_shp_path)

    print(f"Completed! Final shp saved to: {final_shp_path}")

def convert_tif_to_jpg(tif_path, jpg_path, scale=0.5):
    """
    tif 转成 jpg
    参数:
        tif_path: 原始tif路径
        jpg_path: 输出jpg路径
        scale: 缩放比例 (0-1)
    """
    with rasterio.open(tif_path) as src:
        new_width = int(src.width * scale)
        new_height = int(src.height * scale)

        # 读取图像数据
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, new_height, new_width),
            resampling=rasterio.enums.Resampling.bilinear
        )

        # 转换为0-255范围并转置为(height, width, channels)
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb / rgb.max() * 255).astype(np.uint8)
        Image.fromarray(rgb).save(jpg_path)


def split_image_to_patches(jpg_path, output_dir, patch_size=1024, overlap=256):
    """
    图像切片 切成patch_size x patch_size
    通过调整边缘块位置实现，不进行图像填充
    参数:
        jpg_path: (tif转换后的)jpg图像路径
        output_dir: 切分后的小块存放的文件夹路径
        patch_size: 要切分的小块的大小(px)
        overlap: 重叠部分的长度/大小(px)
    返回:
        img_width, img_height: 原始图像尺寸
    """
    img = Image.open(jpg_path)
    img_width, img_height = img.size

    stride = patch_size - overlap

    patch_num = 0
    for y in range(0, img_height, stride):
        if y + patch_size > img_height:
            y = img_height - patch_size

        for x in range(0, img_width, stride):
            if x + patch_size > img_width:
                x = img_width - patch_size

            patch = img.crop((x, y, x + patch_size, y + patch_size))

            patch_path = os.path.join(output_dir, f"patch_{x}_{y}.jpg")
            patch.save(patch_path)
            patch_num += 1

    return img_width, img_height


def mmseg_predict(input_patch_path, output_patch_path, model):
    """
    使用训练好的mmseg模型对切片图像进行预测
    参数:
        input_patch_path: 切片图像路径
        output_patch_path: 模型预测结果(pred_patches)保存路径
        model: 训练好的mmseg模型
    """
    img_bgr = cv2.imread(input_patch_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 创建白色区域掩码
    white_mask = np.all(img_rgb == [254, 254, 254], axis=-1)

    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # 将白色区域强制设为背景(0)
    pred_mask[white_mask] = 0

    # 调试输出
    unique, counts = np.unique(pred_mask, return_counts=True)
    print(f"预测结果类别分布: {dict(zip(unique, counts))}")

    output = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    class_type = [4, 8, 9]
    # 4 squares
    # 8 greenland
    # 9 park
    # output[pred_mask == class_type] = [0, 0, 200]
    output[np.isin(pred_mask, class_type)] = [0, 0, 200]

    Image.fromarray(output).save(output_patch_path)


# === 改进版小区域过滤 ===
def filter_small_areas(binary, min_area=8420, min_hole_area=10000, kernel_size=5, debug=True):
    """
    改进版小区域过滤，可以控制空洞填充的面积阈值
    参数:
        binary: 二值图像
        min_area: 最小保留区域面积(像素数)
        min_hole_area: 最小保留空洞面积(小于此值的空洞会被填充)
        kernel_size: 形态学操作核大小
        debug: 是否输出调试信息
    返回:
        处理后的二值图像 + 面积统计信息(当debug=True时)
    """
    # 确保输入格式正确
    if binary.dtype != np.uint8:
        binary = (binary > 0).astype(np.uint8) * 255

    # 形态学优化
    if kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 层次化轮廓分析
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(binary)

    # 面积统计容器
    areas = {
        'all': [],
        'kept': [],
        'removed': [],
        'holes_kept': [],
        'holes_filled': []
    }

    # 第一次遍历：收集所有区域面积
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        areas['all'].append(area)

        if hierarchy[0][i][3] == -1:  # 外部轮廓
            if area >= min_area:
                areas['kept'].append(area)
            else:
                areas['removed'].append(area)
        else:  # 空洞
            if area >= min_hole_area:
                areas['holes_kept'].append(area)
            else:
                areas['holes_filled'].append(area)

    # 第二次遍历：实际绘制
    valid_parents = set(
        i for i, cnt in enumerate(contours)
        if hierarchy[0][i][3] == -1 and
        cv2.contourArea(cnt) >= min_area
    )

    for i, cnt in enumerate(contours):
        parent_idx = hierarchy[0][i][3]
        if parent_idx == -1 and i in valid_parents:
            # 绘制外部轮廓
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
        elif parent_idx in valid_parents:
            # 只绘制大于min_hole_area的空洞
            if cv2.contourArea(cnt) >= min_hole_area:
                cv2.drawContours(filtered, [cnt], -1, 0, -1)  # 空洞用黑色填充

    # 调试输出
    if debug:
        def stats(name, values):
            if not values: return "无"
            return f"数量: {len(values)}, 最小: {min(values):.1f}, 最大: {max(values):.1f}, 平均: {np.mean(values):.1f}"

        print("\n=== 面积统计 ===")
        print(f"[全部区域] {stats('all', areas['all'])}")
        print(f"[保留区域] {stats('kept', areas['kept'])} (≥{min_area})")
        print(f"[移除区域] {stats('removed', areas['removed'])} (<{min_area})")
        print(f"[保留空洞] {stats('holes_kept', areas['holes_kept'])} (≥{min_hole_area})")
        print(f"[填充空洞] {stats('holes_filled', areas['holes_filled'])} (<{min_hole_area})")
        print(f"建议min_area范围: {np.percentile(areas['all'], 70):.1f}-{max(areas['all']) * 0.9:.1f}")

    return filtered

def smooth_boundaries(binary, iterations=1):
    """
    边界平滑处理
    """
    # 使用(5x5)椭圆核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # 先膨胀后腐蚀可以平滑凸起部分
    smoothed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    # 再腐蚀后膨胀可以平滑凹陷部分
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return smoothed

def merge_patches_to_tif(original_jpg_path, predicted_patch_dir, output_tif_path,
                         patch_size=1024, overlap=256, reference_tif_path=None, min_area=100, min_hole_area=10000):
    """
    合并函数，将所有小切片结果合成最终的完整tif文件
    参数:
        original_jpg_path: 原始tif转换生成的jpg图像路径
        predicted_patch_dir: pred_patches所在的文件目录
        output_tif_path: 输出tif路径
        patch_size: 切片大小
        overlap: 重叠大小

        min_area: 小区域过滤阈值(<min_area的会被过滤)(px)
        min_hole_area: 小空洞填充阈值(<min_hole_area的会被填充)(px)
    """
    # 获取原始JPG尺寸
    with Image.open(original_jpg_path) as img:
        img_width, img_height = img.size

    # 获取地理信息（保持不变）
    if reference_tif_path:
        with rasterio.open(reference_tif_path) as src_ref:
            crs = src_ref.crs
            transform = src_ref.transform
            scale_x = img_width / src_ref.width
            scale_y = img_height / src_ref.height
            scaled_transform = rasterio.Affine(
                transform.a / scale_x, transform.b, transform.c,
                transform.d, transform.e / scale_y, transform.f
            )
    else:
        crs = None
        scaled_transform = from_origin(0, 0, 1, 1)

    # 初始化合并矩阵
    merged = np.zeros((img_height, img_width), dtype=np.float32)
    count = np.zeros((img_height, img_width), dtype=np.float32) + 1e-8

    # 合并逻辑
    for patch_path in glob.glob(os.path.join(predicted_patch_dir, "*.jpg")):
        try:
            # 从文件名解析位置
            patch_name = os.path.basename(patch_path)
            parts = patch_name.split('_')
            x = int(parts[1])
            y = int(parts[2].split('.')[0])

            # 读取预测结果
            pred_patch = np.array(Image.open(patch_path))
            squares_mask = np.all(pred_patch == [1, 0, 200], axis=-1).astype(np.float32)

            # 计算实际可写入区域
            h, w = squares_mask.shape
            x_end = min(x + w, img_width)
            y_end = min(y + h, img_height)

            # 确保不会出现空切片
            actual_w = x_end - x
            actual_h = y_end - y

            if actual_w <= 0 or actual_h <= 0:
                print(f"警告：跳过无效位置 patch @ ({x},{y})")
                continue

            # 安全合并（添加形状检查）
            if squares_mask[:actual_h, :actual_w].shape == (y_end - y, x_end - x):
                merged[y:y_end, x:x_end] += squares_mask[:actual_h, :actual_w]
                count[y:y_end, x:x_end] += 1
            else:
                print(f"形状不匹配: patch {patch_name} "
                      f"target={(y_end - y, x_end - x)} "
                      f"source={squares_mask[:actual_h, :actual_w].shape}")

        except Exception as e:
            print(f"处理 {patch_path} 时出错: {str(e)}")
            continue

    # 后续处理
    merged /= count
    merged_smoothed = cv2.blur(merged, (20, 20))
    binary = (merged_smoothed > 0).astype(np.uint8) * 255

    filtered = filter_small_areas(binary, min_area, min_hole_area, kernel_size=5)
    filtered = smooth_boundaries(filtered, iterations=1)
    binary = filtered

    with rasterio.open(
            output_tif_path,
            'w',
            driver='GTiff',
            height=binary.shape[0],
            width=binary.shape[1],
            count=1,
            dtype=binary.dtype,
            crs=crs,
            transform=scaled_transform,
            nodata=0
    ) as dst:
        dst.write(binary, 1)


def coordinate_mapping_to_shp(predicted_tif_path, output_shp_path):
    """
    简化版坐标映射（因输入TIFF已含正确坐标信息）
    输入:
        predicted_tif_path: 已包含正确坐标的预测tiff路径
        output_shp_path: 输出shp路径
    """
    # 临时文件路径
    temp_shp_path = output_shp_path.replace('.shp', '_temp.shp')

    # 1. 打开已地理参考的预测TIFF
    src_ds = gdal.Open(predicted_tif_path)
    if src_ds is None:
        raise RuntimeError(f"无法打开预测TIFF文件: {predicted_tif_path}")

    # 2. 直接读取坐标系统（不再需要reference_tif_path）
    crs_wkt = src_ds.GetProjection()
    band = src_ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    print(f"输入TIFF像素统计: 有效像素={(arr > 0).sum()}/{arr.size}")

    # 3. 多边形化（直接使用原坐标）
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # 清理旧文件
    for ext in ['.shp', '.dbf', '.prj', '.shx']:
        path = output_shp_path.replace('.shp', ext)
        if os.path.exists(path):
            driver.DeleteDataSource(path)

    # 创建输出Shapefile
    out_ds = driver.CreateDataSource(output_shp_path)
    out_layer = out_ds.CreateLayer("squares", srs=osr.SpatialReference(crs_wkt))
    out_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))

    # 执行多边形化
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)

    # 5. 释放资源
    out_ds = None
    src_ds = None

    print(f"生成Shapefile成功: {output_shp_path}")


input_tif = "./mmsegmentation/input_tif/建邺区-19级.tif"
output_dir = "./mmsegmentation/output_shp" # 输出结果和部分中间结果均在此文件夹中
config_file = './mmsegmentation/MyConfigs/UpsDataset_KNet.py'
checkpoint_file = './mmsegmentation/work_dirs/UpsDataset-KNet/iter_12500.pth'

squares_all_in_one(input_tif, config_file, checkpoint_file, output_dir)
