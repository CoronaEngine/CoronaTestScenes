import numpy as np
import OpenImageIO as oiio
import matplotlib.pyplot as plt
import argparse
import sys
import os

def load_exr(filename):
    """使用 OpenImageIO 读取 EXR 文件并转换为三维 Numpy 数组 (H, W, 3)"""
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)
        
    inp = oiio.ImageInput.open(filename)
    if not inp:
        print(f"Error: Could not open {filename}: {oiio.geterror()}")
        sys.exit(1)
        
    spec = inp.spec()
    # 读取所有通道，强制转换为 float32 精度
    data = inp.read_image(format=oiio.FLOAT)
    inp.close()
    
    # 将形状转换为 (height, width, channels)
    data = data.reshape((spec.height, spec.width, spec.nchannels))
    
    # 如果有 Alpha 通道，只取 RGB (前 3 个通道) 进行颜色比对
    if spec.nchannels >= 3:
        return data[:, :, :3]
    else:
        # 如果是单通道（如深度图），复制成三通道以便统一处理
        return np.repeat(data, 3, axis=2)

def calculate_metrics(img_ref, img_test):
    """计算图像序列间的 MSE 和 PSNR"""
    # 确保尺寸一致
    if img_ref.shape != img_test.shape:
        print(f"Error: Image dimensions mismatch! Ref: {img_ref.shape}, Test: {img_test.shape}")
        sys.exit(1)
        
    # 计算均方误差 (MSE)
    mse = np.mean((img_ref - img_test) ** 2)
    
    if mse == 0:
        return 0, float('inf') # 完全一致
        
    # 计算峰值信噪比 (PSNR)
    # 对于 HDR 图像，峰值通常设定为 1.0 (假设数据已归一化到 0-1) 
    # 或者使用数据中的最大值。这里假设最大值为 1.0 用于比对。
    max_pixel = 1.0 
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return mse, psnr

def generate_diff_map(img_ref, img_test, output_diff, threshold=0.001):
    """生成差异热力图，并标出超过阈值的区域"""
    # 计算逐像素的绝对差异
    diff = np.abs(img_ref - img_test)
    
    # 将 RGB 差异合并为单通道亮度差异 (感知加权)
    diff_map = 0.299 * diff[:,:,0] + 0.587 * diff[:,:,1] + 0.114 * diff[:,:,2]
    
    # 找出超过阈值的像素点
    bad_pixels = diff_map > threshold
    num_bad_pixels = np.sum(bad_pixels)
    total_pixels = diff_map.size
    bad_ratio = (num_bad_pixels / total_pixels) * 100
    
    # 使用 Matplotlib 生成热力图
    plt.figure(figsize=(12, 8))
    
    # 显示参考图
    plt.subplot(2, 2, 1)
    plt.imshow(np.clip(img_ref, 0, 1)) # 裁剪到 0-1 用于显示
    plt.title("Reference (Clipped)")
    plt.axis('off')
    
    # 显示测试图
    plt.subplot(2, 2, 2)
    plt.imshow(np.clip(img_test, 0, 1))
    plt.title("Test (Clipped)")
    plt.axis('off')
    
    # 显示热力图 (使用 'jet' 或 'inferno' 颜色映射)
    plt.subplot(2, 2, 3)
    # 设置显示上限，以便突出微小差异。这里设置最大显示 0.1 的差异。
    plt.imshow(diff_map, cmap='jet', vmax=0.1) 
    plt.colorbar(label='Absolute Difference')
    plt.title("Difference Heatmap (vmax=0.1)")
    plt.axis('off')
    
    # 显示二值化的失败区域图（红色标出失败点）
    plt.subplot(2, 2, 4)
    # 创建黑色背景
    mask = np.zeros((diff_map.shape[0], diff_map.shape[1], 3))
    # 将失败点设为红色
    mask[bad_pixels] = [1, 0, 0] 
    plt.imshow(mask)
    plt.title(f"Fail Mask (Red: > {threshold})\n{num_bad_pixels}/{total_pixels} pixels ({bad_ratio:.2f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_diff)
    plt.close()
    
    print(f"Diff map saved to: {output_diff}")
    return num_bad_pixels

def main():
    parser = argparse.ArgumentParser(description="Automated OpenEXR Regression Test Tool")
    parser.add_argument("reference", help="Path to the reference (golden) EXR image")
    parser.add_argument("test", help="Path to the test output EXR image")
    parser.add_argument("--mse-threshold", type=float, default=1e-5, help="Threshold for Mean Squared Error to pass the test")
    parser.add_argument("--pixel-threshold", type=float, default=0.001, help="Threshold for absolute pixel difference for heatmap generation")
    parser.add_argument("--output-diff", default="diff.png", help="Path to save the generated difference heatmap image")
    
    args = parser.parse_args()
    
    print(f"--------------------------------------------------")
    print(f"EXR Regression Test")
    print(f"Ref:  {args.reference}")
    print(f"Test: {args.test}")
    print(f"--------------------------------------------------")
    
    # 1. 加载图像
    img_ref = load_exr(args.reference)
    img_test = load_exr(args.test)
    
    # 2. 计算整体指标
    mse, psnr = calculate_metrics(img_ref, img_test)
    print(f"MSE  (Mean Squared Error): {mse:.8f}")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB")
    
    # 3. 生成差异热力图并标出失败像素
    num_bad_pixels = generate_diff_map(img_ref, img_test, args.output_diff, args.pixel_threshold)
    
    # 4. 执行断言 (决定测试是否通过)
    if mse < args.mse_threshold:
        print(f"--------------------------------------------------")
        print(f"RESULT: PASSED")
        print(f"--------------------------------------------------")
        sys.exit(0) # 退出码 0 表示成功
    else:
        print(f"--------------------------------------------------")
        print(f"RESULT: FAILED (MSE {mse:.8f} >= Threshold {args.mse_threshold})")
        print(f"--------------------------------------------------")
        sys.exit(1) # 退出码 1 表示失败

if __name__ == "__main__":
    main()