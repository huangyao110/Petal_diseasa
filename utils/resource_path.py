import os
import sys
import logging

def get_resource_path(relative_path):
    """
    获取资源文件的绝对路径，适用于开发环境和打包后的环境
    Args:
        relative_path: 相对于应用程序根目录的路径
    Returns:
        资源文件的绝对路径
    """
    # 判断应用程序是否已打包
    if getattr(sys, 'frozen', False):
        # 如果是打包后的应用程序，使用sys._MEIPASS或sys.executable的目录
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller打包
            base_path = sys._MEIPASS
        else:
            # cx_Freeze等其他打包工具
            base_path = os.path.dirname(sys.executable)
    else:
        # 如果是开发环境，使用当前脚本所在目录
        base_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.join(base_path, relative_path)
    if not os.path.exists(abs_path):
        logging.warning(f"资源文件未找到: {abs_path}")
    return abs_path

# 示例用法
def get_config_path(config_file):
    """
    获取配置文件的绝对路径
    Args:
        config_file: 配置文件名
    Returns:
        配置文件的绝对路径
    """
    return get_resource_path(os.path.join('configs', 'app_config', config_file))

def get_model_path(model_file):
    """
    获取模型文件的绝对路径
    Args:
        model_file: 模型文件名
    Returns:
        模型文件的绝对路径
    """
    return get_resource_path(os.path.join('weights_from_online', model_file))