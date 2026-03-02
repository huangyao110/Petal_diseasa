"""
简化的PyInstaller打包脚本
避免复杂的依赖检查问题
"""

import os
import sys
import subprocess

def simple_build():
    """简单的打包过程"""
    print("=" * 50)
    print("简化打包脚本")
    print("=" * 50)
    
    # 直接使用PyInstaller命令
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onedir',  # 生成目录版本
        '--windowed',  # 无控制台窗口
        '--name=PetalSpot',
        '--add-data=configs;configs',  # 添加配置文件
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=PIL',
        '--clean',
        'app.py'
    ]
    
    print("执行打包命令...")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("✅ 打包成功完成!")
        print(f"输出目录: {os.path.abspath('dist/PetalSpot')}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 打包失败: {e}")
        return False

if __name__ == "__main__":
    success = simple_build()
    if success:
        print("\n🎉 打包完成！可执行文件在 dist/PetalSpot/ 目录中")
    else:
        print("\n❌ 打包失败，请检查错误信息")
    
    input("\n按回车键退出...")