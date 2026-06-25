"""查看缓存文件对应的参数信息"""

import os
import json
from pathlib import Path
import numpy as np
from datetime import datetime

def analyze_cache():
    cache_dir = Path(".cache")
    if not cache_dir.exists():
        print("缓存目录不存在")
        return
    
    cache_files = sorted(cache_dir.glob("*.npz"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not cache_files:
        print("没有找到缓存文件")
        return
    
    print(f"\n找到 {len(cache_files)} 个缓存文件：\n")
    
    for i, cache_file in enumerate(cache_files, 1):
        try:
            # 读取缓存文件
            data = np.load(cache_file)
            vertices = data.get('vertices', np.array([]))
            faces = data.get('faces', np.array([]))
            
            # 文件信息
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            mtime = cache_file.stat().st_mtime
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{'='*80}")
            print(f"[{i}] {cache_file.name}")
            print(f"{'='*80}")
            print(f"  文件大小: {size_mb:.2f} MB")
            print(f"  修改时间: {mtime_str}")
            print(f"  顶点数:   {len(vertices):,}")
            print(f"  面数:     {len(faces):,}")
            
            # 读取参数元数据
            json_path = cache_file.with_suffix('.json')
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    print(f"\n  📋 参数信息:")
                    if 'parameters' in metadata:
                        params = metadata['parameters']
                        for key, value in params.items():
                            if isinstance(value, (list, dict)):
                                print(f"    {key}: {json.dumps(value, ensure_ascii=False)}")
                            else:
                                print(f"    {key}: {value}")
                except Exception as e:
                    print(f"  ⚠️  参数文件读取失败: {e}")
            else:
                print(f"  ⚠️  无参数记录（旧缓存文件）")
            
            print()
            
        except Exception as e:
            print(f"❌ {cache_file.name} 读取失败: {e}\n")
    
    print("=" * 80)
    print("\n💡 提示：")
    print("  - 新生成的缓存会自动保存参数到 .json 文件")
    print("  - 旧缓存文件没有参数记录，可以删除后重新生成")
    print("  - 在程序界面点击 '文件 -> 清空缓存' 可以清理所有缓存")

if __name__ == "__main__":
    analyze_cache()
