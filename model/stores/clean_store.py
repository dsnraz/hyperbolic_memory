# 读取文件
with open('hierarchical_vector_store.py', 'r') as f:
    lines = f.readlines()

# 删除第70-73 行的重复调用（旧的错误位置）
# 找到"# 从数据库恢复计数器" 第一次出现的位置并删除它及其后面3行
first_restore_idx = None
for i, line in enumerate(lines):
    if '# 从数据库恢复计数器' in line and first_restore_idx is None:
        first_restore_idx = i
        break

if first_restore_idx is not None:
    # 删除这4行（注释 + if + 调用 + 空行）
    del lines[first_restore_idx:first_restore_idx+4]

# 写回文件
with open('hierarchical_vector_store.py', 'w') as f:
    f.writelines(lines)

print("清理完成！")
