import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# 在遥感领域，雷达图可用于多光谱或SAR数据的特征对比，如分析不同地物（植被、水体、建筑物）在多个波段或极化通道的反射特性，辅助地物分类和目标识别；
# 在机器学习领域，雷达图可用于模型性能的多指标评估（如精度、召回率、F1分数、训练时间），直观比较不同算法的综合表现。

# Streamlit应用标题
st.title("雷达图可视化")

# 添加应用说明
st.markdown("""
在遥感领域，雷达图可用于多光谱或SAR数据的特征对比，如分析不同地物（植被、水体、建筑物）在多个波段或极化通道的反射特性，辅助地物分类和目标识别；

在机器学习领域，雷达图可用于模型性能的多指标评估（如精度、召回率、F1分数、训练时间），直观比较不同算法的综合表现。
""")

# 创建侧边栏
st.sidebar.title("参数设置")

# 提示用户数据格式
st.markdown("### 数据格式示例")
st.markdown("请上传一个 CSV 文件，格式如下（第一列为组别，每列代表不同指标）：")
example_data = pd.DataFrame({
    '组别': ['组别1', '组别2', '组别3'],
    '指标A': [4, 5, 3],
    '指标B': [4, 5, 4],
    '指标C': [5, 4, 5],
    '指标D': [4, 5, 3],
    '指标E': [3, 2, 5]
})
st.table(example_data)

# 上传CSV文件
uploaded_file = st.file_uploader("上传CSV文件", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("上传的数据：")
    st.write(data)
    
    # 使用用户上传的数据进行绘图
    group_names = data.iloc[:, 0].tolist()  # 第一列为组别名称
    
    # 获取所有指标名称（列名，除了第一列）
    categories = data.columns[1:].tolist()
    
    # 为每个组别准备数据
    groups_data = {}
    # 使用更丰富的配色方案替代简单的颜色列表
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, group in enumerate(group_names):
        # 获取该组别的所有指标数据
        values = data.iloc[i, 1:].tolist()
        # 闭合数据（首尾相连）
        values.append(values[0])
        groups_data[group] = values
    
    # 闭合类别
    categories.append(categories[0])
else:
    # 默认数据
    categories = ['A', 'B', 'C', 'D', 'E']
    categories = [*categories, categories[0]]
    groups_data = {
        'Group_1': [4, 4, 5, 4, 3, 4],
        'Group_2': [5, 5, 4, 5, 2, 5],
        'Group_3': [3, 4, 5, 3, 5, 3]
    }
    # 修改默认颜色为更丰富的配色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 创建输出目录
output_dir = "Figures"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# 全局图形参数选择 - 移到侧边栏
st.sidebar.markdown("### 图形参数设置")

# 雷达图样式选择
radar_styles = ["标准网格", "多边形网格"]
radar_style_choice = st.sidebar.selectbox("雷达图网格样式", radar_styles)

# 添加数据点标记样式选择
marker_styles = {
    "无": "none", 
    "圆点": "o", 
    "方块": "s", 
    "三角形": "^", 
    "菱形": "D", 
    "五角星": "*", 
    "十字": "x", 
    "加号": "+"
}
marker_style_choice = st.sidebar.selectbox("数据点标记样式", options=list(marker_styles.keys()), index=1)  # 默认选择"圆点"
marker_size_choice = st.sidebar.slider("标记大小", min_value=5, max_value=50, value=20, step=5)

# 使用2x2格式排列参数选择
col1, col2 = st.sidebar.columns(2)

with col1:
    # 字体选择
    font_options = ["Roboto", "Arial", "Times New Roman", "SimHei", "Microsoft YaHei"]
    font_choice = st.sidebar.selectbox("选择字体", font_options)
    
    # 网格颜色选择
    grid_color_options = ["0.1", "0.3", "0.5", "0.7", "0.9"]
    grid_color_choice = st.sidebar.selectbox("选择网格颜色（灰度值）", grid_color_options)
    
    # 布尔值参数选择 - 第一列
    show_ytick_minor = st.sidebar.checkbox("显示Y轴次刻度线", value=True)
    show_grid = st.sidebar.checkbox("显示网格", value=True)

with col2:
    # 字体粗细选择
    weight_options = ["light", "normal", "medium", "semibold", "bold"]
    weight_choice = st.sidebar.selectbox("选择字体粗细", weight_options)
    
    # 字体大小选择
    size_options = [8, 10, 12, 14, 16]
    size_choice = st.sidebar.selectbox("选择字体大小", size_options)
    
    # 网格线宽选择
    grid_linewidth_options = [0.3, 0.5, 0.8, 1.0, 1.5]
    grid_linewidth_choice = st.sidebar.selectbox("选择网格线宽", grid_linewidth_options)
    
    # 布尔值参数选择 - 第二列
    show_xtick_minor = st.sidebar.checkbox("显示X轴次刻度线", value=True)

# 图形尺寸选择 - 单独一行
figsize_options = [(6, 6), (8, 8), (10, 10), (12, 12), (14, 14)]
figsize_index = st.sidebar.selectbox("选择图形尺寸", options=range(len(figsize_options)), 
                            format_func=lambda x: f"{figsize_options[x][0]}x{figsize_options[x][1]}")
figsize_choice = figsize_options[figsize_index]

# 添加图例边框设置
st.sidebar.markdown("### 图例设置")
legend_frame = st.sidebar.checkbox("显示图例边框", value=True)
legend_frame_color_options = {
    "黑色": "black", "灰色": "gray", "白色": "white", "红色": "red", 
    "蓝色": "blue", "绿色": "green", "黄色": "yellow"
}
legend_frame_color = st.sidebar.selectbox("图例边框颜色", options=list(legend_frame_color_options.keys()))
legend_alpha = st.sidebar.slider("图例透明度", min_value=0.0, max_value=1.0, value=0.8, step=0.1)

# 为每个组别设置颜色 - 移到侧边栏
st.sidebar.markdown("### 组别颜色设置")
group_colors = {}
color_options = {
    "蓝色": "blue", "红色": "red", "绿色": "green", "青色": "cyan", 
    "品红": "magenta", "黄色": "yellow", "黑色": "black", "橙色": "orange",
    "紫色": "purple", "棕色": "brown", "粉色": "pink", "灰色": "gray"
}

# 为每个组别创建颜色选择器
for i, group_name in enumerate(groups_data.keys()):
    # 使用更丰富的默认颜色
    default_color = colors[i % len(colors)]
    # 将matplotlib颜色转换为color_options中的颜色名称
    if default_color == '#1f77b4':
        default_index = list(color_options.values()).index('blue')
    elif default_color == '#ff7f0e':
        default_index = list(color_options.values()).index('orange')
    elif default_color == '#2ca02c':
        default_index = list(color_options.values()).index('green')
    elif default_color == '#d62728':
        default_index = list(color_options.values()).index('red')
    elif default_color == '#9467bd':
        default_index = list(color_options.values()).index('purple')
    elif default_color == '#8c564b':
        default_index = list(color_options.values()).index('brown')
    else:
        default_index = 0
        
    color_key = st.sidebar.selectbox(
        f"选择 {group_name} 的颜色",
        options=list(color_options.keys()),
        index=default_index,
        key=f"color_{group_name}"
    )
    group_colors[group_name] = color_options[color_key]

# 保存图形选项 - 移到侧边栏
st.sidebar.markdown("### 保存设置")
format_options = ["png", "jpg", "svg", "pdf", "tif"]
format_choice = st.sidebar.selectbox("选择保存格式", format_options)

# 根据图像格式类型决定是否显示分辨率选项
if format_choice in ["png", "jpg", "tif"]:  # 位图格式
    resolution_choice = st.sidebar.slider("选择分辨率 (dpi)", min_value=72, max_value=1000, value=150, step=10)
else:  # 矢量图格式
    resolution_choice = None
    st.sidebar.info("矢量图格式(SVG/PDF)不需要设置分辨率参数")

# 使用文本输入来选择保存路径
save_directory = st.sidebar.text_input("输入保存文件夹路径", value=output_dir)
save_filename = st.sidebar.text_input("输入保存文件名", value=f'雷达图.{format_choice}')
save_path = os.path.join(save_directory, save_filename)

# 将雷达图绘制代码封装成函数
def create_radar_chart(categories, groups_data, group_colors, radar_style_choice, figsize_choice, 
                      grid_color_choice, grid_linewidth_choice, show_grid, alpha=0.1,
                      marker_style="o", marker_size=20, legend_frame=True, 
                      legend_frame_color="black", legend_alpha=0.8):
    """
    创建雷达图
    
    参数:
    categories - 类别列表（已闭合）
    groups_data - 组别数据字典，格式为 {组名: [数据值]}
    group_colors - 组别颜色字典，格式为 {组名: 颜色}
    radar_style_choice - 雷达图样式选择 ("标准网格" 或 "多边形网格")
    figsize_choice - 图表尺寸元组 (宽, 高)
    grid_color_choice - 网格颜色
    grid_linewidth_choice - 网格线宽
    show_grid - 是否显示网格
    alpha - 填充透明度
    marker_style - 数据点标记样式
    marker_size - 数据点标记大小
    legend_frame - 是否显示图例边框
    legend_frame_color - 图例边框颜色
    legend_alpha - 图例透明度
    
    返回:
    fig - matplotlib图形对象
    ax - matplotlib坐标轴对象
    """
    # 图形初始化
    fig = plt.figure(figsize=figsize_choice)
    
    # 设置雷达图网格样式
    if radar_style_choice == "多边形网格":
        # 使用直角坐标系创建多边形雷达图
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 计算角度
        num_vars = len(categories) - 1  # 减去闭合点
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        
        # 计算数据范围，用于动态调整刻度
        all_values = []
        for values in groups_data.values():
            all_values.extend(values[:-1])  # 不包括闭合点
        
        data_min = min(all_values)
        data_max = max(all_values)
        
        # 动态确定刻度范围
        # 根据数据范围确定合适的最大刻度值
        if data_max <= 5:
            scale_max = 5
        elif data_max <= 10:
            scale_max = 10
        elif data_max <= 25:
            scale_max = 25
        elif data_max <= 50:
            scale_max = 50
        elif data_max <= 100:
            scale_max = 100
        elif data_max <= 200:
            scale_max = 200
        elif data_max <= 500:
            scale_max = 500
        else:
            scale_max = 1000
            
        # 创建多边形网格 - 传递刻度最大值和网格参数
        create_polygon_grid(ax, angles, categories, num_vars, max_radius=1.0, scale_max=scale_max,
                           grid_color=grid_color_choice, grid_linewidth=grid_linewidth_choice, 
                           show_grid=show_grid, font_size=size_choice)
        
        # 绘制所有数据组
        for group_name, values in groups_data.items():
            # 使用用户选择的颜色
            color = group_colors.get(group_name, "blue")
            
            # 去掉闭合点进行计算
            values_without_last = values[:-1]
            
            # 根据刻度最大值缩放数据
            scaled_values = [(np.sin(theta)*val/scale_max, np.cos(theta)*val/scale_max) 
                           for theta, val in zip(angles, values_without_last)]
            
            # 绘制多边形 - 使用与标准雷达图一致的样式
            poly = plt.Polygon(scaled_values, closed=True, alpha=alpha, 
                              edgecolor=color, facecolor=color, lw=2)
            ax.add_patch(poly)
            
            # 添加数据点和连线 - 与标准雷达图保持一致
            prev_point = None
            points_x = []
            points_y = []
            
            for i, (theta, val) in enumerate(zip(angles, values_without_last)):
                x = np.sin(theta) * val/scale_max
                y = np.cos(theta) * val/scale_max
                points_x.append(x)
                points_y.append(y)
                
                # 根据用户选择的标记样式绘制数据点
                if marker_style != "none":
                    ax.scatter(x, y, color=color, s=marker_size, marker=marker_style, zorder=10)
                
                # 如果有前一个点，绘制连线
                if prev_point is not None:
                    ax.plot([prev_point[0], x], [prev_point[1], y], color=color, linewidth=2, zorder=9)
                prev_point = (x, y)
            
            # 连接最后一个点和第一个点，形成闭环
            first_point = (np.sin(angles[0]) * values_without_last[0]/scale_max, np.cos(angles[0]) * values_without_last[0]/scale_max)
            ax.plot([prev_point[0], first_point[0]], [prev_point[1], first_point[1]], color=color, linewidth=2, zorder=9)
            
            # 为图例添加一个单独的线条 - 确保与标准雷达图的图例一致
            ax.plot([], [], color=color, linewidth=2, label=group_name)
        
        # 设置显示范围
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # 添加图例 - 与标准雷达图保持一致，但应用边框设置
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), 
                  frameon=legend_frame, edgecolor=legend_frame_color, 
                  framealpha=legend_alpha)
        
    else:
        # 使用标准极坐标雷达图
        ax = fig.add_subplot(projection='polar')
        
        # 角度计算（将分类转换为弧度）
        label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
        
        # 使用标准网格
        ax.grid(show_grid, color=grid_color_choice, linewidth=grid_linewidth_choice)
        
        # 绘制所有数据组
        for group_name, values in groups_data.items():
            # 使用用户选择的颜色
            color = group_colors.get(group_name, "blue")
            plot_radar_data(ax, label_loc, values, color, group_name, alpha, marker_style, marker_size)
        
        lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
        # 修改这里，应用图例边框设置
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), 
                  frameon=legend_frame, edgecolor=legend_frame_color, 
                  framealpha=legend_alpha)
    
    return fig, ax

def create_polygon_grid(ax, angles, categories, num_vars, max_radius, scale_max=5, 
                       grid_color="0.5", grid_linewidth=0.5, show_grid=True, font_size=10):
    """
    创建多边形网格系统
    
    参数:
    ax - matplotlib坐标轴对象
    angles - 角度列表
    categories - 类别列表
    num_vars - 变量数量
    max_radius - 最大半径
    scale_max - 刻度最大值
    grid_color - 网格颜色
    grid_linewidth - 网格线宽
    show_grid - 是否显示网格
    font_size - 字体大小
    """
    # 主多边形框架
    poly_verts = [(np.sin(theta)*max_radius, np.cos(theta)*max_radius) 
                 for theta in angles]
    poly = plt.Polygon(poly_verts, closed=True, fill=False, color='black', lw=1.0)
    ax.add_patch(poly)
    
    # 根据刻度最大值动态生成刻度
    if scale_max <= 5:
        tick_values = [1, 2, 3, 4, 5]
        tick_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif scale_max <= 10:
        tick_values = [2, 4, 6, 8, 10]
        tick_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif scale_max <= 25:
        tick_values = [5, 10, 15, 20, 25]
        tick_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif scale_max <= 50:
        tick_values = [10, 20, 30, 40, 50]
        tick_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif scale_max <= 100:
        tick_values = [20, 40, 60, 80, 100]
        tick_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif scale_max <= 200:
        tick_values = [40, 80, 120, 160, 200]
        tick_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    elif scale_max <= 500:
        tick_values = [100, 200, 300, 400, 500]
        tick_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        tick_values = [200, 400, 600, 800, 1000]
        tick_positions = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # 径向刻度线 - 应用网格参数
    if show_grid:
        for r in tick_positions:
            scaled_verts = [(x*r, y*r) for (x,y) in poly_verts]
            poly = plt.Polygon(scaled_verts, closed=True, fill=False, 
                          color=grid_color, lw=grid_linewidth, ls='-')
            ax.add_patch(poly)
        
        # 添加中心与节点的连线 - 应用网格参数
        for theta in angles:
            x = np.sin(theta) * max_radius
            y = np.cos(theta) * max_radius
            ax.plot([0, x], [0, y], color=grid_color, lw=grid_linewidth, ls='-')
    
    # 轴标签和刻度（改进标签位置和样式）
    label_radius = max_radius * 1.15  # 增加标签距离
    for i, (theta, label) in enumerate(zip(angles, categories[:-1])):  # 不包括闭合点
        x = np.sin(theta) * label_radius
        y = np.cos(theta) * label_radius
        
        # 调整标签对齐方式
        if x < 0:
            ha = 'right'
        elif x == 0:
            ha = 'center'
        else:
            ha = 'left'
            
        ax.text(x, y, label, 
                ha=ha, va='center',
                fontsize=font_size)  # 使用传入的字体大小
    
    # 径向刻度标签（改进刻度值显示）
    for i, r in enumerate(tick_positions):
        # 在45度角位置添加刻度值
        x = np.sin(np.pi/4) * r * max_radius
        y = np.cos(np.pi/4) * r * max_radius
        
        # 显示整数刻度值
        actual_value = tick_values[i]
        value_str = f'{actual_value}'
            
        ax.text(x, y, value_str, 
                ha='left', va='bottom',
                fontsize=font_size-2, color='grey')  # 刻度值字体略小

def plot_radar_data(ax, angles, values, color, label, alpha=0.1, marker_style="o", marker_size=20):
    """
    在雷达图上绘制一组数据
    
    参数:
    ax - matplotlib坐标轴对象
    angles - 角度列表
    values - 数据值列表
    color - 线条颜色
    label - 图例标签
    alpha - 填充透明度
    marker_style - 数据点标记样式
    marker_size - 数据点标记大小
    """
    ax.plot(angles, values, color=color, linewidth=2, label=label)
    ax.fill(angles, values, color=color, alpha=alpha)
    
    # 添加数据点标记
    if marker_style != "none":
        ax.scatter(angles, values, color=color, s=marker_size, marker=marker_style, zorder=10)

# 应用选择的参数
p = plt.rcParams
p["font.sans-serif"] = [font_choice]
p["font.weight"] = weight_choice
p["font.size"] = size_choice
p["ytick.minor.visible"] = show_ytick_minor
p["xtick.minor.visible"] = show_xtick_minor
p["axes.grid"] = show_grid
p["grid.color"] = grid_color_choice
p["grid.linewidth"] = grid_linewidth_choice

# 使用封装的函数创建雷达图
fig, ax = create_radar_chart(
    categories=categories,
    groups_data=groups_data,
    group_colors=group_colors,
    radar_style_choice=radar_style_choice,
    figsize_choice=figsize_choice,
    grid_color_choice=grid_color_choice,
    grid_linewidth_choice=grid_linewidth_choice,
    show_grid=show_grid,
    alpha=0.1,
    marker_style=marker_styles[marker_style_choice],
    marker_size=marker_size_choice,
    legend_frame=legend_frame,
    legend_frame_color=legend_frame_color_options[legend_frame_color],
    legend_alpha=legend_alpha
)

# 显示图形
st.pyplot(fig)

# 保存图形按钮的处理
if st.sidebar.button("保存图形"):
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)
    
    # 根据是否有分辨率参数来调用不同的保存方法
    if resolution_choice is not None:
        fig.savefig(save_path, format=format_choice, dpi=resolution_choice, bbox_inches='tight')
    else:
        fig.savefig(save_path, format=format_choice, bbox_inches='tight')
    
    st.success(f"图形已保存到 {save_path}")